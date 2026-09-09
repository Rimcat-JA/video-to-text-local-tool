"""発話の抽出 (設計 7)。

- チャンク内の時刻を元動画の時刻へ戻してから統合する。
- 重複範囲内にある同一発話だけを整理し、講師が実際に繰り返した発言は残す。
- 無音区間に発話が生成された場合は削除せず、確認対象として残す (設計 14.3)。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..config import RunConfig
from ..db.store import Store, new_id
from ..models import CoverageSpan, ReviewItem, Utterance
from ..util.textnorm import normalize_for_dedup_utterance
from ..util.timeutil import overlap_us
from .audio_prepare import (
    AudioChunk,
    detect_silence,
    extract_master_wav,
    make_chunks,
    verify_audio_alignment,
)
from .probe import MediaProbe

log = logging.getLogger(__name__)


def run_asr(
    store: Store,
    cfg: RunConfig,
    probe: MediaProbe,
    adapter,
    media_id: str,
    job_id: str,
    start_us: int,
    end_us: int,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "chunks": 0,
        "chunks_done": 0,
        "segments_raw": 0,
        "utterances": 0,
        "overlap_dropped": 0,
        "duplicate_dropped": 0,
        "failed_chunks": 0,
        "silence_utterances": 0,
    }
    if probe.audio is None:
        log.warning("音声ストリームがありません。発話抽出を行いません。")
        store.replace_coverage(
            media_id,
            "audio",
            [CoverageSpan(new_id("cov"), media_id, "audio", 0, probe.duration_us, "unextracted", "音声ストリームなし")],
        )
        return stats

    master = extract_master_wav(cfg, probe, media_id)
    alignment_check = verify_audio_alignment(probe, master)
    if alignment_check["mismatch"]:
        store.add_review(
            ReviewItem(
                id=new_id("rev"),
                media_id=media_id,
                target_kind="media",
                target_ref=media_id,
                reason="asr:audio_length_mismatch",
                detail=f"書き出した音声とコンテナの音声長が {alignment_check['diff_us'] / 1000:.0f}ms ずれています。",
            )
        )

    chunks = make_chunks(cfg, probe, media_id, master, start_us, end_us)
    stats["chunks"] = len(chunks)
    silence_spans = detect_silence(master, probe.audio_offset_us)

    job = store.get_job(media_id, "asr")
    done_chunks = set(job["checkpoint"].get("done_chunks", [])) if (job and resume) else set()
    if not done_chunks:
        store.clear_reviews_by_reason_prefix(media_id, "asr:")

    out_dir = Path(cfg.work_dir) / "asr" / media_id
    out_dir.mkdir(parents=True, exist_ok=True)

    kept: list[Utterance] = []
    for chunk in chunks:
        if chunk.chunk_id in done_chunks:
            stats["chunks_done"] += 1
            continue
        store.delete_utterances_for_chunk(media_id, chunk.chunk_id)
        result = None
        attempts = 0
        while attempts <= cfg.asr.retry_limit:
            attempts += 1
            result = adapter.transcribe(
                chunk.path,
                out_prefix=str(out_dir / chunk.chunk_id),
                chunk_start_us=chunk.start_us,
            )
            if result.ok:
                break
            log.warning("チャンク %s の認識に失敗しました (%d 回目): %s", chunk.chunk_id, attempts, result.error)
        assert result is not None
        if not result.ok:
            stats["failed_chunks"] += 1
            store.add_review(
                ReviewItem(
                    id=new_id("rev"),
                    media_id=media_id,
                    target_kind="audio_chunk",
                    target_ref=chunk.chunk_id,
                    reason="asr:chunk_failed",
                    detail=f"[{chunk.start_us}, {chunk.end_us}) の認識に失敗: {result.error}",
                )
            )
            continue

        stats["segments_raw"] += len(result.segments)
        chunk_utts: list[Utterance] = []
        for seg in result.segments:
            abs_start = chunk.start_us + seg.start_us
            abs_end = chunk.start_us + seg.end_us
            if abs_end < abs_start:
                abs_start, abs_end = abs_end, abs_start
            midpoint = (abs_start + abs_end) // 2
            # 重複範囲の担当を一意に決める。原時刻は変えない。
            if not (chunk.core_start_us <= midpoint < chunk.core_end_us):
                if not (chunk.index == 0 and midpoint < chunk.core_start_us) and not (
                    chunk.index == len(chunks) - 1 and midpoint >= chunk.core_end_us
                ):
                    stats["overlap_dropped"] += 1
                    continue
            tokens = [
                {
                    "text": t["text"],
                    "start_us": chunk.start_us + t["start_us"],
                    "end_us": chunk.start_us + t["end_us"],
                    "experimental": True,
                }
                for t in seg.tokens
            ]
            utt = Utterance(
                id=new_id("utt"),
                media_id=media_id,
                start_us=abs_start,
                end_us=abs_end,
                text_raw=seg.text,
                language=seg.language or result.language,
                tokens=tokens,
                quality_flags=list(seg.flags),
                source="asr",
                chunk_id=chunk.chunk_id,
            )
            chunk_utts.append(utt)

        # チャンクを完了として記録する前に、その発話を保存する。
        # 先にチェックポイントだけ進めると、中断時に「処理済みなのに発話が無い」
        # 区間ができ、再開しても復元されない。
        for utt in chunk_utts:
            store.upsert_utterance(utt)
        kept.extend(chunk_utts)

        done_chunks.add(chunk.chunk_id)
        stats["chunks_done"] += 1
        store.update_checkpoint(job_id, {"done_chunks": sorted(done_chunks)})

    # 既存 (再開前に保存済み) の発話も含めて、重複整理と保存を行う。
    existing = store.utterances(media_id)
    all_utts = {u.id: u for u in existing}
    for utt in kept:
        all_utts[utt.id] = utt
    ordered = sorted(all_utts.values(), key=lambda u: (u.start_us, u.end_us))

    deduped: list[Utterance] = []
    for utt in ordered:
        duplicate_of = None
        for prev in reversed(deduped[-8:]):
            if prev.chunk_id == utt.chunk_id:
                continue
            if overlap_us(prev.start_us, prev.end_us, utt.start_us, utt.end_us) <= 0:
                continue
            if normalize_for_dedup_utterance(prev.text_raw) == normalize_for_dedup_utterance(utt.text_raw):
                duplicate_of = prev
                break
        if duplicate_of is not None:
            stats["duplicate_dropped"] += 1
            continue
        deduped.append(utt)

    for utt in ordered:
        if utt not in deduped:
            store.conn.execute("DELETE FROM utterance WHERE id = ?", (utt.id,))
    for utt in deduped:
        if _inside_silence(utt, silence_spans):
            if "generated_in_silence" not in utt.quality_flags:
                utt.quality_flags = sorted(set(utt.quality_flags) | {"generated_in_silence"})
            stats["silence_utterances"] += 1
            store.add_review(
                ReviewItem(
                    id=new_id("rev"),
                    media_id=media_id,
                    target_kind="utterance",
                    target_ref=utt.id,
                    reason="asr:utterance_in_silence",
                    detail="無音として検出した区間に発話が出力されています。削除せず確認対象として残しています。",
                )
            )
        store.upsert_utterance(utt)

    stats["utterances"] = len(deduped)
    store.update_checkpoint(job_id, {"done_chunks": sorted(done_chunks)})
    _update_audio_coverage(store, media_id, probe, chunks, deduped, silence_spans, start_us, end_us)
    for name, value in stats.items():
        store.add_metric(media_id, "asr", name, float(value))
    log.info("asr 完了: %s", stats)
    return stats


def _inside_silence(utt: Utterance, silence_spans: list[tuple[int, int]]) -> bool:
    duration = max(1, utt.end_us - utt.start_us)
    covered = sum(overlap_us(utt.start_us, utt.end_us, s, e) for s, e in silence_spans)
    return covered / duration > 0.9


def _update_audio_coverage(
    store: Store,
    media_id: str,
    probe: MediaProbe,
    chunks: list[AudioChunk],
    utterances: list[Utterance],
    silence_spans: list[tuple[int, int]],
    start_us: int,
    end_us: int,
) -> None:
    """音声側の時間軸を、処理状態が不明な区間が残らないように記録する (設計 9.2)。"""
    spans: list[CoverageSpan] = []
    if start_us > 0:
        spans.append(CoverageSpan(new_id("cov"), media_id, "audio", 0, start_us, "out_of_range", "解析範囲外"))
    processed: list[tuple[int, int]] = [
        (c.core_start_us, c.core_end_us) for c in chunks
    ]
    processed.sort()
    cursor = start_us
    for c_start, c_end in processed:
        if c_start > cursor:
            spans.append(
                CoverageSpan(new_id("cov"), media_id, "audio", cursor, c_start, "unextracted", "認識していない区間")
            )
        spans.append(
            CoverageSpan(new_id("cov"), media_id, "audio", c_start, c_end, "extracted", "認識済み")
        )
        cursor = max(cursor, c_end)
    if cursor < end_us:
        spans.append(CoverageSpan(new_id("cov"), media_id, "audio", cursor, end_us, "unextracted", "認識していない区間"))
    if end_us < probe.duration_us:
        spans.append(
            CoverageSpan(new_id("cov"), media_id, "audio", end_us, probe.duration_us, "out_of_range", "解析範囲外")
        )
    for s, e in silence_spans:
        spans.append(CoverageSpan(new_id("cov"), media_id, "audio_silence", s, e, "silence", "無音として検出"))
    store.replace_coverage(media_id, "audio", [s for s in spans if s.track == "audio"])
    store.replace_coverage(media_id, "audio_silence", [s for s in spans if s.track == "audio_silence"])
