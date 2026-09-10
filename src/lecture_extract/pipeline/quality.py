"""品質報告と manifest (設計 10 / 13.2 / 14)。

品質報告は「処理完了」と「全文検証済み」を区別して書く。
達成済みでない目標値を、達成したかのように表示しない。
"""

from __future__ import annotations

import json
import logging
import platform
import sys
from collections import Counter, defaultdict
from importlib import metadata
from pathlib import Path
from typing import Any

from ..config import RunConfig
from ..db.store import Store
from ..models import FLAG_NOT_EXTRACTED, FLAG_TRUNCATED, FLAG_UNREADABLE
from ..util.atomic import atomic_write_text
from ..util.timeutil import format_timestamp
from .exporter import OUTPUT_FORMAT_VERSION
from .ffmpeg_tools import ffmpeg_path, ffprobe_path, tool_version

log = logging.getLogger(__name__)

# 設計 14.4
STATUS_PROCESSING = "processing"
STATUS_COMPLETED_WITH_REVIEW = "completed_with_review"
STATUS_REVIEWED = "reviewed"
STATUS_FAILED = "failed"

_TRACKED_PACKAGES = ["av", "opencv-python-headless", "opencv-python", "numpy", "pillow", "requests"]


def completion_status(store: Store, media_id: str, all_stages_done: bool) -> str:
    if not all_stages_done:
        return STATUS_PROCESSING
    open_items = store.reviews(media_id, status="open")
    return STATUS_COMPLETED_WITH_REVIEW if open_items else STATUS_REVIEWED


def collect_quality(store: Store, media_id: str, duration_us: int) -> dict[str, Any]:
    occurrences = store.occurrences(media_id)
    utterances = store.utterances(media_id)
    contents = {c.id: c for c in store.all_contents()}
    coverage = store.coverage(media_id)
    reviews = store.reviews(media_id)
    alignments = store.alignments(media_id)

    coverage_by_track: dict[str, Counter] = defaultdict(Counter)
    for span in coverage:
        coverage_by_track[span.track][span.state] += max(0, span.end_us - span.start_us)

    unextracted_spans = [s for s in coverage if s.track == "screen" and s.state in ("unextracted", "failed")]
    unreadable_regions = 0
    truncated_contents = 0
    for content in contents.values():
        if FLAG_TRUNCATED in content.quality_flags:
            truncated_contents += 1
        for region in content.regions:
            if region.unreadable or FLAG_UNREADABLE in region.flags:
                unreadable_regions += 1

    aligned_utts = {a.utterance_id for a in alignments}
    time_anomalies = []
    for occ in occurrences:
        if occ.end_us < occ.start_us:
            time_anomalies.append(f"occurrence {occ.id}: end < start")
        if occ.start_us < 0 or occ.end_us > duration_us:
            time_anomalies.append(f"occurrence {occ.id}: 動画範囲外")
    for utt in utterances:
        if utt.end_us < utt.start_us:
            time_anomalies.append(f"utterance {utt.id}: end < start")
        if utt.start_us < 0 or utt.end_us > duration_us + 1_000_000:
            time_anomalies.append(f"utterance {utt.id}: 動画範囲外")

    short_states = [o for o in occurrences if o.state_kind in ("short", "transition")]
    short_unextracted = [o for o in short_states if not o.content_id]

    return {
        "occurrences": len(occurrences),
        "occurrences_extracted": sum(1 for o in occurrences if o.content_id),
        "occurrences_unextracted": sum(1 for o in occurrences if not o.content_id),
        "unique_contents": len(contents),
        "utterances": len(utterances),
        "utterances_unaligned": sum(1 for u in utterances if u.id not in aligned_utts),
        "alignments": len(alignments),
        "unreadable_regions": unreadable_regions,
        "truncated_contents": truncated_contents,
        "short_states": len(short_states),
        "short_states_unextracted": len(short_unextracted),
        "time_anomalies": time_anomalies,
        "coverage_us": {track: dict(counter) for track, counter in coverage_by_track.items()},
        "unextracted_spans": [(s.start_us, s.end_us, s.detail) for s in unextracted_spans],
        "reviews_open": [r for r in reviews if r.review_status == "open"],
        "reviews_total": len(reviews),
        "attempt_stats": store.attempt_stats(),
        "metrics": store.metrics(media_id),
    }


def write_quality_report(
    store: Store, cfg: RunConfig, media: dict[str, Any], status: str, verification_scope: str
) -> Path:
    media_id = media["id"]
    duration_us = media["duration_us"]
    q = collect_quality(store, media_id, duration_us)
    path = Path(cfg.out_dir) / "quality_report.md"

    lines: list[str] = []
    lines.append("# 品質報告")
    lines.append("")
    lines.append(f"- 完了状態: **{status}**")
    lines.append(f"- 検証範囲: {verification_scope}")
    lines.append(
        "- 注意: この状態は、元動画全体との完全一致を保証するものではありません。"
        "処理が終了したことと、全文が検証済みであることは別です（設計 14.4）。"
    )
    lines.append("")

    lines.append("## 時間軸の網羅")
    lines.append("")
    lines.append("| トラック | 状態 | 合計時間 |")
    lines.append("|---|---|---|")
    for track, states in sorted(q["coverage_us"].items()):
        for state, total in sorted(states.items()):
            lines.append(f"| {track} | {state} | {format_timestamp(total)} |")
    unknown = _unknown_coverage(store, media_id, duration_us)
    lines.append("")
    lines.append(f"- 処理状態が記録されていない区間: **{len(unknown)} 件**")
    for start, end in unknown[:20]:
        lines.append(f"  - {format_timestamp(start)} – {format_timestamp(end)}")
    lines.append("")

    lines.append("## 未抽出区間")
    lines.append("")
    if not q["unextracted_spans"]:
        lines.append("- なし")
    else:
        lines.append(f"- 件数: {len(q['unextracted_spans'])}")
        for start, end, detail in q["unextracted_spans"][:50]:
            lines.append(f"  - {format_timestamp(start)} – {format_timestamp(end)}: {detail}")
        if len(q["unextracted_spans"]) > 50:
            lines.append(f"  - ...ほか {len(q['unextracted_spans']) - 50} 件")
    lines.append("")
    lines.append(
        f"- 短時間表示・遷移として抽出対象外にした状態: {q['short_states']} 件"
        f"（うち全文未確定 {q['short_states_unextracted']} 件）"
    )
    lines.append("  これらは期間と境界だけを記録しています。個々の期間は screen_occurrences.jsonl にあります。")
    lines.append("  短時間表示は別集計です。未回収・未認識を隠していません（設計 14.2）。")
    lines.append("")

    lines.append("## 不明文字・切り詰め")
    lines.append("")
    lines.append(f"- 判読不能を含む領域: {q['unreadable_regions']} 件")
    lines.append(f"- 出力が長さ制限で切れた画面本文: {q['truncated_contents']} 件")
    lines.append(f"- モデル応答の状態内訳: {q['attempt_stats'] or 'なし'}")
    lines.append("")

    lines.append("## 時刻異常")
    lines.append("")
    if not q["time_anomalies"]:
        lines.append("- なし（終了が開始より前になる区間、動画範囲外の区間はありません）")
    else:
        for item in q["time_anomalies"][:50]:
            lines.append(f"- {item}")
    lines.append(f"- どの表示期間にも重ならない発話: {q['utterances_unaligned']} 件")
    lines.append("")

    lines.append("## 要確認項目")
    lines.append("")
    open_items = q["reviews_open"]
    if not open_items:
        lines.append("- なし")
    else:
        by_reason: Counter = Counter(r.reason for r in open_items)
        lines.append("| 理由 | 件数 |")
        lines.append("|---|---|")
        for reason, count in by_reason.most_common():
            lines.append(f"| {reason} | {count} |")
        lines.append("")
        lines.append("### 先頭 50 件")
        lines.append("")
        for item in open_items[:50]:
            lines.append(f"- `{item.reason}` {item.target_kind}={item.target_ref}: {item.detail}")
    lines.append("")

    lines.append("## 性能計測")
    lines.append("")
    lines.append("| ステージ | 指標 | 値 |")
    lines.append("|---|---|---|")
    for metric in q["metrics"]:
        value = metric["value"]
        shown = f"{value:.3f}" if abs(value - int(value)) > 1e-9 else str(int(value))
        lines.append(f"| {metric['stage']} | {metric['name']} | {shown}{metric['unit']} |")
    lines.append("")

    lines.append("## 評価指標について")
    lines.append("")
    lines.append(
        "CER / WER / 表示状態の回収率は、人が作成した正解データが必要です。"
        "本報告は正解データ無しで得られる観測値だけを載せています。"
        "検出された候補だけを分母にした回収率は掲載しません（設計 14.2）。"
    )
    lines.append("")

    atomic_write_text(path, "\n".join(lines) + "\n")
    return path


def _unknown_coverage(store: Store, media_id: str, duration_us: int) -> list[tuple[int, int]]:
    """処理状態が記録されていない区間を求める (設計 14.2: 0 件が目標)。"""
    gaps: list[tuple[int, int]] = []
    for track in ("screen", "audio"):
        spans = sorted(store.coverage(media_id, track), key=lambda s: s.start_us)
        cursor = 0
        for span in spans:
            if span.start_us > cursor:
                gaps.append((cursor, span.start_us))
            cursor = max(cursor, span.end_us)
        if cursor < duration_us:
            gaps.append((cursor, duration_us))
    return gaps


def write_manifest(
    store: Store,
    cfg: RunConfig,
    media: dict[str, Any],
    vision_info: dict[str, Any],
    asr_info: dict[str, Any],
    status: str,
    stage_results: dict[str, Any],
) -> Path:
    """設計 13.2: 再現可能な環境の記録。"""
    path = Path(cfg.out_dir) / "manifest.json"
    packages = {}
    for name in _TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    manifest = {
        "output_format_version": OUTPUT_FORMAT_VERSION,
        "status": status,
        "input": {
            "source_path": media["source_path"],
            "sha256": media["sha256"],
            "size_bytes": media["size_bytes"],
            "duration_us": media["duration_us"],
            "container": media["container"],
            "time_origin_us": media["time_origin_us"],
            "video_start_us": media["video_start_us"],
            "audio_start_us": media["audio_start_us"],
            "is_vfr": bool(media["is_vfr"]),
            "timeline_flags": media["timeline_flags"],
        },
        "models": {
            "vision": vision_info,
            "asr": asr_info,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": packages,
            "ffmpeg": tool_version(ffmpeg_path),
            "ffprobe": tool_version(ffprobe_path),
        },
        "config": cfg.to_dict(),
        "stage_config_hashes": {
            stage: cfg.stage_config_hash(stage)
            for stage in [
                "ingest",
                "timeline",
                "frame_scan",
                "audio_prepare",
                "asr",
                "vision",
                "align",
                "blocks",
                "export",
            ]
        },
        "stage_results": stage_results,
        "jobs": [
            {k: v for k, v in job.items() if k in ("stage", "status", "input_hash", "config_hash", "started_at", "finished_at", "error")}
            for job in store.jobs()
        ],
        "licenses_note": (
            "採用したソフトウェアとモデルのライセンス情報は docs/LICENSES.md に記載しています。"
        ),
    }
    atomic_write_text(path, json.dumps(manifest, ensure_ascii=False, indent=2))
    return path
