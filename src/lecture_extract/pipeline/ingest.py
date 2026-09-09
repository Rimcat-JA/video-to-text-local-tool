"""取り込み (設計 11: ingest)。

元動画のファイル自体は変更しない。読み取りのみ。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..config import RunConfig
from ..db.store import Store, new_id
from ..models import CoverageSpan, ReviewItem
from ..util.hashing import sha256_file
from .probe import MediaProbe, check_frame_timestamps, probe_media

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".ts", ".flv", ".wmv", ".mpg", ".mpeg"}


class IngestError(RuntimeError):
    pass


def ingest(store: Store, cfg: RunConfig) -> tuple[str, MediaProbe]:
    path = Path(cfg.input_path)
    if not path.exists():
        raise IngestError(f"入力ファイルが見つかりません: {path}")
    if not path.is_file():
        raise IngestError(f"入力がファイルではありません: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        log.warning("未知の拡張子です (%s)。ffprobe の判定に従って続行します。", path.suffix)

    log.info("入力のハッシュを計算しています: %s", path.name)
    digest = sha256_file(path)
    probe = probe_media(path)
    if probe.video is None:
        raise IngestError("映像ストリームがありません。画面抽出の対象になりません。")
    if probe.duration_us <= 0:
        raise IngestError("再生時間を確定できませんでした。")

    existing = store.find_media_by_sha(digest)
    media_id = existing["id"] if existing else new_id("med")

    store.upsert_media(
        {
            "id": media_id,
            "source_path": str(path.resolve()),
            "sha256": digest,
            "size_bytes": path.stat().st_size,
            "duration_us": probe.duration_us,
            "container": probe.container,
            "stream_info": probe.to_dict(),
            "time_origin_us": probe.time_origin_us,
            "video_start_us": probe.video_offset_us,
            "audio_start_us": probe.audio_offset_us,
            "width": probe.video.width,
            "height": probe.video.height,
            "nominal_fps": probe.nominal_fps,
            "is_vfr": int(probe.is_vfr),
            "timeline_flags": probe.flags,
        }
    )

    ts_check = check_frame_timestamps(path)
    log.info("タイムスタンプ検査: %s", ts_check)
    store.clear_reviews_by_reason_prefix(media_id, "timeline:")
    if ts_check["missing_timestamps"]:
        store.add_review(
            ReviewItem(
                id=new_id("rev"),
                media_id=media_id,
                target_kind="media",
                target_ref=media_id,
                reason="timeline:missing_timestamps",
                detail=f"先頭 {ts_check['checked_frames']} フレーム中 {ts_check['missing_timestamps']} 件で時刻が取得できません。",
            )
        )
    if ts_check["non_monotonic"]:
        store.add_review(
            ReviewItem(
                id=new_id("rev"),
                media_id=media_id,
                target_kind="media",
                target_ref=media_id,
                reason="timeline:non_monotonic_pts",
                detail=f"時刻が戻るフレームが {ts_check['non_monotonic']} 件あります。",
            )
        )
    if ts_check["large_gaps"]:
        store.add_review(
            ReviewItem(
                id=new_id("rev"),
                media_id=media_id,
                target_kind="media",
                target_ref=media_id,
                reason="timeline:pts_gaps",
                detail=f"中央値の 3 倍を超える時刻の飛びが {ts_check['large_gaps']} 件あります。",
            )
        )
    if probe.is_vfr:
        store.add_review(
            ReviewItem(
                id=new_id("rev"),
                media_id=media_id,
                target_kind="media",
                target_ref=media_id,
                reason="timeline:vfr_suspected",
                detail="可変フレームレートの疑いがあります。時刻は PTS から求めます。",
            )
        )

    # 設計 9.2: 未抽出区間も含めて時間軸の状態を必ず記録する。
    start, end = analysis_range(cfg, probe.duration_us)
    spans: list[CoverageSpan] = []
    if start > 0:
        spans.append(
            CoverageSpan(new_id("cov"), media_id, "screen", 0, start, "out_of_range", "解析範囲外")
        )
    spans.append(CoverageSpan(new_id("cov"), media_id, "screen", start, end, "unextracted", "未解析"))
    if end < probe.duration_us:
        spans.append(
            CoverageSpan(new_id("cov"), media_id, "screen", end, probe.duration_us, "out_of_range", "解析範囲外")
        )
    store.replace_coverage(media_id, "screen", spans)

    audio_spans: list[CoverageSpan] = []
    if probe.audio is None:
        audio_spans.append(
            CoverageSpan(new_id("cov"), media_id, "audio", 0, probe.duration_us, "unextracted", "音声ストリームなし")
        )
    else:
        if start > 0:
            audio_spans.append(
                CoverageSpan(new_id("cov"), media_id, "audio", 0, start, "out_of_range", "解析範囲外")
            )
        audio_spans.append(CoverageSpan(new_id("cov"), media_id, "audio", start, end, "unextracted", "未解析"))
        if end < probe.duration_us:
            audio_spans.append(
                CoverageSpan(
                    new_id("cov"), media_id, "audio", end, probe.duration_us, "out_of_range", "解析範囲外"
                )
            )
    store.replace_coverage(media_id, "audio", audio_spans)

    return media_id, probe


def analysis_range(cfg: RunConfig, duration_us: int) -> tuple[int, int]:
    """解析対象の区間。切り出しても、出力は元動画に対する時刻で表す (設計 4.1)。"""
    start = max(0, cfg.range_start_us or 0)
    end = min(duration_us, cfg.range_end_us if cfg.range_end_us is not None else duration_us)
    if end <= start:
        raise IngestError(f"解析範囲が空です: [{start}, {end})")
    return start, end


def input_hash(cfg: RunConfig, digest: str) -> str:
    """ステージの入力同一性。元動画の内容と解析範囲で決まる。"""
    from ..util.hashing import config_hash

    return config_hash(
        {"sha256": digest, "range": [cfg.range_start_us, cfg.range_end_us]}
    )
