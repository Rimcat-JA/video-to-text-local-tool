"""時刻基準の確定 (設計 4.1)。

- 原データとしてストリームの time_base と start_time を保持する。
- t0 は「入力メディアで採用した再生開始原点」として明示的に保存する。
- 映像・音声それぞれの開始オフセットを保存する。
- 可変フレームレートの疑いを検出し、フラグとして残す。
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from ..util.timeutil import seconds_to_us
from .ffmpeg_tools import ffprobe_json

log = logging.getLogger(__name__)

# 可変フレームレートの疑いを立てる比率のずれ
_VFR_RATIO_TOLERANCE = 0.01


def _parse_rate(value: str | None) -> float | None:
    if not value or value in ("0/0", "N/A"):
        return None
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None


def _parse_time_base(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    try:
        frac = Fraction(value)
        return frac.numerator, frac.denominator
    except (ValueError, ZeroDivisionError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        if value in (None, "N/A"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class StreamInfo:
    index: int
    codec_type: str
    codec_name: str = ""
    time_base: tuple[int, int] | None = None
    start_time_us: int | None = None
    duration_us: int | None = None
    width: int | None = None
    height: int | None = None
    r_frame_rate: float | None = None
    avg_frame_rate: float | None = None
    nb_frames: int | None = None
    sample_rate: int | None = None
    channels: int | None = None
    channel_layout: str = ""


@dataclass
class MediaProbe:
    path: str
    container: str
    duration_us: int
    video: StreamInfo | None
    audio: StreamInfo | None
    all_streams: list[StreamInfo] = field(default_factory=list)
    time_origin_us: int = 0
    video_offset_us: int = 0
    audio_offset_us: int = 0
    nominal_fps: float | None = None
    is_vfr: bool = False
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def probe_media(path: str | Path) -> MediaProbe:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {path}")

    data = ffprobe_json(["-show_format", "-show_streams", str(path)])
    fmt = data.get("format", {})
    streams: list[StreamInfo] = []
    for s in data.get("streams", []):
        info = StreamInfo(
            index=int(s.get("index", -1)),
            codec_type=s.get("codec_type", "unknown"),
            codec_name=s.get("codec_name", ""),
            time_base=_parse_time_base(s.get("time_base")),
            start_time_us=(
                seconds_to_us(_float_or_none(s.get("start_time")))
                if _float_or_none(s.get("start_time")) is not None
                else None
            ),
            duration_us=(
                seconds_to_us(_float_or_none(s.get("duration")))
                if _float_or_none(s.get("duration")) is not None
                else None
            ),
            width=s.get("width"),
            height=s.get("height"),
            r_frame_rate=_parse_rate(s.get("r_frame_rate")),
            avg_frame_rate=_parse_rate(s.get("avg_frame_rate")),
            nb_frames=int(s["nb_frames"]) if str(s.get("nb_frames", "")).isdigit() else None,
            sample_rate=int(s["sample_rate"]) if str(s.get("sample_rate", "")).isdigit() else None,
            channels=s.get("channels"),
            channel_layout=s.get("channel_layout", ""),
        )
        streams.append(info)

    video = next((s for s in streams if s.codec_type == "video"), None)
    audio = next((s for s in streams if s.codec_type == "audio"), None)

    flags: list[str] = []
    if video is None:
        flags.append("no_video_stream")
    if audio is None:
        flags.append("no_audio_stream")

    format_start_us = (
        seconds_to_us(_float_or_none(fmt.get("start_time")))
        if _float_or_none(fmt.get("start_time")) is not None
        else None
    )
    duration_us = (
        seconds_to_us(_float_or_none(fmt.get("duration")))
        if _float_or_none(fmt.get("duration")) is not None
        else None
    )
    if duration_us is None:
        candidates = [s.duration_us for s in streams if s.duration_us is not None]
        duration_us = max(candidates) if candidates else 0
        flags.append("duration_from_streams")

    # t0: 映像ストリームの開始時刻を優先。無ければコンテナ、それも無ければ 0。
    if video is not None and video.start_time_us is not None:
        t0 = video.start_time_us
        origin_source = "video_stream_start_time"
    elif format_start_us is not None:
        t0 = format_start_us
        origin_source = "format_start_time"
    else:
        t0 = 0
        origin_source = "assumed_zero"
        flags.append("time_origin_assumed_zero")

    video_offset = (video.start_time_us - t0) if (video and video.start_time_us is not None) else 0
    audio_offset = (audio.start_time_us - t0) if (audio and audio.start_time_us is not None) else 0
    if audio is not None and audio.start_time_us is None:
        flags.append("audio_start_time_missing")
    if abs(audio_offset) > 50_000:
        flags.append("audio_start_offset_nonzero")

    nominal_fps = None
    is_vfr = False
    if video is not None:
        nominal_fps = video.avg_frame_rate or video.r_frame_rate
        if video.r_frame_rate and video.avg_frame_rate and video.avg_frame_rate > 0:
            ratio = abs(video.r_frame_rate - video.avg_frame_rate) / video.avg_frame_rate
            if ratio > _VFR_RATIO_TOLERANCE:
                is_vfr = True
                flags.append("vfr_suspected")

    probe = MediaProbe(
        path=str(path),
        container=fmt.get("format_name", ""),
        duration_us=int(duration_us),
        video=video,
        audio=audio,
        all_streams=streams,
        time_origin_us=int(t0),
        video_offset_us=int(video_offset),
        audio_offset_us=int(audio_offset),
        nominal_fps=nominal_fps,
        is_vfr=is_vfr,
        flags=flags,
    )
    log.info(
        "probe: duration=%.3fs t0=%dus video_offset=%dus audio_offset=%dus vfr=%s flags=%s",
        probe.duration_us / 1e6,
        probe.time_origin_us,
        probe.video_offset_us,
        probe.audio_offset_us,
        probe.is_vfr,
        ",".join(probe.flags) or "-",
    )
    probe.flags.append(f"time_origin_source={origin_source}")
    return probe


def check_frame_timestamps(path: str | Path, *, max_frames: int = 3000) -> dict[str, Any]:
    """フレームのタイムスタンプの欠落・飛び・連続性を検査する (設計 4.1)。

    先頭 max_frames 分だけを検査する軽量チェック。全域の検査は frame_scan が担う。
    """
    data = ffprobe_json(
        [
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=pts,pts_time,best_effort_timestamp_time",
            "-read_intervals",
            f"%+#{max_frames}",
            str(path),
        ]
    )
    times: list[float] = []
    missing = 0
    for f in data.get("frames", []):
        t = _float_or_none(f.get("best_effort_timestamp_time")) or _float_or_none(f.get("pts_time"))
        if t is None:
            missing += 1
            continue
        times.append(t)
    result: dict[str, Any] = {
        "checked_frames": len(data.get("frames", [])),
        "missing_timestamps": missing,
        "non_monotonic": 0,
        "large_gaps": 0,
        "median_delta_us": None,
    }
    if len(times) < 2:
        return result
    deltas = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    result["non_monotonic"] = sum(1 for d in deltas if d <= 0)
    positive = sorted(d for d in deltas if d > 0)
    if positive:
        median = positive[len(positive) // 2]
        result["median_delta_us"] = seconds_to_us(median)
        result["large_gaps"] = sum(1 for d in deltas if d > median * 3)
    return result
