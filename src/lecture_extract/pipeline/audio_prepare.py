"""音声の準備 (設計 7.1)。

- 動画から 16kHz・モノラル・16bit PCM を作成し、元の時間軸との対応を保存する。
- 長時間動画は連続した音声区間に分けて処理する。外側チャンク 10 分、重複 2 秒が初期候補。
- 無音を切り詰めて映像との対応を壊さない。無音は「検出して記録する」だけにする。
"""

from __future__ import annotations

import logging
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import RunConfig
from ..util.timeutil import US
from .ffmpeg_tools import ffmpeg_path, run
from .probe import MediaProbe

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
SILENCE_DBFS = -45.0
MIN_SILENCE_US = 1_000_000


@dataclass
class AudioChunk:
    index: int
    chunk_id: str
    path: str
    start_us: int  # 元動画の時間軸における開始時刻
    end_us: int
    core_start_us: int  # 重複部分を除いた「この チャンクが正本になる」範囲
    core_end_us: int


def _channel_filter(channel: str) -> list[str]:
    """設計 7.1: 単純平均で音声が消える場合に備え、チャンネルを選べるようにする。"""
    if channel == "mix" or channel == "":
        return ["-ac", "1"]
    if channel == "left":
        return ["-af", "pan=mono|c0=c0"]
    if channel == "right":
        return ["-af", "pan=mono|c0=c1"]
    if channel.isdigit():
        return ["-af", f"pan=mono|c0=c{int(channel)}"]
    raise ValueError(f"未知のチャンネル指定です: {channel}")


def extract_master_wav(cfg: RunConfig, probe: MediaProbe, media_id: str) -> Path:
    out_dir = Path(cfg.work_dir) / "audio" / media_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"master_{cfg.asr.channel}.wav"
    if out.exists() and out.stat().st_size > 44:
        log.info("既存の音声中間ファイルを使います: %s", out.name)
        return out
    tmp = out.with_suffix(".wav.partial")
    cmd = [
        ffmpeg_path(),
        "-v",
        "error",
        "-y",
        "-i",
        probe.path,
        "-vn",
        "-map",
        "0:a:0",
        *_channel_filter(cfg.asr.channel),
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(tmp),
    ]
    log.info("音声を書き出しています (16kHz mono s16, channel=%s)", cfg.asr.channel)
    run(cmd, timeout=None)
    tmp.replace(out)
    return out


def wav_duration_us(path: Path) -> int:
    with wave.open(str(path), "rb") as wf:
        return int(round(wf.getnframes() / wf.getframerate() * US))


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2 or wf.getnchannels() != 1:
            raise ValueError("16bit モノラルの WAV を想定しています")
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16)


def write_wav(path: Path, samples: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    with wave.open(str(tmp), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.astype(np.int16).tobytes())
    tmp.replace(path)


def make_chunks(
    cfg: RunConfig,
    probe: MediaProbe,
    media_id: str,
    master: Path,
    start_us: int,
    end_us: int,
) -> list[AudioChunk]:
    """解析範囲を、重複を持つチャンクへ切り分ける。

    切り出し対応表 (元動画の時間軸との対応) をチャンク自身に持たせる。
    """
    audio_offset = probe.audio_offset_us  # 元動画の t 軸における音声の開始位置
    samples = read_wav(master)
    total_us = int(round(len(samples) / SAMPLE_RATE * US))
    out_dir = master.parent / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_us = max(10 * US, cfg.asr.chunk_us)
    overlap_us = max(0, cfg.asr.chunk_overlap_us)
    chunks: list[AudioChunk] = []
    core_start = max(start_us, audio_offset)
    index = 0
    while core_start < end_us:
        core_end = min(end_us, core_start + chunk_us)
        chunk_start = max(audio_offset, core_start - overlap_us)
        chunk_end = min(audio_offset + total_us, core_end + overlap_us)
        if chunk_end <= chunk_start:
            break
        s0 = int(round((chunk_start - audio_offset) / US * SAMPLE_RATE))
        s1 = int(round((chunk_end - audio_offset) / US * SAMPLE_RATE))
        s0 = max(0, min(len(samples), s0))
        s1 = max(s0, min(len(samples), s1))
        chunk_id = f"chunk_{index:05d}"
        path = out_dir / f"{chunk_id}.wav"
        if not path.exists():
            write_wav(path, samples[s0:s1])
        chunks.append(
            AudioChunk(
                index=index,
                chunk_id=chunk_id,
                path=str(path),
                start_us=chunk_start,
                end_us=chunk_end,
                core_start_us=core_start,
                core_end_us=core_end,
            )
        )
        index += 1
        core_start = core_end
    log.info("音声チャンクを %d 個作成しました (chunk=%.0fs overlap=%.1fs)", len(chunks), chunk_us / US, overlap_us / US)
    return chunks


def detect_silence(
    master: Path, audio_offset_us: int, *, threshold_dbfs: float = SILENCE_DBFS, min_us: int = MIN_SILENCE_US
) -> list[tuple[int, int]]:
    """無音区間を検出する。除去はしない (設計 7.1)。"""
    samples = read_wav(master).astype(np.float32) / 32768.0
    win = SAMPLE_RATE // 50  # 20ms
    if win <= 0 or samples.size < win:
        return []
    n = samples.size // win
    frames = samples[: n * win].reshape(n, win)
    rms = np.sqrt(np.maximum(1e-12, (frames**2).mean(axis=1)))
    dbfs = 20 * np.log10(rms)
    quiet = dbfs < threshold_dbfs

    spans: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < n and quiet[j]:
            j += 1
        start_us = audio_offset_us + int(i * win / SAMPLE_RATE * US)
        end_us = audio_offset_us + int(j * win / SAMPLE_RATE * US)
        if end_us - start_us >= min_us:
            spans.append((start_us, end_us))
        i = j
    return spans


def verify_audio_alignment(probe: MediaProbe, master: Path) -> dict[str, Any]:
    """書き出した音声と、コンテナ側の音声長が食い違っていないか確認する。"""
    produced = wav_duration_us(master)
    expected = None
    if probe.audio is not None and probe.audio.duration_us:
        expected = probe.audio.duration_us
    elif probe.duration_us:
        expected = probe.duration_us - probe.audio_offset_us
    diff = None if expected is None else produced - expected
    result = {
        "produced_us": produced,
        "expected_us": expected,
        "diff_us": diff,
        "audio_offset_us": probe.audio_offset_us,
        "mismatch": bool(diff is not None and abs(diff) > 200_000),
    }
    if result["mismatch"]:
        log.warning("音声長が想定と %dms ずれています。時刻対応を確認してください。", (diff or 0) // 1000)
    return result
