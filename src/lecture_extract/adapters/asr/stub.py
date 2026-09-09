"""検証用の音声認識アダプター。

台本 (元動画の時間軸で書かれた発話一覧) から、チャンク相対時刻の結果を返す。
チャンク分割・重複整理・時刻の戻し変換を、モデルなしで検証するために使う。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import AsrResult, AsrSegment


class ScriptedStubAsr:
    def __init__(self, script_path: str | Path | None = None, segments: list[dict[str, Any]] | None = None):
        if segments is not None:
            self.segments = segments
            self.script_path = None
        else:
            self.script_path = Path(script_path) if script_path else None
            self.segments = (
                json.loads(self.script_path.read_text(encoding="utf-8")) if self.script_path else []
            )
        self.calls = 0

    def describe(self) -> dict[str, Any]:
        return {
            "adapter": "stub",
            "model_revision": "scripted-stub",
            "runtime_version": "stub",
            "model_sha256": "",
            "params_hash": "stub",
            "params": {},
        }

    def ensure_ready(self) -> None:
        return None

    def transcribe(self, wav_path: str, *, out_prefix: str, chunk_start_us: int = 0) -> AsrResult:
        self.calls += 1
        import wave

        with wave.open(str(wav_path), "rb") as wf:
            duration_us = int(round(wf.getnframes() / wf.getframerate() * 1_000_000))
        chunk_end_us = chunk_start_us + duration_us
        out: list[AsrSegment] = []
        for seg in self.segments:
            start = int(seg["start_us"])
            end = int(seg["end_us"])
            if end <= chunk_start_us or start >= chunk_end_us:
                continue
            out.append(
                AsrSegment(
                    start_us=start - chunk_start_us,
                    end_us=end - chunk_start_us,
                    text=seg["text"],
                    language=seg.get("language", "ja"),
                    tokens=[],
                )
            )
        Path(f"{out_prefix}.json").write_text(
            json.dumps({"stub": True, "segments": len(out)}, ensure_ascii=False), encoding="utf-8"
        )
        return AsrResult(
            status="ok", segments=out, raw_ref=f"{out_prefix}.json", latency_ms=1, language="ja"
        )

    def close(self) -> None:
        return None
