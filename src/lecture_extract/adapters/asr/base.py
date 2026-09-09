"""音声認識アダプターの共通インターフェース (設計 11)。

モデルを差し替えても Utterance の形式が変わらないようにする。
時刻はチャンク内の相対時刻 (マイクロ秒) で返し、元動画の時刻への変換は呼び出し側が行う。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class AsrSegment:
    start_us: int  # チャンク先頭からの相対時刻
    end_us: int
    text: str
    language: str = ""
    tokens: list[dict[str, Any]] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


@dataclass
class AsrResult:
    status: str  # 'ok' | 'error'
    segments: list[AsrSegment] = field(default_factory=list)
    raw_ref: str = ""
    error: str = ""
    latency_ms: int = 0
    language: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class AsrAdapter(Protocol):
    def describe(self) -> dict[str, Any]:
        ...

    def transcribe(self, wav_path: str, *, out_prefix: str, chunk_start_us: int = 0) -> AsrResult:
        ...

    def close(self) -> None:
        ...
