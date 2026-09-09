"""画像読み取りアダプターの共通インターフェース (設計 11)。

モデルを差し替えても ScreenContent の形式が変わらないようにする。
ネットワーク通信は同一 PC のループバック接続だけに限定する (設計 13.1)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

STATUS_OK = "ok"
STATUS_TRUNCATED = "truncated"
STATUS_SCHEMA_INVALID = "schema_invalid"
STATUS_ERROR = "error"


@dataclass
class VisionResult:
    """1 回の推論結果。原応答は必ず保持する。"""

    status: str
    payload: dict[str, Any] | None
    raw_text: str
    finish_reason: str = ""
    latency_ms: int = 0
    error: str = ""
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK


class VisionAdapter(Protocol):
    """画像 1 枚を読み取る部品。時刻の付与は呼び出し側が行う。"""

    def describe(self) -> dict[str, Any]:
        """モデル・ランタイムの同一性判定に使う情報を返す。"""

    def health(self) -> bool:
        """すぐに推論できる状態かどうか。"""

    def extract(self, image_png: bytes, kind: str, *, extra_instruction: str = "") -> VisionResult:
        """kind は 'full' | 'crop' | 'tile'。"""

    def close(self) -> None:
        ...
