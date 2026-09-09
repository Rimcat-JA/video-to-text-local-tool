"""検証用の画像読み取りアダプター。

モデルなしでパイプライン全体 (frame_scan → vision → align → export) を通すために使う。
実運用の抽出には使わない。manifest には adapter=stub と記録され、
品質報告でも「モデル未使用」であることが分かるようにする。

MarkerStubVision は、画像左上に描かれた白黒のマーカー列から状態 ID を復号し、
対応する台本 (JSON) の抽出結果を返す。画像を実際に読むため、代表フレーム選択や
クロップ座標の受け渡しまで含めて検証できる。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .base import STATUS_ERROR, STATUS_OK, VisionResult

MARKER_CELL = 24
MARKER_BITS = 12


def encode_marker(image: np.ndarray, value: int) -> np.ndarray:
    """左上に value を白黒セル列として描く (検証用画像の生成側で使う)。"""
    out = image.copy()
    for i in range(MARKER_BITS):
        bit = (value >> (MARKER_BITS - 1 - i)) & 1
        color = (255, 255, 255) if bit else (0, 0, 0)
        x0 = i * MARKER_CELL
        out[0:MARKER_CELL, x0 : x0 + MARKER_CELL] = color
    return out


def decode_marker(image_png: bytes) -> int | None:
    arr = cv2.imdecode(np.frombuffer(image_png, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if arr is None or arr.shape[0] < MARKER_CELL or arr.shape[1] < MARKER_BITS * MARKER_CELL:
        return None
    value = 0
    for i in range(MARKER_BITS):
        cx = i * MARKER_CELL + MARKER_CELL // 2
        cy = MARKER_CELL // 2
        patch = arr[max(0, cy - 4) : cy + 4, max(0, cx - 4) : cx + 4]
        bit = 1 if float(patch.mean()) > 127 else 0
        value = (value << 1) | bit
    return value


class ScriptedStubVision:
    """どの画像に対しても同じ結果を返す。下流の単体試験用。"""

    def __init__(self, payload: dict[str, Any] | None = None):
        self.payload = payload or {
            "screen_kind": "slide",
            "context": {},
            "regions": [
                {
                    "kind": "paragraph",
                    "role": "material_body",
                    "bbox": [0.1, 0.1, 0.9, 0.5],
                    "reading_order": 0,
                    "text": "stub",
                }
            ],
            "structure_notes": [],
            "output_complete": True,
        }
        self.calls = 0

    def describe(self) -> dict[str, Any]:
        return {
            "adapter": "stub",
            "model_revision": "scripted-stub",
            "runtime_version": "stub",
            "prompt_version": "stub",
            "params_hash": "stub",
            "model_sha256": "",
            "mmproj_sha256": "",
        }

    def health(self) -> bool:
        return True

    def ensure_ready(self) -> None:
        return None

    def extract(self, image_png: bytes, kind: str, *, extra_instruction: str = "") -> VisionResult:
        self.calls += 1
        payload = self.payload if kind == "full" else _to_crop_payload(self.payload)
        return VisionResult(
            status=STATUS_OK,
            payload=json.loads(json.dumps(payload)),
            raw_text=json.dumps(payload, ensure_ascii=False),
            finish_reason="stop",
            latency_ms=1,
        )

    def close(self) -> None:
        return None


class MarkerStubVision(ScriptedStubVision):
    """画像に埋め込まれたマーカーから状態 ID を読み、台本の結果を返す。"""

    def __init__(self, script_path: str | Path):
        super().__init__()
        self.script_path = Path(script_path)
        data = json.loads(self.script_path.read_text(encoding="utf-8"))
        self.script: dict[int, dict[str, Any]] = {int(k): v for k, v in data.items()}
        # クロップやタイルにはマーカーが写らない。直前に全画面で復号した状態を引き継ぐ。
        self._last_value: int | None = None

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info["model_revision"] = f"marker-stub@{self.script_path.name}"
        return info

    def extract(self, image_png: bytes, kind: str, *, extra_instruction: str = "") -> VisionResult:
        self.calls += 1
        started = time.time()
        value = decode_marker(image_png)
        if kind == "full":
            if value in self.script:
                self._last_value = value
        elif value not in self.script:
            value = self._last_value
        if value is None or value not in self.script:
            return VisionResult(
                status=STATUS_ERROR,
                payload=None,
                raw_text="",
                error=f"マーカーを復号できません (value={value})",
                latency_ms=int((time.time() - started) * 1000),
            )
        payload = json.loads(json.dumps(self.script[value]))
        if kind != "full":
            payload = _to_crop_payload(payload, extra_instruction)
        return VisionResult(
            status=STATUS_OK,
            payload=payload,
            raw_text=json.dumps(payload, ensure_ascii=False),
            finish_reason="stop",
            latency_ms=int((time.time() - started) * 1000),
        )


def _to_crop_payload(full_payload: dict[str, Any], hint: str = "") -> dict[str, Any]:
    """全画面用の台本からクロップ用の応答を作る。

    クロップ指定に領域 ID が含まれていればその領域だけを返す。
    """
    regions = full_payload.get("regions") or []
    chosen = None
    if hint:
        for region in regions:
            marker = region.get("stub_region_id")
            if marker and marker in hint:
                chosen = region
                break
        if chosen is None:
            for region in regions:
                if f"region_kind={region.get('kind')}" in hint:
                    chosen = region
                    break
    if chosen is None:
        chosen = regions[0] if regions else {"text": ""}
    text = chosen.get("text", "")
    return {
        "text": text,
        "lines": text.split("\n"),
        "line_numbers": chosen.get("line_numbers", []),
        "flags": chosen.get("flags", []),
        "unreadable": chosen.get("unreadable", []),
        "candidates": chosen.get("candidates", []),
        "output_complete": True,
    }
