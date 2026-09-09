"""比較用の正規化 (設計 5.2)。

比較用の正規化結果と、出力する原文は必ず別に持つ。
コードではインデント・空白・記号・大小文字を勝手に同一視しない。
"""

from __future__ import annotations

import re
import unicodedata

_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)
_MULTI_SPACE = re.compile(r"[ \t　]+")

# 文字化けしやすい OCR の揺れのみを対象にする最小限の対応表。
# 見た目が同じで意味が違う文字 (全角/半角の記号) は勝手に統合しない。
_SOFT_CONFUSABLES = {
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "―": "-",
    "−": "-",
}


def normalize_for_compare(text: str, *, is_code: bool = False) -> str:
    """表示状態が「同じ内容か」を比較するための正規化。

    is_code=True では字下げと内部空白を保持する。行末空白と改行コードだけを揃える。
    """
    if text is None:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if is_code:
        # コードは字下げ・記号を保持。行末空白と末尾の空行のみ揃える。
        text = _TRAILING_WS.sub("", text)
        return text.strip("\n")
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _SOFT_CONFUSABLES.items():
        text = text.replace(src, dst)
    text = _TRAILING_WS.sub("", text)
    text = _MULTI_SPACE.sub(" ", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines).strip("\n")


def normalize_for_dedup_utterance(text: str) -> str:
    """チャンク重複範囲で同一発話を判定するための正規化 (設計 7.1)。"""
    text = unicodedata.normalize("NFKC", (text or "").strip())
    text = _MULTI_SPACE.sub(" ", text)
    return re.sub(r"[、。,\.!?！？…]+", "", text).lower()


def line_similarity(a: str, b: str) -> float:
    """行集合ベースの粗い類似度。0.0-1.0。連続入力の判定にだけ使う。"""
    a_lines = [ln for ln in (a or "").split("\n")]
    b_lines = [ln for ln in (b or "").split("\n")]
    if not a_lines and not b_lines:
        return 1.0
    if not a_lines or not b_lines:
        return 0.0
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a_lines, b_lines).ratio()
