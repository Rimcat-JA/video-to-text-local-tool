"""データモデル (設計 9.1)。

SQLite を正本とし、ここではその行に対応する Python 側の型を定義する。
原文 (text) と比較用正規化 (norm_text) は必ず別フィールドに持つ。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# ---- 品質フラグ (推測で埋めないための明示、設計 1.1 / 9.2) ----
FLAG_UNREADABLE = "unreadable"  # 判読不能な文字がある
FLAG_TRUNCATED = "truncated"  # モデル出力が長さ制限で切れた
FLAG_PARTIAL = "partial"  # 画面外にはみ出す等で部分表示
FLAG_WRAP_AMBIGUOUS = "wrap_ambiguous"  # 折り返しと本来の改行を区別できない
FLAG_ORDER_GUESSED = "order_guessed"  # 読み順が推測
FLAG_NOT_EXTRACTED = "not_extracted"  # 抽出を実行していない / 失敗した
FLAG_SHORT_STATE = "short_state"  # 0.5 秒未満の短時間表示
FLAG_SCHEMA_INVALID = "schema_invalid"
FLAG_MERGED_TILES = "merged_tiles"
FLAG_LOW_RESOLUTION = "low_resolution"

# ---- 領域の役割 (設計 5.1) ----
ROLE_MATERIAL_BODY = "material_body"  # 教材本文
ROLE_MATERIAL_CONTEXT = "material_context"  # ファイル名・タブ名・行番号など
ROLE_OTHER_SCREEN_TEXT = "other_screen_text"  # ツールバー・通知・字幕など
ROLE_BURNED_CAPTION = "burned_caption"  # 焼き込み字幕 (音声文字起こしと出典を分ける)

REGION_KINDS = [
    "heading",
    "paragraph",
    "bullet",
    "table",
    "code",
    "formula",
    "terminal_output",
    "caption",
    "label",
    "ui",
    "unknown",
]

SCREEN_KINDS = ["slide", "code_editor", "terminal", "browser", "mixed", "blank", "other"]

# ---- 表示状態の種類 (設計 5.4) ----
STATE_STABLE = "stable"  # 安定した本文表示
STATE_SHORT = "short"  # 短時間表示
STATE_TRANSITION = "transition"  # 切り替え途中
STATE_UNEXTRACTED = "unextracted"  # 未抽出
STATE_BLANK = "blank"  # 黒画面・空画面


@dataclass
class Region:
    """画面上の 1 領域。座標は元画像 (original) のピクセル系。"""

    region_id: str
    kind: str
    role: str
    text: str  # 原文。要約・翻訳・修正をしない
    bbox: list[int]  # [x0, y0, x1, y1] 元画像ピクセル
    coord_space: str = "original_px"
    reading_order: int = 0
    line_numbers: list[str] = field(default_factory=list)  # コード本文と混ぜない (設計 6.3)
    language_hint: str = ""
    flags: list[str] = field(default_factory=list)
    unreadable: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)  # 推測候補は原文と別欄
    evidence: dict[str, Any] = field(default_factory=dict)  # frame_id / crop / attempt_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Region":
        known = {f: d.get(f) for f in cls.__dataclass_fields__ if f in d}
        return cls(**known)  # type: ignore[arg-type]


@dataclass
class ScreenContent:
    """再利用可能な画面本文 (同一本文は 1 レコード、出現は別)。"""

    id: str
    text_hash: str
    body_text: str  # 教材本文だけを読み順に連結した原文
    regions: list[Region] = field(default_factory=list)
    reading_order: list[str] = field(default_factory=list)
    screen_kind: str = "other"
    context: dict[str, Any] = field(default_factory=dict)
    structure_notes: list[dict[str, Any]] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)
    source_attempt_ids: list[str] = field(default_factory=list)


@dataclass
class ScreenOccurrence:
    """その本文が表示された期間。半開区間 [start_us, end_us)。"""

    id: str
    media_id: str
    content_id: str | None
    start_us: int
    end_us: int
    boundary_start_lo_us: int
    boundary_start_hi_us: int
    boundary_end_lo_us: int
    boundary_end_hi_us: int
    state_kind: str = STATE_STABLE
    evidence_refs: list[dict[str, Any]] = field(default_factory=list)
    parent_block_id: str | None = None
    change_summary: str = ""
    quality_flags: list[str] = field(default_factory=list)


@dataclass
class Utterance:
    """発話の原文。原文と原時刻は一度だけ正本として保存する (設計 8.2)。"""

    id: str
    media_id: str
    start_us: int
    end_us: int
    text_raw: str
    language: str = ""
    tokens: list[dict[str, Any]] = field(default_factory=list)  # 単語時刻 (任意)
    quality_flags: list[str] = field(default_factory=list)
    source: str = "asr"
    chunk_id: str = ""


@dataclass
class Alignment:
    occurrence_id: str
    utterance_id: str
    overlap_us: int
    is_primary: bool = False


@dataclass
class ReadingBlock:
    id: str
    media_id: str
    index: int
    start_us: int
    end_us: int
    occurrence_ids: list[str] = field(default_factory=list)
    kind: str = "single"  # "single" | "editing" | "scroll"
    title_hint: str = ""


@dataclass
class VisualEvent:
    id: str
    media_id: str
    start_us: int
    end_us: int
    region_ref: str
    event_kind: str  # "cursor" | "highlight" | "selection" | "pointer"
    annotation: str = ""


@dataclass
class ReviewItem:
    id: str
    media_id: str
    target_kind: str
    target_ref: str
    reason: str
    detail: str = ""
    resolution: str = ""
    review_status: str = "open"  # "open" | "resolved" | "wontfix"


@dataclass
class CoverageSpan:
    """時間軸の網羅 (設計 9.2 / 14.2)。処理状態が不明な区間を 0 にする。"""

    id: str
    media_id: str
    track: str  # "screen" | "audio"
    start_us: int
    end_us: int
    state: str  # "extracted" | "unextracted" | "failed" | "blank" | "silence" | "out_of_range"
    detail: str = ""
