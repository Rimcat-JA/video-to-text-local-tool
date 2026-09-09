"""解析設定 (設計 3.3 / 5 / 6.2 / 7.1 / 12)。

初期値は設計案の「初期候補」であり、実測で見直す前提の値。
設定はステージ単位でハッシュ化し、再実行時の再解析範囲判定に使う (設計 12.2)。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from .util.hashing import config_hash


@dataclass
class ScanConfig:
    """画像の変化候補検出 (設計 5.2 / 5.3 / 5.5)。"""

    mode: str = "preserve"  # "preserve" | "fast"
    fast_fps: float = 2.0  # 高速モードの初期抽出レート
    diff_width: int = 960  # 変化検出用の縮小幅 (認識には元解像度を使う)
    tile_cols: int = 12
    tile_rows: int = 8
    pixel_delta: int = 18  # この差以上を「変化画素」とみなす (0-255)
    global_change_ratio: float = 0.004  # 全画面の変化画素率の閾値
    tile_change_ratio: float = 0.05  # 1タイル内の変化画素率の閾値
    # 比率だけで判定すると、1080p のコード 1 文字の変更が閾値に届かない。
    # 変化画素の実数でも判定し、小さな変更を落とさない (設計 5.2 / 14.3)。
    min_changed_pixels: int = 10  # 全画面で、この画素数以上の変化を候補にする
    min_changed_pixels_tile: int = 8  # 1 タイルで、この画素数以上の変化を候補にする
    min_changed_tiles: int = 1
    cursor_max_pixels: int = 900  # これ以下の変化画素数は「小さな変化」として保留判定へ回す
    stable_us: int = 500_000  # 本文表示候補とみなす安定時間 (0.5 秒)
    min_state_us: int = 0  # 0 未満の状態も破棄しない。短時間表示も保存する
    cursor_max_area_ratio: float = 0.0015  # カーソル相当とみなす変化面積の上限
    cursor_revert_us: int = 1_500_000  # この時間内に元へ戻ればカーソル点滅候補
    periodic_full_recheck_us: int = 120_000_000  # 定期的な全文再認識の間隔 (設計 5.3)
    representative_samples: int = 5  # 代表フレーム選定のためのサンプル数
    ignore_regions: list[list[float]] = field(default_factory=list)  # 正規化 xyxy の除外範囲
    max_cache_bytes: int = 20 * 1024**3  # 画像キャッシュ上限 (設計 3.3)


@dataclass
class VisionConfig:
    """VLM による画面全文抽出 (設計 6)。"""

    adapter: str = "llama_server"  # "llama_server" | "stub"
    server_url: str = "http://127.0.0.1:8080"
    model_path: str = ""
    mmproj_path: str = ""
    model_alias: str = "qwen3-vl-8b-instruct-q4_k_m"
    manage_server: bool = False  # True なら llama-server を起動・停止まで面倒を見る
    server_binary: str = "llama-server"
    n_ctx: int = 8192
    n_gpu_layers: int = 99
    temperature: float = 0.0
    seed: int = 42
    max_tokens: int = 4096
    request_timeout_s: int = 600
    concurrency: int = 1  # 設計 3.3: GPU 1 枚につき重い推論は 1 件
    crop_reread_kinds: list[str] = field(
        default_factory=lambda: ["code", "terminal_output", "table", "formula", "caption"]
    )
    crop_reread_all_body_when_downscaled: bool = True
    crop_padding_ratio: float = 0.02
    crop_min_upscale: float = 1.0
    crop_max_pixels: int = 1_600 * 1_600
    tile_overlap_ratio: float = 0.18  # 設計 6.2: 15-20% を初期候補
    tile_max_height_px: int = 1_100
    max_image_long_side: int = 1_600  # 全体パス用。領域別読み取りは原寸クロップを使う
    retry_limit: int = 2  # 設計 12.2: 実行時エラーの初期再試行上限
    prompt_version: str = "v1"


@dataclass
class AsrConfig:
    """発話抽出 (設計 7)。"""

    adapter: str = "whisper_cpp"  # "whisper_cpp" | "faster_whisper" | "stub"
    binary: str = "whisper-cli"
    model_path: str = ""
    language: str = "auto"
    threads: int = 0  # 0 なら実行時に決める
    chunk_us: int = 600_000_000  # 10 分
    chunk_overlap_us: int = 2_000_000  # 前後 2 秒
    channel: str = "mix"  # "mix" | "left" | "right" | "0".."n"
    word_timestamps: bool = False  # whisper.cpp では experimental (設計 7.3)
    extra_args: list[str] = field(default_factory=list)
    retry_limit: int = 2
    initial_prompt: str = ""


@dataclass
class BlockConfig:
    """読書ブロック構築 (設計 5.4 / 8.2)。"""

    group_editing: bool = True
    editing_similarity: float = 0.72  # この類似度以上で連続入力とみなす
    editing_max_gap_us: int = 8_000_000
    editing_min_states: int = 3  # これ未満なら親ブロック化しない
    editing_state_max_us: int = 12_000_000


@dataclass
class ExportConfig:
    """出力 (設計 10)。"""

    include_ui_appendix: bool = True
    include_structure_notes: bool = True
    srt_max_chars: int = 0  # 0 なら発話原文を分割しない
    video_link: bool = True


@dataclass
class RunConfig:
    """1 回の解析全体の設定。"""

    input_path: str = ""
    work_dir: str = "work"
    out_dir: str = "out"
    range_start_us: int | None = None
    range_end_us: int | None = None
    scan: ScanConfig = field(default_factory=ScanConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    block: BlockConfig = field(default_factory=BlockConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    # ---- 直列化 ----
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        return _from_dict(cls, data)

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2, sort_keys=True)

    # ---- ハッシュ ----
    def stage_config_hash(self, stage: str) -> str:
        """ステージごとの設定ハッシュ。無関係な設定変更で再解析させない。"""
        common = {
            "range_start_us": self.range_start_us,
            "range_end_us": self.range_end_us,
        }
        table = {
            "ingest": {},
            "timeline": {},
            "frame_scan": asdict(self.scan),
            "audio_prepare": {k: v for k, v in asdict(self.asr).items() if k in ("channel", "chunk_us", "chunk_overlap_us")},
            "asr": asdict(self.asr),
            "vision": asdict(self.vision),
            "align": {},
            "blocks": asdict(self.block),
            "export": asdict(self.export),
        }
        if stage not in table:
            raise KeyError(f"unknown stage: {stage}")
        return config_hash({"stage": stage, "common": common, "config": table[stage]})


def _from_dict(cls, data: Any):
    if not is_dataclass(cls):
        return data
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        if is_dataclass(f.type) if isinstance(f.type, type) else False:
            kwargs[f.name] = _from_dict(f.type, value)
        else:
            kwargs[f.name] = value
    obj = cls(**{k: v for k, v in kwargs.items() if not is_dataclass(v)})
    # ネストしたデータクラスを型注釈から解決する。
    for f in fields(cls):
        if f.name not in data:
            continue
        nested_cls = _NESTED.get((cls.__name__, f.name))
        if nested_cls is not None:
            setattr(obj, f.name, _from_dict(nested_cls, data[f.name]))
        elif f.name in kwargs:
            setattr(obj, f.name, kwargs[f.name])
    return obj


_NESTED = {
    ("RunConfig", "scan"): ScanConfig,
    ("RunConfig", "vision"): VisionConfig,
    ("RunConfig", "asr"): AsrConfig,
    ("RunConfig", "block"): BlockConfig,
    ("RunConfig", "export"): ExportConfig,
}

STAGES = ["ingest", "timeline", "frame_scan", "audio_prepare", "asr", "vision", "align", "blocks", "export"]
