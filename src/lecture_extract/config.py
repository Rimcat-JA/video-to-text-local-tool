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
    # 符号化ノイズは 1〜2 画素が全画面に散る。文字の変化は塊になる。
    # 局所密度で両者を分け、散らばった孤立画素を変化候補から除く。
    noise_density_window: int = 5  # 局所密度を測る窓 (画素)
    # 画面の最外周には符号化の端部アーティファクトが固定的に出る。差分から除く幅 (縮小画像上)。
    border_margin_px: int = 2
    min_local_density: float = 0.2  # 窓内でこの割合以上が変化していれば本物の変化とみなす
    min_changed_pixels_tile: int = 8  # 1 タイルで、この画素数以上の変化を候補にする
    min_changed_tiles: int = 1
    cursor_max_pixels: int = 900  # これ以下の変化画素数は「小さな変化」として保留判定へ回す
    # マウスポインタの移動判定。移動した物体は「消えた位置」「現れた位置」の 2 つの塊を作る。
    # 参照画像の一方の見た目が現在画像のもう一方に現れていれば移動とみなす (0 で無効)。
    pointer_move_similarity: float = 22.0  # 平均輝度差 (0-255) の上限
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
    # 空ならモデルファイル名から決める。固定文字列にすると、モデルを差し替えても
    # 抽出履歴に同じ名前が記録され、どのモデルの結果か分からなくなる (設計 13.2)。
    model_alias: str = ""
    manage_server: bool = False  # True なら llama-server を起動・停止まで面倒を見る
    server_binary: str = "llama-server"
    n_ctx: int = 16384
    n_gpu_layers: int = 99
    # llama.cpp が Qwen-VL について警告する下限。文字の読み取り精度に効く。
    # 0 なら指定しない (ランタイムの既定に任せる)。
    image_min_tokens: int = 1024
    temperature: float = 0.0
    seed: int = 42
    max_tokens: int = 4096
    # クロップは 1 領域分しか転記しないので、全画面より小さい上限で十分。
    # 上限を絞ることで、繰り返しループに入ったときの損失時間を抑える。
    crop_max_tokens: int = 1024
    # 同じ記号を延々と生成する退行を抑える。1.0 で無効。
    repeat_penalty: float = 1.05
    # 繰り返した並びを抑える DRY 抑制。allowed_length より長い繰り返しに罰則をかける。
    # 0 で無効。短い繰り返し（コードの同一行など）は許す。
    dry_multiplier: float = 0.8
    dry_base: float = 1.75
    dry_allowed_length: int = 8
    request_timeout_s: int = 600
    concurrency: int = 1  # 設計 3.3: GPU 1 枚につき重い推論は 1 件
    crop_reread_kinds: list[str] = field(
        default_factory=lambda: ["code", "terminal_output", "table", "formula", "caption"]
    )
    # 全画面パスが原寸で読めている場合、同じ領域をもう一度読んでも解像度は上がらない。
    # 原寸クロップが効くのは、細かい文字が密に並ぶ画面 (コードエディタ・端末など)。
    # 画面種別で絞り、スライドでは全画面パスの結果を採用する (設計 6.1 の手順 3)。
    crop_reread_screen_kinds: list[str] = field(
        default_factory=lambda: ["code_editor", "terminal", "browser", "mixed"]
    )
    crop_reread_all_body_when_downscaled: bool = True
    # 1 画面あたりのクロップ再認識の上限。実測では、コード画面で全領域を読み直すと
    # 1 画面 22 回に達し、その 68% が全画面パスと食い違って断片化した。
    # 判読不能の印がある領域を優先し、それ以外は読み直さない。
    crop_reread_max_per_screen: int = 3
    crop_reread_only_uncertain: bool = True
    # 設計 5.2 第 2 段階: 変化した領域だけを読み直し、文字が変わったか確かめる。
    region_check: bool = True
    region_check_max_area: float = 0.15  # 画面に対するこの割合以下の変化だけを局所判定にする
    region_check_padding: float = 0.6  # 変化範囲の周囲をどれだけ広げて読むか
    crop_padding_ratio: float = 0.02
    crop_min_upscale: float = 1.0
    crop_max_pixels: int = 1_600 * 1_600
    tile_overlap_ratio: float = 0.18  # 設計 6.2: 15-20% を初期候補
    tile_max_height_px: int = 1_100
    # 切り詰めが起きた場合の再帰分割 (設計 6.2)。切り詰めた側だけを更に分ける。
    split_max_depth: int = 3  # 最大 8 片まで細かくする
    split_min_height_px: int = 180  # これ以下は分けない
    split_join_lines: int = 12  # 結合の照合に使う重複行数
    # 全体パス用の上限。1080p を縮小しないことで、全本文領域の再認識を避ける。
    # 領域別読み取りは常に原寸クロップを使う。
    max_image_long_side: int = 1_920
    retry_limit: int = 2  # 設計 12.2: 実行時エラーの初期再試行上限
    prompt_version: str = "v1"
    # VLM 抽出の対象にする最小の表示時間。これより短い状態も期間としては必ず保存し、
    # 「短時間表示・全文未確定」として残す (設計 5.4)。
    extract_min_state_us: int = 1_000_000


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
    # GPU を使うビルドかどうか。CPU ビルドで GPU ロックを取ると、GPU を使わないのに
    # 画像認識をブロックしてしまう (設計 12.2 のロックは GPU 競合を避けるためのもの)。
    uses_gpu: bool = False


@dataclass
class BlockConfig:
    """読書ブロック（教材単位）の構築 (設計 5.4 / 8.2)。"""

    # 同じスライド・同じコード領域・同じ文脈が続く範囲をひとつの教材単位にまとめる。
    group_units: bool = True
    unit_similarity: float = 0.45  # この類似度以上なら同じ教材の変更とみなす
    unit_max_gap_us: int = 15_000_000  # これ以上間が空いたら別の教材
    unit_max_duration_us: int = 900_000_000  # ひとつの単位が長くなりすぎないようにする

    # 旧: 連続入力のまとめ（教材単位の判定に統合済み。設定は互換のため残す）
    group_editing: bool = True
    editing_similarity: float = 0.72
    editing_max_gap_us: int = 8_000_000
    editing_min_states: int = 3
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
