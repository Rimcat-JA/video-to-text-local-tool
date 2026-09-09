"""画面の変化候補検出と表示状態の切り出し (設計 5.2 / 5.3 / 5.4 / 5.5)。

第 1 段階: 画像から変化候補を拾う。全体差分と、タイルごとの局所差分を併用する。
コード 1 文字の変更は全画面に占める割合が小さいため、全体類似度だけで判定しない。

第 2 段階 (文字内容が実際に変わったかの確認) は vision / screen_tracker が担当する。
ここでは「変化候補」と「表示期間」だけを決め、境界は推定であることを保存する (設計 4.2)。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator

import av
import cv2
import numpy as np

from ..config import RunConfig, ScanConfig
from ..db.store import Store, new_id
from ..models import (
    FLAG_NOT_EXTRACTED,
    FLAG_SHORT_STATE,
    STATE_BLANK,
    STATE_SHORT,
    STATE_STABLE,
    STATE_TRANSITION,
    CoverageSpan,
    ScreenOccurrence,
    VisualEvent,
)
from ..util.hashing import sha256_bytes
from ..util.timeutil import US, pts_to_us
from .cache import check_free_space
from .probe import MediaProbe

log = logging.getLogger(__name__)

# VLM 抽出の対象にする最小の表示時間。これより短い状態も「期間」としては必ず保存する。
EXTRACT_MIN_STATE_US = 150_000
# 連続する短い状態がこの長さ以下で BURST_MIN_LEN 個以上続いたら遷移とみなす。
BURST_STATE_US = 400_000
BURST_MIN_LEN = 4
# 代表フレーム候補を取る最小間隔。
SAMPLE_MIN_GAP_US = 200_000


@dataclass
class FrameSample:
    t_us: int
    pts: int | None
    sharpness: float
    png: bytes
    width: int
    height: int


@dataclass
class ScreenState:
    """観測された表示状態。文字内容の同一性はまだ確認していない。"""

    index: int
    start_us: int
    end_us: int
    boundary_start_lo_us: int  # 直前の旧状態を最後に観測した時刻
    boundary_start_hi_us: int  # 新状態を最初に観測した時刻
    boundary_end_lo_us: int
    boundary_end_hi_us: int
    state_kind: str = STATE_STABLE
    change_ratio: float = 0.0
    changed_tiles: int = 0
    is_blank: bool = False
    # 変化した範囲 (縮小画像に対する正規化 xyxy)。第 2 段階の局所再認識に使う。
    change_bbox_norm: list[float] | None = None
    trigger: str = "diff"  # 'diff' | 'small_persistent' | 'periodic' | 'start' | 'resume'
    samples: list[FrameSample] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def duration_us(self) -> int:
        return max(0, self.end_us - self.start_us)


def _target_small_size(width: int, height: int, diff_width: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        return diff_width, max(1, diff_width * 9 // 16)
    dw = min(diff_width, width)
    dh = max(1, int(round(height * dw / width)))
    # swscale は偶数サイズを好むため揃える。
    return dw - (dw % 2), dh - (dh % 2)


def _tile_counts(mask: np.ndarray, cols: int, rows: int) -> tuple[np.ndarray, np.ndarray]:
    """タイルごとの (変化画素数, 面積) を返す。

    比率だけで判定すると、高解像度の画面ではコード 1 文字の変更が閾値に届かない。
    実数と比率の両方で判定できるように、生の画素数を返す。
    """
    h, w = mask.shape
    ys = np.linspace(0, h, rows + 1).astype(int)
    xs = np.linspace(0, w, cols + 1).astype(int)
    integral = cv2.integral(mask.astype(np.uint8))
    counts = np.zeros((rows, cols), dtype=np.int64)
    areas = np.zeros((rows, cols), dtype=np.int64)
    for r in range(rows):
        for c in range(cols):
            y0, y1, x0, x1 = ys[r], ys[r + 1], xs[c], xs[c + 1]
            areas[r, c] = max(1, (y1 - y0) * (x1 - x0))
            counts[r, c] = (
                integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
            )
    return counts, areas


def _drop_border(mask: np.ndarray, margin: int) -> np.ndarray:
    """最外周を差分から除く。

    符号化の端部アーティファクトが最終行・最終列に固定的に現れ、
    毎回 1 画素幅の塊として検出されてしまうため。
    """
    if margin <= 0:
        return mask
    out = mask.copy()
    out[:margin, :] = False
    out[-margin:, :] = False
    out[:, :margin] = False
    out[:, -margin:] = False
    return out


def _significant_mask(mask: np.ndarray, window: int, min_density: float) -> np.ndarray:
    """符号化ノイズを落とし、塊になっている変化だけを残す。

    実写・画面録画の H.264 では、静止した画面でも量子化ノイズが 1〜2 画素単位で
    全画面に散る。文字やコードの変化は必ず連続した塊になるため、局所密度で分ける。
    孤立画素を落とすだけで、1 文字の変更は残る (グリフ内の局所密度は十分高い)。
    """
    if window <= 1 or min_density <= 0.0:
        return mask
    density = cv2.blur(mask.astype(np.float32), (window, window))
    return mask & (density >= min_density)


def _blobs(mask: np.ndarray, max_blobs: int = 6) -> list[tuple[int, int, int, int, int]]:
    """変化画素の塊を (x0, y0, x1, y1, 画素数) で返す。大きい順。"""
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        out.append((int(x), int(y), int(x + w), int(y + h), int(area)))
    out.sort(key=lambda b: -b[4])
    return out[:max_blobs]


def _patch_similarity(a: np.ndarray, b: np.ndarray, size: int = 24) -> float:
    """2 つの小片の見た目の近さ。0 に近いほど似ている (0-255)。"""
    if a.size == 0 or b.size == 0:
        return 255.0
    pa = cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    pb = cv2.resize(b, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    return float(np.abs(pa - pb).mean())


def _looks_like_pointer_move(
    current: np.ndarray,
    reference: np.ndarray,
    blobs: list[tuple[int, int, int, int, int]],
    *,
    max_similarity: float,
) -> bool:
    """マウスポインタなどの移動か (設計 5.5)。

    移動した物体は「元の位置から消える」「新しい位置に現れる」の 2 つの塊を作り、
    大きさがほぼ揃う。参照画像の一方の位置の見た目が、現在画像のもう一方の位置に
    現れていれば、文字内容の変化ではなく移動とみなす。

    塊が 1 つだけの場合は、1 文字の編集と区別できないため移動とみなさない。
    """
    if len(blobs) < 2:
        return False
    a, b = blobs[0], blobs[1]
    # 上位 2 つが変化の大半を占める場合だけ、移動として扱う。
    rest = sum(x[4] for x in blobs[2:])
    if rest > (a[4] + b[4]) * 0.3:
        return False
    area_a, area_b = a[4], b[4]
    if min(area_a, area_b) <= 0 or max(area_a, area_b) > min(area_a, area_b) * 2.5:
        return False

    def patch(img: np.ndarray, box: tuple[int, int, int, int, int]) -> np.ndarray:
        return img[box[1] : box[3], box[0] : box[2]]

    forward = _patch_similarity(patch(reference, a), patch(current, b))
    backward = _patch_similarity(patch(reference, b), patch(current, a))
    return min(forward, backward) <= max_similarity


def _region_consistent(
    current: np.ndarray,
    reference: np.ndarray,
    bbox: tuple[int, int, int, int],
    pixel_delta: int,
    reference_changed_px: int,
) -> bool:
    """保留中の小変化が、同じ内容のまま続いているか。

    続いていれば実際の変更 (1 文字編集など)。内容が毎回変わるなら、点滅や
    符号化ノイズ、時計のような周期的表示の可能性が高い。

    許容量は「保留中の変化の画素数」を基準にする。範囲の面積を基準にすると、
    広く散らばった変化に対して許容量が過大になり、何でも一致と判定してしまう。
    """
    x0, y0, x1, y1 = bbox
    a = current[y0:y1, x0:x1]
    b = reference[y0:y1, x0:x1]
    if a.size == 0 or a.shape != b.shape:
        return False
    changed = int((cv2.absdiff(a, b) > pixel_delta).sum())
    return changed <= max(2, reference_changed_px // 5)


def _changed_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _bbox_union(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _bbox_near(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], shape: tuple[int, ...], margin_ratio: float = 0.04
) -> bool:
    """2 つの変化範囲が、同じ編集の一部とみなせる近さにあるか。"""
    mx = int(shape[1] * margin_ratio)
    my = int(shape[0] * margin_ratio)
    return not (
        a[2] + mx < b[0] or b[2] + mx < a[0] or a[3] + my < b[1] or b[3] + my < a[1]
    )


def _apply_ignore_mask(img: np.ndarray, ignore_regions: list[list[float]]) -> np.ndarray:
    if not ignore_regions:
        return img
    out = img.copy()
    h, w = out.shape[:2]
    for region in ignore_regions:
        x0, y0, x1, y1 = region
        out[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)] = 0
    return out


class FrameScanner:
    """1 本の動画をひと続きに走査し、表示状態を切り出す。"""

    def __init__(self, cfg: RunConfig, probe: MediaProbe, media_id: str, work_dir: Path):
        self.cfg = cfg
        self.scan: ScanConfig = cfg.scan
        self.probe = probe
        self.media_id = media_id
        self.frames_dir = work_dir / "frames" / media_id
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.visual_events: list[VisualEvent] = []
        self.stats: dict[str, Any] = {
            "decoded_frames": 0,
            "processed_frames": 0,
            "change_candidates": 0,
            "cursor_events": 0,
            "small_promoted": 0,
            "periodic_boundaries": 0,
            "timestamp_gaps": 0,
            "pointer_moves": 0,
        }

    # ------------------------------------------------------------------ scan
    def scan_states(
        self, start_us: int, end_us: int, *, resume_from_us: int | None = None
    ) -> Iterator[ScreenState]:
        """[start_us, end_us) を走査し、確定した表示状態を順に返す。"""
        scan = self.scan
        t0 = self.probe.time_origin_us
        video_index = self.probe.video.index if self.probe.video else 0

        container = av.open(self.probe.path)
        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            time_base = stream.time_base or Fraction(1, 1000)
            width = self.probe.video.width or stream.codec_context.width
            height = self.probe.video.height or stream.codec_context.height
            dw, dh = _target_small_size(width, height, scan.diff_width)

            seek_to_us = resume_from_us if resume_from_us is not None else start_us
            if seek_to_us > 0:
                # PTS 単位へ戻してから seek する。キーフレーム前へ戻るので後段で読み捨てる。
                target_pts = int((Fraction(seek_to_us + t0, US) / time_base))
                try:
                    container.seek(max(0, target_pts), stream=stream, backward=True, any_frame=False)
                except (av.AVError, ValueError) as exc:  # pragma: no cover - 環境依存
                    log.warning("seek に失敗しました (%s)。先頭から走査します。", exc)

            state_index = 0
            state: ScreenState | None = None
            ref_small: np.ndarray | None = None
            last_t: int | None = None
            last_match_t: int | None = None  # 参照画像と一致した最後の時刻
            last_sample_t: int | None = None
            pending: dict[str, Any] | None = None
            min_process_gap = int(US / scan.fast_fps) if scan.mode == "fast" else 0
            last_processed_t: int | None = None
            deltas: list[int] = []

            for frame in container.decode(video=0):
                self.stats["decoded_frames"] += 1
                if frame.pts is None:
                    # 時刻の無いフレームは時間軸に載せられない。件数だけ数える。
                    self.stats["timestamp_gaps"] += 1
                    continue
                t_us = pts_to_us(frame.pts, time_base, t0)
                if t_us < seek_to_us:
                    continue
                if t_us >= end_us:
                    break
                if last_processed_t is not None and min_process_gap and t_us - last_processed_t < min_process_gap:
                    continue
                prev_frame_t = last_t
                if last_t is not None:
                    delta = t_us - last_t
                    if delta > 0:
                        deltas.append(delta)
                last_t = t_us
                last_processed_t = t_us
                self.stats["processed_frames"] += 1

                small = frame.reformat(width=dw, height=dh, format="gray").to_ndarray()
                small = _apply_ignore_mask(small, scan.ignore_regions)

                if state is None:
                    state = self._open_state(
                        state_index,
                        start_us=t_us,
                        boundary_lo=t_us,
                        boundary_hi=t_us,
                        trigger="resume" if resume_from_us else "start",
                    )
                    ref_small = small
                    last_match_t = t_us
                    last_sample_t = None
                    self._maybe_sample(state, frame, small, t_us, last_sample_t)
                    last_sample_t = t_us
                    state.is_blank = self._is_blank(small)
                    continue

                assert ref_small is not None
                diff = cv2.absdiff(small, ref_small)
                raw_mask = diff > scan.pixel_delta
                mask = _significant_mask(raw_mask, scan.noise_density_window, scan.min_local_density)
                mask = _drop_border(mask, scan.border_margin_px)
                changed_px = int(mask.sum())
                global_ratio = changed_px / float(mask.size)
                counts, areas = _tile_counts(mask, scan.tile_cols, scan.tile_rows)
                tile_hit_mask = (counts >= scan.min_changed_pixels_tile) | (
                    counts / np.maximum(1, areas) > scan.tile_change_ratio
                )
                changed_tiles = int(tile_hit_mask.sum())
                is_change = (
                    changed_px >= scan.min_changed_pixels
                    or global_ratio > scan.global_change_ratio
                    or changed_tiles >= scan.min_changed_tiles
                )

                if not is_change:
                    last_match_t = t_us
                    if pending is not None:
                        # 変化が元へ戻った。カーソル点滅など、文字内容以外の変化とみなす。
                        kind = "unstable_small_region" if pending.get("unstable", 0) > 2 else "cursor"
                        self._record_cursor_event(pending, t_us, kind)
                        pending = None
                    if last_sample_t is None or t_us - last_sample_t >= SAMPLE_MIN_GAP_US:
                        self._maybe_sample(state, frame, small, t_us, last_sample_t)
                        last_sample_t = t_us
                    # 定期的な境界を入れて、見落としを後段の全文再認識で拾えるようにする。
                    if scan.periodic_full_recheck_us and (
                        t_us - state.start_us >= scan.periodic_full_recheck_us
                    ):
                        state.end_us = t_us
                        state.boundary_end_lo_us = t_us
                        state.boundary_end_hi_us = t_us
                        self._finalize(state)
                        yield state
                        self.stats["periodic_boundaries"] += 1
                        state_index += 1
                        state = self._open_state(
                            state_index, t_us, t_us, t_us, trigger="periodic"
                        )
                        state.flags.append("periodic_recheck")
                        ref_small = small
                        last_sample_t = None
                        self._maybe_sample(state, frame, small, t_us, last_sample_t)
                        last_sample_t = t_us
                    continue

                bbox = _changed_bbox(mask)
                small_change = bbox is not None and changed_px <= scan.cursor_max_pixels

                # --- ポインタ・カーソルの移動は文字内容の変化と分ける (設計 5.5) ---
                if small_change and scan.pointer_move_similarity > 0:
                    blobs = _blobs(mask)
                    if _looks_like_pointer_move(
                        small, ref_small, blobs, max_similarity=scan.pointer_move_similarity
                    ):
                        self.stats["pointer_moves"] += 1
                        self.visual_events.append(
                            VisualEvent(
                                id=new_id("vev"),
                                media_id=self.media_id,
                                start_us=t_us,
                                end_us=t_us,
                                region_ref=";".join(
                                    ",".join(str(v) for v in b[:4]) for b in blobs
                                ),
                                event_kind="pointer",
                                annotation="同じ見た目の小片が別位置へ移動したため、文字内容の変化としては扱わない",
                            )
                        )
                        # 移動後の見た目を基準にし、同じ移動を繰り返し検出しない。
                        ref_small = small
                        last_match_t = t_us
                        if pending is not None:
                            self._record_cursor_event(pending, t_us, "cursor")
                            pending = None
                        continue

                trigger_flags: list[str] = []
                change_bbox = bbox
                carried_samples: list[FrameSample] = []
                emit_intermediate = False
                intermediate_ratio = 0.0
                intermediate_tiles = 0

                # --- 小さな変化: カーソル点滅と 1 文字編集を取り違えない (設計 5.5) ---
                if small_change:
                    assert bbox is not None
                    if pending is None:
                        pending = {
                            "start_us": t_us,
                            "prev_match_us": last_match_t if last_match_t is not None else t_us,
                            "bbox": bbox,
                            "frame": small.copy(),
                            "changed_px": changed_px,
                            "ratio": global_ratio,
                            "tiles": changed_tiles,
                            "unstable": 0,
                            "samples": [],
                            "last_sample_t": None,
                        }
                        self._sample_into(pending, frame, small, t_us)
                        continue
                    union = _bbox_union(pending["bbox"], bbox)
                    elapsed = t_us - pending["start_us"]
                    grew = changed_px > 3 * pending["changed_px"] + scan.min_changed_pixels
                    consistent = _region_consistent(
                        small, pending["frame"], union, scan.pixel_delta, pending["changed_px"]
                    )
                    if grew:
                        # 変化が広がっている = 入力が進んでいる。実変化として扱う。
                        trigger_flags.append("small_change_grew")
                    elif elapsed >= scan.cursor_revert_us and consistent:
                        # 同じ内容のまま元へ戻らない = 1 文字編集などの実変化。
                        trigger_flags.append("small_persistent")
                        self.stats["small_promoted"] += 1
                    elif elapsed >= scan.cursor_revert_us * 4:
                        # 長く続く差分は、内容が揺れていても実変化として残す。
                        trigger_flags.append("small_unstable_promoted")
                        self.stats["small_promoted"] += 1
                    else:
                        pending["bbox"] = union
                        pending["ratio"] = max(pending["ratio"], global_ratio)
                        pending["tiles"] = max(pending["tiles"], changed_tiles)
                        if not consistent:
                            pending["unstable"] += 1
                            pending["frame"] = small.copy()
                        self._sample_into(pending, frame, small, t_us)
                        continue
                    change_t = pending["start_us"]
                    prev_match = pending["prev_match_us"]
                    ratio = max(pending["ratio"], global_ratio)
                    tiles_n = max(pending["tiles"], changed_tiles)
                    carried_samples = list(pending["samples"])
                    change_bbox = _bbox_union(pending["bbox"], bbox) if bbox else pending["bbox"]
                    pending = None
                else:
                    change_t = t_us
                    prev_match = last_match_t if last_match_t is not None else t_us
                    if pending is not None:
                        # 大きな変化が来た。保留中の小変化が同じ編集の一部なら、
                        # 独立した表示状態として確定させる。無関係な点滅なら取り込まない。
                        # どちらの場合も、変更の前後の画像を 1 つの状態へ混ぜない (設計 6.1)。
                        assert bbox is not None
                        localized = changed_px < mask.size * 0.2
                        if localized and _bbox_near(pending["bbox"], bbox, mask.shape):
                            emit_intermediate = True
                            change_t = pending["start_us"]
                            prev_match = pending["prev_match_us"]
                            carried_samples = list(pending["samples"])
                            intermediate_ratio = pending["ratio"]
                            intermediate_tiles = pending["tiles"]
                        else:
                            kind = (
                                "unstable_small_region" if pending.get("unstable", 0) > 2 else "cursor"
                            )
                            self._record_cursor_event(pending, t_us, kind)
                            if prev_frame_t is not None:
                                # 小変化を除けば直前フレームまで内容は同じだった。
                                prev_match = max(prev_match, prev_frame_t)
                        pending = None
                    ratio = global_ratio
                    tiles_n = changed_tiles

                self.stats["change_candidates"] += 1
                state.end_us = change_t
                state.boundary_end_lo_us = prev_match
                state.boundary_end_hi_us = change_t
                self._finalize(state)
                yield state
                state_index += 1

                if emit_intermediate:
                    # 保留中だった小変化を、それ自体の表示期間として残す。
                    mid_lo = prev_frame_t if prev_frame_t is not None else t_us
                    mid = self._open_state(
                        state_index,
                        start_us=change_t,
                        boundary_lo=prev_match,
                        boundary_hi=change_t,
                        trigger="small_pending",
                    )
                    mid.samples = carried_samples
                    mid.change_ratio = intermediate_ratio
                    mid.changed_tiles = intermediate_tiles
                    mid.end_us = t_us
                    mid.boundary_end_lo_us = mid_lo
                    mid.boundary_end_hi_us = t_us
                    mid.flags.append("small_change_confirmed_by_next_change")
                    self.stats["small_promoted"] += 1
                    self._finalize(mid)
                    yield mid
                    state_index += 1
                    carried_samples = []
                    new_start = t_us
                    new_lo = mid_lo
                else:
                    new_start = change_t
                    new_lo = prev_match

                state = self._open_state(
                    state_index,
                    start_us=new_start,
                    boundary_lo=new_lo,
                    boundary_hi=new_start,
                    trigger="diff",
                )
                state.change_ratio = ratio
                state.changed_tiles = tiles_n
                state.flags.extend(trigger_flags)
                state.samples = carried_samples
                if change_bbox is not None:
                    mh, mw = mask.shape
                    state.change_bbox_norm = [
                        change_bbox[0] / mw,
                        change_bbox[1] / mh,
                        change_bbox[2] / mw,
                        change_bbox[3] / mh,
                    ]
                ref_small = small
                last_match_t = t_us
                last_sample_t = None
                self._maybe_sample(state, frame, small, t_us, last_sample_t)
                last_sample_t = t_us
                state.is_blank = self._is_blank(small)

            if state is not None:
                state.end_us = end_us
                state.boundary_end_lo_us = last_t if last_t is not None else end_us
                state.boundary_end_hi_us = end_us
                self._finalize(state)
                yield state

            if deltas:
                median = sorted(deltas)[len(deltas) // 2]
                self.stats["timestamp_gaps"] += sum(1 for d in deltas if d > median * 3)
                self.stats["median_frame_delta_us"] = median
        finally:
            container.close()

    # --------------------------------------------------------------- helpers
    def _open_state(
        self, index: int, start_us: int, boundary_lo: int, boundary_hi: int, trigger: str
    ) -> ScreenState:
        return ScreenState(
            index=index,
            start_us=start_us,
            end_us=start_us,
            boundary_start_lo_us=boundary_lo,
            boundary_start_hi_us=boundary_hi,
            boundary_end_lo_us=start_us,
            boundary_end_hi_us=start_us,
            trigger=trigger,
        )

    def _is_blank(self, small: np.ndarray) -> bool:
        return bool(small.std() < 2.0)

    def _maybe_sample(
        self, state: ScreenState, frame, small: np.ndarray, t_us: int, last_sample_t: int | None
    ) -> None:
        if len(state.samples) >= self.scan.representative_samples:
            return
        if last_sample_t is not None and t_us - last_sample_t < SAMPLE_MIN_GAP_US:
            return
        sharpness = float(cv2.Laplacian(small, cv2.CV_32F).var())
        try:
            rgb = frame.to_ndarray(format="bgr24")
        except Exception as exc:  # pragma: no cover - デコード失敗時
            log.warning("フレームの変換に失敗しました t=%dus: %s", t_us, exc)
            return
        ok, buf = cv2.imencode(".png", rgb, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            return
        state.samples.append(
            FrameSample(
                t_us=t_us,
                pts=frame.pts,
                sharpness=sharpness,
                png=buf.tobytes(),
                width=rgb.shape[1],
                height=rgb.shape[0],
            )
        )

    def _sample_into(self, pending: dict[str, Any], frame, small: np.ndarray, t_us: int) -> None:
        """保留中の小変化の期間からも代表フレーム候補を取る。

        後で独立した表示状態として確定した場合に、その期間内の画像を使えるようにする。
        """
        holder = ScreenState(
            index=-1,
            start_us=pending["start_us"],
            end_us=t_us,
            boundary_start_lo_us=pending["start_us"],
            boundary_start_hi_us=pending["start_us"],
            boundary_end_lo_us=t_us,
            boundary_end_hi_us=t_us,
            samples=pending["samples"],
        )
        self._maybe_sample(holder, frame, small, t_us, pending.get("last_sample_t"))
        if holder.samples and (
            pending.get("last_sample_t") is None or holder.samples[-1].t_us == t_us
        ):
            pending["last_sample_t"] = t_us
        pending["samples"] = holder.samples

    def _record_cursor_event(self, pending: dict[str, Any], end_us: int, kind: str = "cursor") -> None:
        self.stats["cursor_events"] += 1
        self.visual_events.append(
            VisualEvent(
                id=new_id("vev"),
                media_id=self.media_id,
                start_us=pending["start_us"],
                end_us=end_us,
                region_ref=",".join(str(v) for v in pending["bbox"]),
                event_kind=kind,
                annotation="小さな変化が元へ戻ったため、文字内容の変化としては扱わない",
            )
        )

    def _finalize(self, state: ScreenState) -> None:
        duration = state.duration_us
        if state.is_blank:
            state.state_kind = STATE_BLANK
        elif duration < self.scan.stable_us:
            state.state_kind = STATE_SHORT
            state.flags.append(FLAG_SHORT_STATE)
        else:
            state.state_kind = STATE_STABLE

    # ------------------------------------------------------------ persistence
    def save_representative(self, state: ScreenState) -> tuple[str, bytes] | None:
        """代表フレームを保存し、(相対パス, png) を返す。

        ぼけ・カーソルの被りが少ない画像を選ぶ (設計 6.1)。ただし、文字が変更された
        前後の画像を混ぜないよう、選ぶ対象はこの状態の期間内に限る。
        """
        if not state.samples:
            return None
        candidates = state.samples[1:] if len(state.samples) > 1 else state.samples
        best = max(candidates, key=lambda s: s.sharpness)
        digest = sha256_bytes(best.png)
        rel = Path("frames") / self.media_id / f"{best.t_us:012d}_{digest[:12]}.png"
        abs_path = self.frames_dir.parent.parent / rel
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        if not abs_path.exists():
            tmp = abs_path.with_suffix(".png.partial")
            tmp.write_bytes(best.png)
            tmp.replace(abs_path)
        return str(rel).replace("\\", "/"), best.png


def run_frame_scan(
    store: Store,
    cfg: RunConfig,
    probe: MediaProbe,
    media_id: str,
    start_us: int,
    end_us: int,
    job_id: str,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    """走査を実行し、表示期間を screen_occurrence として保存する。

    まだ文字内容は確定していないので content_id は NULL のままにする。
    「認識失敗の区間を空欄の成功として保存しない」(設計 9.2) ため、
    未抽出であることを state_kind / quality_flags に必ず残す。
    """
    work_dir = Path(cfg.work_dir)
    # 保存領域が不足していれば、書き始める前に停止する (設計 12.2)。
    check_free_space(work_dir)
    scanner = FrameScanner(cfg, probe, media_id, work_dir)

    job = store.get_job(media_id, "frame_scan")
    checkpoint = dict(job["checkpoint"]) if (job and resume) else {}
    resume_from = checkpoint.get("last_committed_us")
    committed = int(checkpoint.get("committed_states", 0))
    if resume_from:
        log.info("frame_scan を %.3fs から再開します (確定済み状態 %d 件)", resume_from / 1e6, committed)
    else:
        store.clear_occurrences(media_id)
        store.clear_visual_events(media_id)

    short_runs: list[ScreenState] = []
    n_states = committed
    saved_occurrences: list[tuple[int, int]] = []

    def flush_burst(states: list[ScreenState]) -> None:
        """短い状態の連続は遷移として扱うが、期間は必ず残す (設計 5.4)。"""
        if len(states) >= BURST_MIN_LEN:
            for s in states[:-1]:
                s.state_kind = STATE_TRANSITION
                if "burst" not in s.flags:
                    s.flags.append("burst")

    pending_short: list[ScreenState] = []

    for state in scanner.scan_states(start_us, end_us, resume_from_us=resume_from):
        # 短い状態の連続 (アニメーション・スクロール) をまとめて種別付けする。
        if state.duration_us <= BURST_STATE_US:
            pending_short.append(state)
            continue
        flush_burst(pending_short)
        for s in pending_short:
            _persist_state(store, scanner, media_id, s)
            n_states += 1
            saved_occurrences.append((s.start_us, s.end_us))
        pending_short = []
        _persist_state(store, scanner, media_id, state)
        n_states += 1
        saved_occurrences.append((state.start_us, state.end_us))
        store.update_checkpoint(
            job_id, {"last_committed_us": state.end_us, "committed_states": n_states}
        )
        if n_states % 25 == 0:
            check_free_space(work_dir)

    flush_burst(pending_short)
    for s in pending_short:
        _persist_state(store, scanner, media_id, s)
        n_states += 1
        saved_occurrences.append((s.start_us, s.end_us))

    for ev in scanner.visual_events:
        store.add_visual_event(ev)

    store.update_checkpoint(job_id, {"last_committed_us": end_us, "committed_states": n_states})

    # 画面側の時間軸網羅を更新する。表示期間として切り出せた範囲は「未抽出だが期間は確定」。
    spans: list[CoverageSpan] = []
    if start_us > 0:
        spans.append(CoverageSpan(new_id("cov"), media_id, "screen", 0, start_us, "out_of_range", "解析範囲外"))
    cursor = start_us
    for s_start, s_end in sorted(saved_occurrences):
        if s_start > cursor:
            spans.append(
                CoverageSpan(new_id("cov"), media_id, "screen", cursor, s_start, "unextracted", "表示状態を切り出せていない区間")
            )
        spans.append(
            CoverageSpan(new_id("cov"), media_id, "screen", s_start, max(s_start, s_end), "unextracted", "期間は確定、本文は未抽出")
        )
        cursor = max(cursor, s_end)
    if cursor < end_us:
        spans.append(CoverageSpan(new_id("cov"), media_id, "screen", cursor, end_us, "unextracted", "表示状態を切り出せていない区間"))
    if end_us < probe.duration_us:
        spans.append(
            CoverageSpan(new_id("cov"), media_id, "screen", end_us, probe.duration_us, "out_of_range", "解析範囲外")
        )
    store.replace_coverage(media_id, "screen", spans)

    stats = dict(scanner.stats)
    stats["states"] = n_states
    for name in ("decoded_frames", "processed_frames", "change_candidates", "cursor_events", "states"):
        store.add_metric(media_id, "frame_scan", name, float(stats.get(name, 0)))
    log.info("frame_scan 完了: %s", stats)
    return stats


def _persist_state(store: Store, scanner: FrameScanner, media_id: str, state: ScreenState) -> str:
    saved = scanner.save_representative(state)
    evidence_refs: list[dict[str, Any]] = []
    if saved is not None:
        rel, png = saved
        best = max(state.samples[1:] if len(state.samples) > 1 else state.samples, key=lambda s: s.sharpness)
        frame_id = store.add_frame(
            {
                "id": new_id("frm"),
                "media_id": media_id,
                "pts": best.pts,
                "time_base_num": None,
                "time_base_den": None,
                "t_us": best.t_us,
                "image_hash": sha256_bytes(png),
                "crop": None,
                "evidence_ref": rel,
                "kind": "representative",
                "width": best.width,
                "height": best.height,
            }
        )
        evidence_refs.append({"frame_id": frame_id, "path": rel, "t_us": best.t_us, "role": "representative"})
        # 追加の候補画像も、再認識用に時刻だけ残しておく。
        for s in state.samples:
            if s.t_us != best.t_us:
                evidence_refs.append({"t_us": s.t_us, "role": "alternate", "sharpness": s.sharpness})

    if state.change_bbox_norm is not None:
        evidence_refs.append({"role": "change_region", "bbox_norm": state.change_bbox_norm})
    flags = list(state.flags)
    flags.append(FLAG_NOT_EXTRACTED)
    occ = ScreenOccurrence(
        id=new_id("occ"),
        media_id=media_id,
        content_id=None,
        start_us=state.start_us,
        end_us=state.end_us,
        boundary_start_lo_us=state.boundary_start_lo_us,
        boundary_start_hi_us=state.boundary_start_hi_us,
        boundary_end_lo_us=state.boundary_end_lo_us,
        boundary_end_hi_us=state.boundary_end_hi_us,
        state_kind=state.state_kind,
        evidence_refs=evidence_refs,
        change_summary=f"trigger={state.trigger} ratio={state.change_ratio:.5f} tiles={state.changed_tiles}",
        quality_flags=flags,
    )
    return store.upsert_occurrence(occ)


def should_extract(state_kind: str, duration_us: int) -> bool:
    """VLM 抽出の対象にするか。対象外でも期間は必ず保存されている。"""
    if state_kind == STATE_BLANK:
        return False
    if state_kind == STATE_TRANSITION:
        return False
    return duration_us >= EXTRACT_MIN_STATE_US
