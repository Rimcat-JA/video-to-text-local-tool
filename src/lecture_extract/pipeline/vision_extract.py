"""VLM による画面全文の抽出 (設計 6)。

処理手順 (設計 6.1):
 1. 画面全体で一度読み取り、すべての文字領域が対象に入っているか確認する。
 2. 細かい文字・多段組み・コードは領域別に原画像を切り出して読む。
 3. 長い領域は重なりを持つタイルへ分割し、重複部分の一致で結合する。
 4. 認識結果をスキーマ検証し、不明箇所を局所的に再試行する。
 5. 原文、構造、根拠画像、再試行履歴を保存する。

キャッシュ鍵は設計 12.1 の組み合わせで作る。類似画像を同一本文と断定しない。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import RunConfig
from ..db.store import Store, new_id
from ..models import (
    FLAG_LOW_RESOLUTION,
    FLAG_MERGED_TILES,
    FLAG_NOT_EXTRACTED,
    FLAG_SCHEMA_INVALID,
    FLAG_TRUNCATED,
    FLAG_UNREADABLE,
    ROLE_MATERIAL_BODY,
    ROLE_MATERIAL_CONTEXT,
    STATE_BLANK,
    CoverageSpan,
    Region,
    ReviewItem,
    ScreenContent,
    ScreenOccurrence,
)
from ..util.hashing import config_hash, sha256_bytes, sha256_text
from ..util.textnorm import normalize_for_compare
from ..adapters.vision.base import (
    STATUS_ERROR,
    STATUS_OK,
    STATUS_SCHEMA_INVALID,
    STATUS_TRUNCATED,
    VisionResult,
)
from ..adapters.vision.prompts import PROMPT_VERSION, schema_for
from .frame_scan import should_extract

log = logging.getLogger(__name__)

CODE_KINDS = {"code", "terminal_output"}
SCHEMA_VERSION = sha256_text(json.dumps(schema_for("full"), sort_keys=True))[:12]


def content_signature(regions: list[Region]) -> str:
    """本文の同一性判定に使う署名。UI 文字は本文境界に使わない (設計 5.1)。"""
    parts = []
    for region in sorted(regions, key=lambda r: r.reading_order):
        if region.role not in (ROLE_MATERIAL_BODY, ROLE_MATERIAL_CONTEXT):
            continue
        is_code = region.kind in CODE_KINDS
        parts.append(f"{region.kind}\x1f{normalize_for_compare(region.text, is_code=is_code)}")
    return "\x1e".join(parts)


def build_body_text(regions: list[Region]) -> str:
    return "\n\n".join(
        r.text for r in sorted(regions, key=lambda r: r.reading_order) if r.role == ROLE_MATERIAL_BODY
    )


def _clamp_bbox(bbox: list[float], width: int, height: int, padding: float) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    if max(abs(v) for v in bbox) <= 1.5:  # 正規化座標
        x0, x1 = x0 * width, x1 * width
        y0, y1 = y0 * height, y1 * height
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    pad_x = (x1 - x0) * padding + 4
    pad_y = (y1 - y0) * padding + 4
    x0 = int(max(0, x0 - pad_x))
    y0 = int(max(0, y0 - pad_y))
    x1 = int(min(width, x1 + pad_x))
    y1 = int(min(height, y1 + pad_y))
    if x1 <= x0:
        x1 = min(width, x0 + 1)
    if y1 <= y0:
        y1 = min(height, y0 + 1)
    return x0, y0, x1, y1


def merge_tile_lines(acc: list[str], new: list[str], max_overlap: int) -> tuple[list[str], bool]:
    """タイル境界を重複行の一致で結合する (設計 6.1)。

    一致が見つからなかった場合は False を返し、結合が未検証であることを残す。
    """
    if not acc:
        return list(new), True
    limit = min(len(acc), len(new), max(1, max_overlap))
    for k in range(limit, 0, -1):
        if [line.rstrip() for line in acc[-k:]] == [line.rstrip() for line in new[:k]]:
            return acc + new[k:], True
    return acc + new, False


def _encode_png(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError("PNG へのエンコードに失敗しました")
    return buf.tobytes()


class VisionExtractor:
    def __init__(self, store: Store, cfg: RunConfig, adapter, work_dir: Path):
        self.store = store
        self.cfg = cfg
        self.vcfg = cfg.vision
        self.adapter = adapter
        self.work_dir = Path(work_dir)
        self.responses_dir = self.work_dir / "responses"
        self.responses_dir.mkdir(parents=True, exist_ok=True)
        self.model_info = adapter.describe()
        self.stats: dict[str, Any] = {
            "occurrences": 0,
            "extracted": 0,
            "skipped_short": 0,
            "skipped_blank": 0,
            "cache_hits": 0,
            "requests": 0,
            "failed": 0,
            "truncated": 0,
            "crop_rereads": 0,
            "tiles": 0,
            "reread_mismatch": 0,
            "region_checks": 0,
            "region_merged": 0,
            "region_changed": 0,
            "server_restarts": 0,
        }

    # ------------------------------------------------------------------ keys
    def cache_key(self, image_hash: str, crop: tuple[int, int, int, int] | None, preprocess: dict[str, Any], kind: str) -> str:
        return config_hash(
            {
                "image_hash": image_hash,
                "crop": list(crop) if crop else None,
                "preprocess": preprocess,
                "model_sha256": self.model_info.get("model_sha256", ""),
                "mmproj_sha256": self.model_info.get("mmproj_sha256", ""),
                "runtime_version": self.model_info.get("runtime_version", ""),
                "prompt_version": PROMPT_VERSION,
                "schema_version": SCHEMA_VERSION,
                "params_hash": self.model_info.get("params_hash", ""),
                "request_kind": kind,
            }
        )

    # ------------------------------------------------------------- inference
    def _call(
        self,
        image: np.ndarray,
        kind: str,
        frame_id: str,
        image_hash: str,
        crop: tuple[int, int, int, int] | None,
        preprocess: dict[str, Any],
        *,
        extra_instruction: str = "",
    ) -> tuple[dict[str, Any] | None, str, str]:
        """(payload, status, attempt_id) を返す。キャッシュと再試行を扱う。"""
        key = self.cache_key(image_hash, crop, preprocess, kind)
        cached = self.store.cache_get(key)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return cached.get("payload"), cached.get("status", STATUS_OK), cached.get("attempt_id", "")

        png = _encode_png(image)
        attempts = 0
        result: VisionResult | None = None
        attempt_id = ""
        while attempts <= self.vcfg.retry_limit:
            attempts += 1
            self.stats["requests"] += 1
            result = self.adapter.extract(png, kind, extra_instruction=extra_instruction)
            attempt_id = self._record_attempt(frame_id, key, kind, result, crop)
            if result.status == STATUS_OK:
                break
            if result.status == STATUS_ERROR:
                # 通信・プロセス障害の再試行 (設計 12.2)。
                log.warning("推論に失敗しました (%d/%d): %s", attempts, self.vcfg.retry_limit + 1, result.error)
                # 管理下のサーバーが落ちている場合は起動し直す。長時間実行では
                # サーバー側が落ちることがあり、再試行だけでは復帰できない。
                self._recover_server()
                time.sleep(min(5.0, 1.0 * attempts))
                continue
            if result.status == STATUS_SCHEMA_INVALID and attempts <= self.vcfg.retry_limit:
                # 認識結果の疑義に対する再認識。同じ失敗を繰り返すだけのループにしない。
                log.warning("スキーマ検証に失敗しました。再認識します: %s", result.error)
                continue
            break

        assert result is not None
        if result.status == STATUS_OK:
            self.store.cache_put(key, kind, {"payload": result.payload, "status": result.status, "attempt_id": attempt_id}, attempt_id)
        return result.payload, result.status, attempt_id

    def _recover_server(self) -> None:
        """推論サーバーが応答しなくなった場合に、起動し直す。"""
        ensure_ready = getattr(self.adapter, "ensure_ready", None)
        health = getattr(self.adapter, "health", None)
        if ensure_ready is None:
            return
        try:
            if health is not None and health():
                return  # 生きている。障害は別の原因
            log.warning("推論サーバーが応答しません。起動し直します。")
            ensure_ready()
            self.stats["server_restarts"] = self.stats.get("server_restarts", 0) + 1
            log.info("推論サーバーを再起動しました。")
        except Exception as exc:  # noqa: BLE001 - 再起動に失敗しても次の再試行へ進む
            log.warning("推論サーバーの再起動に失敗しました: %s", exc)

    def _record_attempt(
        self,
        frame_id: str,
        cache_key: str,
        kind: str,
        result: VisionResult,
        crop: tuple[int, int, int, int] | None,
    ) -> str:
        attempt_id = new_id("att")
        rel = Path("responses") / f"{attempt_id}.json"
        (self.responses_dir / f"{attempt_id}.json").write_text(
            json.dumps(
                {
                    "status": result.status,
                    "finish_reason": result.finish_reason,
                    "error": result.error,
                    "usage": result.usage,
                    "crop": list(crop) if crop else None,
                    "request_kind": kind,
                    "raw": result.raw_text,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.store.add_attempt(
            {
                "id": attempt_id,
                "frame_id": frame_id,
                "cache_key": cache_key,
                "model_revision": self.model_info.get("model_revision", ""),
                "prompt_version": PROMPT_VERSION,
                "runtime_version": self.model_info.get("runtime_version", ""),
                "params_hash": self.model_info.get("params_hash", ""),
                "request_kind": kind,
                "raw_response_ref": str(rel).replace("\\", "/"),
                "status": result.status,
                "finish_reason": result.finish_reason,
                "error": result.error,
                "latency_ms": result.latency_ms,
            }
        )
        return attempt_id

    # ----------------------------------------------------------- occurrence
    def extract_occurrence(self, occ: ScreenOccurrence) -> tuple[ScreenContent | None, list[str]]:
        """1 つの表示期間の本文を抽出する。失敗しても空欄の成功にしない。"""
        flags: list[str] = []
        frame_ref = next((e for e in occ.evidence_refs if e.get("role") == "representative"), None)
        if frame_ref is None:
            return None, [FLAG_NOT_EXTRACTED, "no_representative_frame"]
        image_path = self.work_dir / frame_ref["path"]
        if not image_path.exists():
            return None, [FLAG_NOT_EXTRACTED, "evidence_image_missing"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return None, [FLAG_NOT_EXTRACTED, "evidence_image_unreadable"]
        image_hash = frame_ref.get("image_hash") or sha256_bytes(image_path.read_bytes())
        frame_id = frame_ref.get("frame_id", "")
        height, width = image.shape[:2]

        # --- 全画面パス ---
        full_image, scale = self._prepare_full(image)
        preprocess = {"max_long_side": self.vcfg.max_image_long_side, "scale": round(scale, 4)}
        payload, status, attempt_id = self._call(
            full_image, "full", frame_id, image_hash, None, preprocess
        )
        attempt_ids = [attempt_id] if attempt_id else []

        if status == STATUS_TRUNCATED:
            flags.append(FLAG_TRUNCATED)
            self.stats["truncated"] += 1
        if status == STATUS_SCHEMA_INVALID:
            flags.append(FLAG_SCHEMA_INVALID)
        if payload is None:
            # 全画面パスが成立しない場合、領域分割で処理可能にする (設計 6.2)。
            regions, band_flags, band_attempts = self._band_fallback(image, frame_id, image_hash)
            attempt_ids.extend(band_attempts)
            flags.extend(band_flags)
            if not regions:
                self.stats["failed"] += 1
                return None, flags + [FLAG_NOT_EXTRACTED]
            content = self._make_content(regions, "other", {}, [], flags, attempt_ids)
            return content, flags

        regions = self._payload_to_regions(payload, width, height, frame_id, attempt_id)

        # --- 領域別の原寸クロップ再認識 ---
        downscaled = scale < 0.999
        if downscaled:
            flags.append(FLAG_LOW_RESOLUTION)
        screen_kind_for_crop = payload.get("screen_kind", "other")
        code_screen = screen_kind_for_crop in self.vcfg.crop_reread_screen_kinds
        for region in regions:
            uncertain = bool(region.unreadable) or FLAG_UNREADABLE in region.flags
            need = (
                (region.kind in self.vcfg.crop_reread_kinds and code_screen)
                or uncertain
                or (
                    downscaled
                    and self.vcfg.crop_reread_all_body_when_downscaled
                    and region.role == ROLE_MATERIAL_BODY
                )
            )
            if not need:
                continue
            attempt_ids.extend(self._reread_region(image, region, frame_id, image_hash))

        screen_kind = payload.get("screen_kind", "other")
        context = payload.get("context") or {}
        structure_notes = payload.get("structure_notes") or []
        if any(FLAG_UNREADABLE in r.flags or r.unreadable for r in regions):
            flags.append(FLAG_UNREADABLE)
        content = self._make_content(regions, screen_kind, context, structure_notes, flags, attempt_ids)
        return content, flags

    # --------------------------------------------- 第 2 段階: 局所の再認識
    def region_text_unchanged(
        self, prev_occ: ScreenOccurrence, occ: ScreenOccurrence
    ) -> tuple[bool | None, list[str]]:
        """変化した領域だけを読み直し、文字が実際に変わったかを確かめる (設計 5.2)。

        戻り値は (変化なし?, attempt_ids)。判定できない場合は None を返し、
        統合を保留して全画面パスへ進む。
        """
        region = next((e for e in occ.evidence_refs if e.get("role") == "change_region"), None)
        if region is None or not region.get("bbox_norm"):
            return None, []
        prev_ref = next((e for e in prev_occ.evidence_refs if e.get("role") == "representative"), None)
        cur_ref = next((e for e in occ.evidence_refs if e.get("role") == "representative"), None)
        if prev_ref is None or cur_ref is None:
            return None, []

        x0n, y0n, x1n, y1n = region["bbox_norm"]
        if (x1n - x0n) * (y1n - y0n) > self.vcfg.region_check_max_area:
            return None, []  # 広い変化は全画面で読む

        attempt_ids: list[str] = []
        texts: list[str | None] = []
        for ref in (prev_ref, cur_ref):
            path = self.work_dir / ref["path"]
            if not path.exists():
                return None, attempt_ids
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                return None, attempt_ids
            height, width = image.shape[:2]
            x0, y0, x1, y1 = _clamp_bbox([x0n, y0n, x1n, y1n], width, height, self.vcfg.region_check_padding)
            crop = image[y0:y1, x0:x1]
            if crop.size == 0:
                return None, attempt_ids
            crop, upscale = self._maybe_upscale(crop)
            image_hash = ref.get("image_hash") or sha256_bytes(path.read_bytes())
            payload, status, attempt_id = self._call(
                crop,
                "crop",
                ref.get("frame_id", ""),
                image_hash,
                (x0, y0, x1, y1),
                {"upscale": round(upscale, 4), "purpose": "region_check", "padding": self.vcfg.region_check_padding},
            )
            if attempt_id:
                attempt_ids.append(attempt_id)
            if payload is None or status != STATUS_OK:
                return None, attempt_ids
            lines = list(payload.get("lines") or [])
            if not lines and payload.get("text"):
                lines = str(payload["text"]).split("\n")
            texts.append("\n".join(lines))

        self.stats["region_checks"] += 1
        # 記号・字下げを勝手に同一視しないため、コードとして厳密に比較する。
        same = normalize_for_compare(texts[0] or "", is_code=True) == normalize_for_compare(
            texts[1] or "", is_code=True
        )
        return same, attempt_ids

    def _prepare_full(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = image.shape[:2]
        long_side = max(width, height)
        limit = self.vcfg.max_image_long_side
        if limit <= 0 or long_side <= limit:
            return image, 1.0
        scale = limit / long_side
        resized = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        return resized, scale

    def _payload_to_regions(
        self, payload: dict[str, Any], width: int, height: int, frame_id: str, attempt_id: str
    ) -> list[Region]:
        regions: list[Region] = []
        for i, raw in enumerate(payload.get("regions") or []):
            bbox = raw.get("bbox") or [0.0, 0.0, 1.0, 1.0]
            x0, y0, x1, y1 = _clamp_bbox(list(bbox), width, height, 0.0)
            region = Region(
                region_id=new_id("reg"),
                kind=raw.get("kind", "unknown"),
                role=raw.get("role", ROLE_MATERIAL_BODY),
                text=raw.get("text", ""),
                bbox=[x0, y0, x1, y1],
                coord_space="original_px",
                reading_order=int(raw.get("reading_order", i)),
                line_numbers=list(raw.get("line_numbers") or []),
                language_hint=raw.get("language_hint", ""),
                flags=list(raw.get("flags") or []),
                unreadable=list(raw.get("unreadable") or []),
                candidates=list(raw.get("candidates") or []),
                evidence={"frame_id": frame_id, "attempt_id": attempt_id, "source": "full_pass"},
            )
            regions.append(region)
        return regions

    def _reread_region(
        self, image: np.ndarray, region: Region, frame_id: str, image_hash: str
    ) -> list[str]:
        """領域を原寸で切り出して読み直す。結果はこちらを原文として採用する。"""
        height, width = image.shape[:2]
        x0, y0, x1, y1 = _clamp_bbox([float(v) for v in region.bbox], width, height, self.vcfg.crop_padding_ratio)
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            return []
        self.stats["crop_rereads"] += 1
        crop, upscale = self._maybe_upscale(crop)
        tiles = self._split_tiles(crop)
        attempt_ids: list[str] = []
        lines: list[str] = []
        line_numbers: list[str] = []
        flags: list[str] = []
        unreadable: list[dict[str, Any]] = []
        candidates: list[str] = []
        joined_ok = True
        succeeded = 0
        overlap_lines = max(1, int(len(tiles) > 1) * 6)

        for index, (tile, tile_box) in enumerate(tiles):
            kind = "tile" if len(tiles) > 1 else "crop"
            if len(tiles) > 1:
                self.stats["tiles"] += 1
            preprocess = {
                "upscale": round(upscale, 4),
                "tile_index": index,
                "tile_count": len(tiles),
                "tile_box": list(tile_box),
                "padding": self.vcfg.crop_padding_ratio,
            }
            payload, status, attempt_id = self._call(
                tile,
                kind,
                frame_id,
                image_hash,
                (x0, y0, x1, y1),
                preprocess,
                extra_instruction=f"region_id={region.region_id}; region_kind={region.kind}",
            )
            if attempt_id:
                attempt_ids.append(attempt_id)
            if status == STATUS_TRUNCATED:
                flags.append(FLAG_TRUNCATED)
                self.stats["truncated"] += 1
            if payload is None:
                flags.append(FLAG_NOT_EXTRACTED)
                continue
            succeeded += 1
            tile_lines = list(payload.get("lines") or [])
            if payload.get("text") and not tile_lines:
                tile_lines = str(payload["text"]).split("\n")
            lines, ok = merge_tile_lines(lines, tile_lines, overlap_lines)
            joined_ok = joined_ok and ok
            line_numbers.extend(payload.get("line_numbers") or [])
            flags.extend(payload.get("flags") or [])
            unreadable.extend(payload.get("unreadable") or [])
            candidates.extend(payload.get("candidates") or [])

        if not attempt_ids:
            return []
        new_text = "\n".join(lines)
        old_text = region.text

        # クロップ再認識に失敗した場合、全画面パスで得た原文を空文字で上書きしない。
        # 認識失敗を「空欄の成功」として保存しないため (設計 9.2)。
        if succeeded < len(tiles) and (not lines or succeeded == 0):
            region.flags = sorted(set(region.flags) | set(flags) | {"crop_reread_failed"})
            region.evidence["crop_reread_status"] = "failed"
            region.evidence["attempt_ids"] = attempt_ids
            region.evidence["crop"] = [x0, y0, x1, y1]
            self.stats["crop_reread_failed"] = self.stats.get("crop_reread_failed", 0) + 1
            return attempt_ids
        if not new_text.strip() and old_text.strip():
            # 原寸で読めなかったのに全画面パスでは文字があった。原文は残し、確認対象にする。
            region.flags = sorted(set(region.flags) | set(flags) | {"crop_reread_empty"})
            region.evidence["crop_reread_status"] = "empty"
            region.evidence["attempt_ids"] = attempt_ids
            return attempt_ids
        if len(tiles) > 1:
            flags.append(FLAG_MERGED_TILES)
            if not joined_ok:
                flags.append("tile_join_unverified")
        is_code = region.kind in CODE_KINDS
        if normalize_for_compare(new_text, is_code=is_code) != normalize_for_compare(old_text, is_code=is_code):
            self.stats["reread_mismatch"] += 1
            flags.append("reread_differs_from_full_pass")
        region.evidence["full_pass_text"] = old_text
        region.evidence["source"] = "crop_reread"
        region.evidence["crop"] = [x0, y0, x1, y1]
        region.evidence["attempt_ids"] = attempt_ids
        region.text = new_text
        if line_numbers:
            region.line_numbers = line_numbers
        region.flags = sorted(set(region.flags) | set(flags))
        region.unreadable.extend(unreadable)
        region.candidates.extend(candidates)
        if region.unreadable and FLAG_UNREADABLE not in region.flags:
            region.flags.append(FLAG_UNREADABLE)
        return attempt_ids

    def _maybe_upscale(self, crop: np.ndarray) -> tuple[np.ndarray, float]:
        """小さすぎるクロップは拡大する。生成型の超解像は使わない (設計 6.2)。"""
        h, w = crop.shape[:2]
        factor = 1.0
        if h < 96:
            factor = min(3.0, 96 / max(1, h))
        if w * factor * h * factor > self.vcfg.crop_max_pixels:
            factor = max(1.0, (self.vcfg.crop_max_pixels / (w * h)) ** 0.5)
        if factor <= 1.001:
            return crop, 1.0
        return cv2.resize(crop, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_CUBIC), factor

    def _split_tiles(self, crop: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
        h, w = crop.shape[:2]
        limit = self.vcfg.tile_max_height_px
        if h <= limit:
            return [(crop, (0, 0, w, h))]
        step = max(1, int(limit * (1.0 - self.vcfg.tile_overlap_ratio)))
        tiles = []
        y = 0
        while y < h:
            y1 = min(h, y + limit)
            tiles.append((crop[y:y1, 0:w], (0, y, w, y1)))
            if y1 >= h:
                break
            y += step
        return tiles

    def _band_fallback(
        self, image: np.ndarray, frame_id: str, image_hash: str
    ) -> tuple[list[Region], list[str], list[str]]:
        """全画面パスが成立しない場合の領域分割 (設計 6.2)。"""
        height, width = image.shape[:2]
        bands = 3
        overlap = int(height * self.vcfg.tile_overlap_ratio / bands)
        regions: list[Region] = []
        attempt_ids: list[str] = []
        flags = ["fallback_band_split"]
        lines: list[str] = []
        joined_ok = True
        for i in range(bands):
            y0 = max(0, i * height // bands - overlap)
            y1 = min(height, (i + 1) * height // bands + overlap)
            band = image[y0:y1, 0:width]
            payload, status, attempt_id = self._call(
                band,
                "tile",
                frame_id,
                image_hash,
                (0, y0, width, y1),
                {"band_index": i, "band_count": bands},
            )
            if attempt_id:
                attempt_ids.append(attempt_id)
            if payload is None:
                flags.append(FLAG_NOT_EXTRACTED)
                continue
            band_lines = list(payload.get("lines") or [])
            lines, ok = merge_tile_lines(lines, band_lines, 6)
            joined_ok = joined_ok and ok
        if lines:
            if not joined_ok:
                flags.append("tile_join_unverified")
            regions.append(
                Region(
                    region_id=new_id("reg"),
                    kind="unknown",
                    role=ROLE_MATERIAL_BODY,
                    text="\n".join(lines),
                    bbox=[0, 0, width, height],
                    reading_order=0,
                    flags=[FLAG_MERGED_TILES, "fallback_band_split"],
                    evidence={"frame_id": frame_id, "attempt_ids": attempt_ids, "source": "band_fallback"},
                )
            )
        return regions, flags, attempt_ids

    def _make_content(
        self,
        regions: list[Region],
        screen_kind: str,
        context: dict[str, Any],
        structure_notes: list[dict[str, Any]],
        flags: list[str],
        attempt_ids: list[str],
    ) -> ScreenContent:
        signature = content_signature(regions)
        text_hash = sha256_text(f"{screen_kind}\x1d{signature}")
        existing = self.store.find_content_by_hash(text_hash)
        if existing is not None:
            # 設計 5.6: 本文の重複排除は行うが、出現は別に残す。
            return existing
        return ScreenContent(
            id=new_id("scr"),
            text_hash=text_hash,
            body_text=build_body_text(regions),
            regions=regions,
            reading_order=[r.region_id for r in sorted(regions, key=lambda r: r.reading_order)],
            screen_kind=screen_kind,
            context=context,
            structure_notes=structure_notes,
            quality_flags=sorted(set(flags)),
            source_attempt_ids=attempt_ids,
        )


def run_vision(
    store: Store,
    cfg: RunConfig,
    adapter,
    media_id: str,
    job_id: str,
    duration_us: int,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    work_dir = Path(cfg.work_dir)
    extractor = VisionExtractor(store, cfg, adapter, work_dir)
    occurrences = store.occurrences(media_id)
    job = store.get_job(media_id, "vision")
    done_ids = set(job["checkpoint"].get("done_occurrence_ids", [])) if (job and resume) else set()
    store.clear_reviews_by_reason_prefix(media_id, "vision:")

    extractor.stats["occurrences"] = len(occurrences)
    prev_extracted: ScreenOccurrence | None = None
    for i, occ in enumerate(occurrences):
        if occ.id in done_ids and occ.content_id:
            continue
        if occ.state_kind == STATE_BLANK:
            extractor.stats["skipped_blank"] += 1
            occ.quality_flags = sorted(set(occ.quality_flags) | {"blank_screen"})
            store.upsert_occurrence(occ)
            done_ids.add(occ.id)
            continue
        if not should_extract(
            occ.state_kind, occ.end_us - occ.start_us, cfg.vision.extract_min_state_us
        ):
            extractor.stats["skipped_short"] += 1
            occ.quality_flags = sorted(set(occ.quality_flags) | {FLAG_NOT_EXTRACTED, "short_or_transition"})
            store.upsert_occurrence(occ)
            store.add_review(
                ReviewItem(
                    id=new_id("rev"),
                    media_id=media_id,
                    target_kind="occurrence",
                    target_ref=occ.id,
                    reason="vision:not_extracted_short_state",
                    detail=(
                        f"表示時間 {(occ.end_us - occ.start_us) / 1000:.0f}ms の状態は全文未確定のまま期間だけ残しています。"
                    ),
                )
            )
            done_ids.add(occ.id)
            continue

        # 設計 5.2 第 2 段階: 変化した領域だけを先に読み、文字が同じなら全画面パスを省く。
        if cfg.vision.region_check and prev_extracted is not None:
            same, region_attempts = extractor.region_text_unchanged(prev_extracted, occ)
            if same is True:
                extractor.stats["region_merged"] += 1
                occ.content_id = prev_extracted.content_id
                occ.quality_flags = sorted(
                    (set(occ.quality_flags) | {"same_text_as_previous_by_region_check"})
                    - {FLAG_NOT_EXTRACTED}
                )
                occ.evidence_refs = occ.evidence_refs + [
                    {"role": "region_check", "attempt_ids": region_attempts, "result": "unchanged"}
                ]
                store.upsert_occurrence(occ)
                prev_extracted = occ
                done_ids.add(occ.id)
                continue
            if same is False:
                extractor.stats["region_changed"] += 1

        content, flags = extractor.extract_occurrence(occ)
        if content is None:
            extractor.stats["failed"] += 1
            occ.quality_flags = sorted(set(occ.quality_flags) | set(flags) | {FLAG_NOT_EXTRACTED})
            store.upsert_occurrence(occ)
            store.add_review(
                ReviewItem(
                    id=new_id("rev"),
                    media_id=media_id,
                    target_kind="occurrence",
                    target_ref=occ.id,
                    reason="vision:extraction_failed",
                    detail=f"抽出に失敗しました: {','.join(flags)}",
                )
            )
            done_ids.add(occ.id)
            continue

        store.upsert_content(content)
        occ.content_id = content.id
        occ.quality_flags = sorted((set(occ.quality_flags) | set(flags)) - {FLAG_NOT_EXTRACTED})
        store.upsert_occurrence(occ)
        extractor.stats["extracted"] += 1
        _add_content_reviews(store, media_id, occ, content)
        prev_extracted = occ
        done_ids.add(occ.id)

        if (i + 1) % 10 == 0:
            store.update_checkpoint(job_id, {"done_occurrence_ids": sorted(done_ids)})

    store.update_checkpoint(job_id, {"done_occurrence_ids": sorted(done_ids)})
    _update_screen_coverage(store, media_id, duration_us)

    for name, value in extractor.stats.items():
        store.add_metric(media_id, "vision", name, float(value))
    log.info("vision 完了: %s", extractor.stats)
    return dict(extractor.stats)


def _add_content_reviews(store: Store, media_id: str, occ: ScreenOccurrence, content: ScreenContent) -> None:
    for region in content.regions:
        if region.unreadable or FLAG_UNREADABLE in region.flags:
            store.add_review(
                ReviewItem(
                    id=new_id("rev"),
                    media_id=media_id,
                    target_kind="region",
                    target_ref=f"{occ.id}:{region.region_id}",
                    reason="vision:unreadable_text",
                    detail=json.dumps(region.unreadable, ensure_ascii=False)[:1000],
                )
            )
        if "reread_differs_from_full_pass" in region.flags:
            store.add_review(
                ReviewItem(
                    id=new_id("rev"),
                    media_id=media_id,
                    target_kind="region",
                    target_ref=f"{occ.id}:{region.region_id}",
                    reason="vision:reread_mismatch",
                    detail="全画面パスとクロップ再認識で文字が一致しません。クロップ側を原文として採用しています。",
                )
            )
        if "crop_reread_failed" in region.flags or "crop_reread_empty" in region.flags:
            store.add_review(
                ReviewItem(
                    id=new_id("rev"),
                    media_id=media_id,
                    target_kind="region",
                    target_ref=f"{occ.id}:{region.region_id}",
                    reason="vision:crop_reread_failed",
                    detail="原寸クロップでの読み直しに失敗しました。全画面パスの原文を残しています。",
                )
            )
        if "tile_join_unverified" in region.flags:
            store.add_review(
                ReviewItem(
                    id=new_id("rev"),
                    media_id=media_id,
                    target_kind="region",
                    target_ref=f"{occ.id}:{region.region_id}",
                    reason="vision:tile_join_unverified",
                    detail="タイルの重複行が一致しなかったため、結合位置が未検証です。",
                )
            )
    if FLAG_TRUNCATED in content.quality_flags:
        store.add_review(
            ReviewItem(
                id=new_id("rev"),
                media_id=media_id,
                target_kind="occurrence",
                target_ref=occ.id,
                reason="vision:truncated_output",
                detail="モデル出力が長さ制限で切れました。完成として扱っていません。",
            )
        )


def _update_screen_coverage(store: Store, media_id: str, duration_us: int) -> None:
    """画面側の時間軸を、抽出状態まで含めて更新する (設計 9.2)。"""
    existing = store.coverage(media_id, "screen")
    out_of_range = [s for s in existing if s.state == "out_of_range"]
    occurrences = store.occurrences(media_id)
    # 抽出を試みて失敗した状態と、まだ処理していない状態を区別する。
    # 中断した場合、未処理の期間を「失敗」と報告してしまうため。
    failed_ids = {
        r.target_ref
        for r in store.reviews(media_id)
        if r.reason in ("vision:extraction_failed", "vision:truncated_output")
    }
    spans: list[CoverageSpan] = list(out_of_range)
    for occ in occurrences:
        if occ.content_id:
            state, detail = "extracted", "本文抽出済み"
        elif occ.state_kind == STATE_BLANK:
            state, detail = "blank", "空画面"
        elif FLAG_NOT_EXTRACTED in occ.quality_flags and "short_or_transition" in occ.quality_flags:
            state, detail = "unextracted", "短時間表示・全文未確定"
        elif occ.id in failed_ids:
            state, detail = "failed", "抽出失敗"
        else:
            state, detail = "unextracted", "未処理（抽出をまだ行っていない）"
        spans.append(
            CoverageSpan(new_id("cov"), media_id, "screen", occ.start_us, occ.end_us, state, detail)
        )
    spans.sort(key=lambda s: s.start_us)
    filled: list[CoverageSpan] = []
    cursor = 0
    for span in spans:
        if span.start_us > cursor:
            filled.append(
                CoverageSpan(new_id("cov"), media_id, "screen", cursor, span.start_us, "unextracted", "表示状態なし")
            )
        filled.append(span)
        cursor = max(cursor, span.end_us)
    if cursor < duration_us:
        filled.append(
            CoverageSpan(new_id("cov"), media_id, "screen", cursor, duration_us, "unextracted", "表示状態なし")
        )
    store.replace_coverage(media_id, "screen", filled)
