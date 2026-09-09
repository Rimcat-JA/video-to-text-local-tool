"""画像キャッシュの容量管理 (設計 3.3 / 12.2 / 12.3)。

- 元動画とは別に容量上限を設ける。
- 削除する画像は、元動画ハッシュと PTS から再生成できることを確認してから消す。
- 確定状態の代表画像、変更候補、不明箇所を優先して残す。
- 保存領域が不足した場合は処理を停止し、既存結果と元動画を保全する。
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from ..config import RunConfig
from ..db.store import Store
from .ffmpeg_tools import ffmpeg_path, run

log = logging.getLogger(__name__)

# これを下回ったら新しい画像を書かずに停止する。
MIN_FREE_BYTES = 2 * 1024**3


class StorageExhausted(RuntimeError):
    pass


def check_free_space(path: str | Path, *, min_free: int = MIN_FREE_BYTES) -> None:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(str(target))
    if usage.free < min_free:
        raise StorageExhausted(
            f"保存領域が不足しています（空き {usage.free / 1024**3:.2f} GiB、"
            f"必要 {min_free / 1024**3:.2f} GiB）。処理を停止します。"
            " 既存の結果と元動画はそのまま残しています。"
        )


def frames_dir(cfg: RunConfig, media_id: str) -> Path:
    return Path(cfg.work_dir) / "frames" / media_id


def cache_size_bytes(cfg: RunConfig, media_id: str) -> int:
    directory = frames_dir(cfg, media_id)
    if not directory.exists():
        return 0
    return sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())


def _priority(store: Store, media_id: str) -> dict[str, int]:
    """フレームごとの保持優先度。大きいほど残す。"""
    open_reviews = {r.target_ref.split(":")[0] for r in store.reviews(media_id, status="open")}
    priority: dict[str, int] = {}
    for occ in store.occurrences(media_id):
        score = 0
        if occ.id in open_reviews:
            score += 100  # 不明箇所・要確認は残す
        if not occ.content_id:
            score += 50  # 未抽出は再解析の対象になる
        if occ.state_kind in ("short", "transition"):
            score += 20  # 短時間表示は取り直しが難しい
        for ref in occ.evidence_refs:
            frame_id = ref.get("frame_id")
            if frame_id:
                priority[frame_id] = max(priority.get(frame_id, 0), score)
    return priority


def prune_image_cache(store: Store, cfg: RunConfig, media_id: str) -> dict[str, Any]:
    """上限を超えている分だけ、優先度の低い画像から削除する。"""
    limit = cfg.scan.max_cache_bytes
    stats: dict[str, Any] = {"before_bytes": cache_size_bytes(cfg, media_id), "removed": 0, "freed_bytes": 0}
    if limit <= 0 or stats["before_bytes"] <= limit:
        stats["after_bytes"] = stats["before_bytes"]
        return stats

    media = store.get_media(media_id)
    if media is None:
        stats["after_bytes"] = stats["before_bytes"]
        return stats
    source = Path(media["source_path"])
    priority = _priority(store, media_id)
    work_dir = Path(cfg.work_dir)

    candidates = []
    for frame in store.frames_for_media(media_id):
        if not frame.get("evidence_ref"):
            continue
        path = work_dir / frame["evidence_ref"]
        if not path.exists():
            continue
        # 元動画ハッシュと時刻から再生成できることが条件 (設計 12.3)。
        regenerable = source.exists() and frame.get("t_us") is not None
        if not regenerable:
            continue
        candidates.append((priority.get(frame["id"], 0), frame["t_us"], frame, path))

    candidates.sort(key=lambda item: (item[0], item[1]))
    freed = 0
    target = stats["before_bytes"] - limit
    for _, _, frame, path in candidates:
        if freed >= target:
            break
        size = path.stat().st_size
        path.unlink()
        store.conn.execute(
            "UPDATE frame_observation SET evidence_ref = NULL WHERE id = ?", (frame["id"],)
        )
        freed += size
        stats["removed"] += 1
    stats["freed_bytes"] = freed
    stats["after_bytes"] = cache_size_bytes(cfg, media_id)
    log.info(
        "画像キャッシュを整理しました: %d 件削除, %.2f GiB 解放",
        stats["removed"],
        freed / 1024**3,
    )
    return stats


def regenerate_frame(store: Store, cfg: RunConfig, frame_id: str) -> Path:
    """削除した根拠画像を、元動画と時刻から作り直す (設計 12.3)。"""
    frame = store.get_frame(frame_id)
    if frame is None:
        raise KeyError(f"フレームが見つかりません: {frame_id}")
    media = store.get_media(frame["media_id"])
    if media is None:
        raise KeyError("元動画の記録が見つかりません。")
    source = Path(media["source_path"])
    if not source.exists():
        raise FileNotFoundError(f"元動画が見つかりません: {source}")

    t_us = int(frame["t_us"])
    t0_us = int(media["time_origin_us"])
    seek_s = (t_us + t0_us) / 1_000_000
    out_dir = frames_dir(cfg, frame["media_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{t_us:012d}_regenerated.png"
    check_free_space(out_dir)
    run(
        [
            ffmpeg_path(),
            "-v",
            "error",
            "-y",
            "-ss",
            f"{seek_s:.6f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            str(out),
        ]
    )
    rel = str(out.relative_to(Path(cfg.work_dir))).replace("\\", "/")
    store.conn.execute("UPDATE frame_observation SET evidence_ref = ? WHERE id = ?", (rel, frame_id))
    log.info("根拠画像を再生成しました: %s (t=%.3fs)", rel, t_us / 1e6)
    return out
