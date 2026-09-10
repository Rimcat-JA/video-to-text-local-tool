"""表示状態の統合 (設計 5.2 第 2 段階 / 5.5 / 5.6)。

変化候補として切り出した期間のうち、文字内容が実際には変わっていないものを
ひとつの表示期間に戻す。判定できない場合 (未抽出の状態を挟む場合など) は
統合を保留し、期間を分けたまま残す。

文字が同じでも変化が検出されていた場合は、その事実を視覚イベントとして残す。
"""

from __future__ import annotations

import logging
from typing import Any

from ..config import RunConfig
from ..db.store import Store, new_id
from ..models import FLAG_NOT_EXTRACTED, VisualEvent

log = logging.getLogger(__name__)


def run_screen_tracker(store: Store, cfg: RunConfig, media_id: str) -> dict[str, Any]:
    occurrences = store.occurrences(media_id)
    stats = {
        "before": len(occurrences),
        "merged": 0,
        "nontext_changes": 0,
        "merge_deferred": 0,
        "after": 0,
    }
    if not occurrences:
        return stats

    contents = {c.id: c for c in store.all_contents()}
    merged: list[Any] = []
    current = occurrences[0]

    for nxt in occurrences[1:]:
        contiguous = nxt.start_us <= current.end_us
        both_extracted = bool(current.content_id) and bool(nxt.content_id)
        same_text = (
            both_extracted
            and contents.get(current.content_id) is not None
            and contents.get(nxt.content_id) is not None
            and contents[current.content_id].text_hash == contents[nxt.content_id].text_hash
        )
        if same_text and contiguous:
            # 文字が同じなので同じ内容の継続。ただし画像上の変化はイベントとして残す。
            store.add_visual_event(
                VisualEvent(
                    id=new_id("vev"),
                    media_id=media_id,
                    start_us=nxt.boundary_start_lo_us,
                    end_us=nxt.boundary_start_hi_us,
                    region_ref="",
                    event_kind="nontext_change",
                    annotation=f"画像に変化があったが文字内容は同一 ({nxt.change_summary})",
                )
            )
            stats["nontext_changes"] += 1
            current.end_us = nxt.end_us
            current.boundary_end_lo_us = nxt.boundary_end_lo_us
            current.boundary_end_hi_us = nxt.boundary_end_hi_us
            current.evidence_refs = current.evidence_refs + nxt.evidence_refs
            current.quality_flags = sorted(set(current.quality_flags) | set(nxt.quality_flags) - {FLAG_NOT_EXTRACTED})
            current.change_summary = f"{current.change_summary}; merged({nxt.change_summary})"
            # 統合で消える表示期間を参照している対応づけを先に消す。
            # 初回は align より前に走るので問題にならないが、再実行時は
            # 既存の alignment が外部キーで参照している。
            store.conn.execute("DELETE FROM alignment WHERE occurrence_id = ?", (nxt.id,))
            store.conn.execute("DELETE FROM screen_occurrence WHERE id = ?", (nxt.id,))
            stats["merged"] += 1
            continue

        if contiguous and both_extracted is False and current.content_id and not nxt.content_id:
            # 未抽出の状態を挟むため統合を保留する (設計 5.2)。
            stats["merge_deferred"] += 1

        merged.append(current)
        current = nxt

    merged.append(current)
    for occ in merged:
        store.upsert_occurrence(occ)
    stats["after"] = len(merged)
    store.add_metric(media_id, "screen_tracker", "occurrences_after_merge", float(len(merged)))
    store.add_metric(media_id, "screen_tracker", "merged", float(stats["merged"]))
    log.info("screen_tracker: %s", stats)
    return stats
