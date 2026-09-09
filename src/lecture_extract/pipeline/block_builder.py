"""読書ブロックの構築 (設計 5.4 / 8.2)。

保存単位 (表示状態) と読書単位 (読書ブロック) を分ける。
連続入力やスクロールは複数状態を含む親ブロックにまとめるが、
内部の時刻付き変更履歴は失わない。
"""

from __future__ import annotations

import logging
from typing import Any

from ..config import RunConfig
from ..db.store import Store, new_id
from ..models import ReadingBlock, ScreenContent, ScreenOccurrence
from ..util.textnorm import line_similarity

log = logging.getLogger(__name__)

CODE_KINDS = {"code", "terminal_output"}


def _body_for(occ: ScreenOccurrence, contents: dict[str, ScreenContent]) -> str | None:
    if not occ.content_id:
        return None
    content = contents.get(occ.content_id)
    return content.body_text if content else None


def _title_for(occ: ScreenOccurrence, contents: dict[str, ScreenContent]) -> str:
    if not occ.content_id:
        return ""
    content = contents.get(occ.content_id)
    if content is None:
        return ""
    title = (content.context or {}).get("title") or (content.context or {}).get("file_name") or ""
    if title:
        return str(title)
    for region in sorted(content.regions, key=lambda r: r.reading_order):
        if region.kind == "heading" and region.text.strip():
            return region.text.strip().splitlines()[0][:120]
    return ""


def _can_continue(
    anchor: ScreenOccurrence,
    candidate: ScreenOccurrence,
    contents: dict[str, ScreenContent],
    cfg: RunConfig,
) -> bool:
    """連続入力・スクロールの一部として、同じ親ブロックへ入れてよいか。

    本文が抽出されている状態どうしの類似度でのみ判断する。
    未抽出の状態を「似ている」ことにしない (設計 5.2: 判定できなければ統合を保留する)。
    """
    if candidate.start_us - anchor.end_us > cfg.block.editing_max_gap_us:
        return False
    if (anchor.end_us - anchor.start_us) > cfg.block.editing_state_max_us:
        return False
    body_a = _body_for(anchor, contents)
    body_b = _body_for(candidate, contents)
    if body_a is None or body_b is None:
        return False
    if not body_a.strip() or not body_b.strip():
        return False
    return line_similarity(body_a, body_b) >= cfg.block.editing_similarity


def build_blocks(
    occurrences: list[ScreenOccurrence], contents: dict[str, ScreenContent], cfg: RunConfig
) -> list[ReadingBlock]:
    occurrences = sorted(occurrences, key=lambda o: (o.start_us, o.end_us))
    blocks: list[ReadingBlock] = []
    i = 0
    index = 0
    while i < len(occurrences):
        group = [occurrences[i]]
        j = i + 1
        if cfg.block.group_editing:
            anchor = occurrences[i]
            deferred: list[ScreenOccurrence] = []
            k = j
            while k < len(occurrences):
                candidate = occurrences[k]
                if candidate.content_id is None:
                    # 未抽出の状態は、直後に類似本文が続いたときだけ取り込む。
                    prev = deferred[-1] if deferred else anchor
                    if candidate.state_kind not in ("short", "transition"):
                        break
                    if candidate.start_us - prev.end_us > cfg.block.editing_max_gap_us:
                        break
                    deferred.append(candidate)
                    k += 1
                    continue
                if not _can_continue(anchor, candidate, contents, cfg):
                    break
                group.extend(deferred)
                deferred = []
                group.append(candidate)
                anchor = candidate
                k += 1
                j = k  # ここまでが確定した範囲。取り込まなかった状態は次のブロックへ回す。

        if len(group) < cfg.block.editing_min_states:
            # まとめる根拠が弱いので 1 状態 = 1 ブロックにする。
            occ = occurrences[i]
            blocks.append(
                ReadingBlock(
                    id=new_id("blk"),
                    media_id=occ.media_id,
                    index=index,
                    start_us=occ.start_us,
                    end_us=occ.end_us,
                    occurrence_ids=[occ.id],
                    kind="single",
                    title_hint=_title_for(occ, contents),
                )
            )
            index += 1
            i += 1
            continue

        kind = "editing"
        first_content = contents.get(group[0].content_id) if group[0].content_id else None
        if first_content and any(r.kind in CODE_KINDS for r in first_content.regions):
            kind = "editing"
        blocks.append(
            ReadingBlock(
                id=new_id("blk"),
                media_id=group[0].media_id,
                index=index,
                start_us=group[0].start_us,
                end_us=group[-1].end_us,
                occurrence_ids=[o.id for o in group],
                kind=kind,
                title_hint=_title_for(group[-1], contents) or _title_for(group[0], contents),
            )
        )
        index += 1
        i = j
    return blocks


def run_block_builder(store: Store, cfg: RunConfig, media_id: str) -> dict[str, Any]:
    occurrences = store.occurrences(media_id)
    contents = {c.id: c for c in store.all_contents()}
    blocks = build_blocks(occurrences, contents, cfg)
    store.replace_blocks(media_id, blocks)
    stats = {
        "blocks": len(blocks),
        "editing_blocks": sum(1 for b in blocks if b.kind != "single"),
        "occurrences": len(occurrences),
    }
    for name, value in stats.items():
        store.add_metric(media_id, "blocks", name, float(value))
    log.info("block_builder 完了: %s", stats)
    return stats
