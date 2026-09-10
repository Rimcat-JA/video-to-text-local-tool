"""読書ブロック（教材単位）の構築 (設計 5.4 / 8.2)。

保存単位 (表示状態) と読書単位 (教材のまとまり) を分ける。

同じスライド・同じコード領域・同じ文脈が続く範囲をひとつの教材単位にまとめ、
その内部に「いつ何がどう変わったか」を残す。画面が一瞬変わるたびに読み物が
分断されないようにする一方、内容が変わった事実は失わない。

同一画面の再表示 (repeat) と、内容が変わった表示 (modified) は区別する。
"""

from __future__ import annotations

import logging
from typing import Any

from ..config import RunConfig
from ..db.store import Store, new_id
from ..models import ReadingBlock, ScreenContent, ScreenOccurrence
from ..util.textnorm import line_similarity, normalize_for_compare

log = logging.getLogger(__name__)

CODE_KINDS = {"code", "terminal_output"}

# ブロック内での各表示状態の位置づけ
CHANGE_FIRST = "first"  # この教材単位で最初に現れた内容
CHANGE_REPEAT = "repeat"  # 同じ内容の再表示
CHANGE_MODIFIED = "modified"  # 内容が変わった
CHANGE_UNEXTRACTED = "unextracted"  # 本文未確定


def _material_similarity(a: str, b: str) -> float:
    """同じ教材の別の版とみなせるか。

    行の並びの近さ (difflib) を使うと、空行や短い行が多いスライド同士で値が
    跳ね上がり、無関係なスライドを同じ教材の変更として統合してしまう。
    中身のある行の重なり (Jaccard) で判断する。
    """
    la = {ln.strip() for ln in (a or "").split("\n") if ln.strip()}
    lb = {ln.strip() for ln in (b or "").split("\n") if ln.strip()}
    if not la or not lb:
        return 0.0
    return len(la & lb) / len(la | lb)


def _body_for(occ: ScreenOccurrence, contents: dict[str, ScreenContent]) -> str | None:
    if not occ.content_id:
        return None
    content = contents.get(occ.content_id)
    return content.body_text if content else None


def _heading_of(content: ScreenContent) -> str:
    """画面の見出し。文脈情報が無い場合、見出しが実質の題名になる。"""
    ctx = content.context or {}
    title = str(ctx.get("file_name") or ctx.get("title") or "").strip()
    if title:
        return normalize_for_compare(title)
    for region in sorted(content.regions, key=lambda r: r.reading_order):
        if region.kind == "heading" and region.text.strip():
            return normalize_for_compare(region.text.strip().splitlines()[0])
    return ""


def _context_key(occ: ScreenOccurrence, contents: dict[str, ScreenContent]) -> tuple[str, str]:
    """画面種別と題名。これが変われば別の教材とみなす。

    題名にはファイル名・題名だけでなく見出しも使う。文脈情報が空のスライドでは
    見出しが唯一の識別子であり、これを見ないと別のスライドを同じ教材の
    「変更後」として統合してしまう。
    """
    if not occ.content_id:
        return ("", "")
    content = contents.get(occ.content_id)
    if content is None:
        return ("", "")
    return (content.screen_kind, _heading_of(content))


def _title_for(occ: ScreenOccurrence, contents: dict[str, ScreenContent]) -> str:
    if not occ.content_id:
        return ""
    content = contents.get(occ.content_id)
    if content is None:
        return ""
    ctx = content.context or {}
    title = ctx.get("title") or ctx.get("file_name") or ""
    if title:
        return str(title)
    for region in sorted(content.regions, key=lambda r: r.reading_order):
        if region.kind == "heading" and region.text.strip():
            return region.text.strip().splitlines()[0][:120]
    return ""


class _Unit:
    """構築中の教材単位。"""

    def __init__(self) -> None:
        self.occurrences: list[ScreenOccurrence] = []
        self.deferred: list[ScreenOccurrence] = []  # 未抽出。後続が続けば取り込む
        self.text_hashes: set[str] = set()
        self.anchor_body: str | None = None
        self.context: tuple[str, str] | None = None
        self.changed = False  # 内容が変わった履歴があるか

    @property
    def last(self) -> ScreenOccurrence:
        return self.occurrences[-1]


def _can_continue(
    unit: _Unit,
    candidate: ScreenOccurrence,
    contents: dict[str, ScreenContent],
    cfg: RunConfig,
) -> tuple[bool, str]:
    """教材単位を続けてよいか。(続けるか, 位置づけ) を返す。"""
    block = cfg.block
    prev = unit.deferred[-1] if unit.deferred else unit.last
    if candidate.start_us - prev.end_us > block.unit_max_gap_us:
        return False, ""
    if candidate.end_us - unit.occurrences[0].start_us > block.unit_max_duration_us:
        return False, ""

    body = _body_for(candidate, contents)
    if body is None:
        # 未抽出は、後に同じ教材が続いたときだけ取り込む (設計 5.2 の統合保留)。
        if candidate.state_kind in ("short", "transition", "blank"):
            return True, CHANGE_UNEXTRACTED
        return False, ""

    content = contents.get(candidate.content_id or "")
    if content is not None and content.text_hash in unit.text_hashes:
        return True, CHANGE_REPEAT  # 同じ画面の再表示

    if _context_key(candidate, contents) != unit.context:
        return False, ""  # ファイル名や画面種別が変わった＝別の教材

    if unit.anchor_body is None or not unit.anchor_body.strip() or not body.strip():
        return False, ""
    if _material_similarity(unit.anchor_body, body) >= block.unit_similarity:
        return True, CHANGE_MODIFIED
    return False, ""


def build_blocks(
    occurrences: list[ScreenOccurrence], contents: dict[str, ScreenContent], cfg: RunConfig
) -> tuple[list[ReadingBlock], dict[str, str]]:
    """教材単位のブロックと、表示状態ごとの位置づけを返す。"""
    occurrences = sorted(occurrences, key=lambda o: (o.start_us, o.end_us))
    blocks: list[ReadingBlock] = []
    change_kinds: dict[str, str] = {}
    index = 0
    i = 0

    while i < len(occurrences):
        first = occurrences[i]

        # 本文が未抽出の状態が続く範囲は、ひとつの「未抽出区間」にまとめる。
        # 1 状態ずつブロックを立てると、読み物としても機械処理としても
        # 教材の区切りが失われる。期間は個々の表示状態に残っている。
        if first.content_id is None:
            j = i
            while j < len(occurrences) and occurrences[j].content_id is None:
                change_kinds[occurrences[j].id] = CHANGE_UNEXTRACTED
                j += 1
            members = occurrences[i:j]
            blocks.append(
                ReadingBlock(
                    id=new_id("blk"),
                    media_id=first.media_id,
                    index=index,
                    start_us=members[0].start_us,
                    end_us=members[-1].end_us,
                    occurrence_ids=[o.id for o in members],
                    kind="unextracted",
                    title_hint="",
                )
            )
            index += 1
            i = j
            continue

        unit = _Unit()
        unit.occurrences.append(first)
        change_kinds[first.id] = (
            CHANGE_FIRST if first.content_id else CHANGE_UNEXTRACTED
        )
        content = contents.get(first.content_id or "")
        if content is not None:
            unit.text_hashes.add(content.text_hash)
            unit.anchor_body = content.body_text
            unit.context = _context_key(first, contents)
        j = i + 1
        accepted_until = j

        if cfg.block.group_units:
            while j < len(occurrences):
                candidate = occurrences[j]
                ok, kind = _can_continue(unit, candidate, contents, cfg)
                if not ok:
                    break
                if kind == CHANGE_UNEXTRACTED:
                    # 確定させず保留。後続が同じ教材なら一緒に取り込む。
                    unit.deferred.append(candidate)
                    j += 1
                    continue
                unit.occurrences.extend(unit.deferred)
                for d in unit.deferred:
                    change_kinds[d.id] = CHANGE_UNEXTRACTED
                unit.deferred = []
                unit.occurrences.append(candidate)
                change_kinds[candidate.id] = kind
                if kind == CHANGE_MODIFIED:
                    unit.changed = True
                    unit.anchor_body = _body_for(candidate, contents)
                cand_content = contents.get(candidate.content_id or "")
                if cand_content is not None:
                    unit.text_hashes.add(cand_content.text_hash)
                    if unit.context is None:
                        unit.context = _context_key(candidate, contents)
                j += 1
                accepted_until = j

        members = sorted(unit.occurrences, key=lambda o: o.start_us)
        kind = "single"
        if len(members) > 1:
            kind = "unit_edited" if unit.changed else "unit"
        blocks.append(
            ReadingBlock(
                id=new_id("blk"),
                media_id=first.media_id,
                index=index,
                start_us=members[0].start_us,
                end_us=members[-1].end_us,
                occurrence_ids=[o.id for o in members],
                kind=kind,
                title_hint=_title_for(members[0], contents)
                or _title_for(members[-1], contents),
            )
        )
        index += 1
        i = accepted_until if accepted_until > i else i + 1

    return blocks, change_kinds


def run_block_builder(store: Store, cfg: RunConfig, media_id: str) -> dict[str, Any]:
    occurrences = store.occurrences(media_id)
    contents = {c.id: c for c in store.all_contents()}
    blocks, change_kinds = build_blocks(occurrences, contents, cfg)
    store.replace_blocks(media_id, blocks)

    # 各表示状態の位置づけ (初出 / 再表示 / 変更 / 未確定) を残す。
    for occ in occurrences:
        kind = change_kinds.get(occ.id)
        if not kind:
            continue
        flags = [f for f in occ.quality_flags if not f.startswith("change:")]
        flags.append(f"change:{kind}")
        occ.quality_flags = sorted(set(flags))
        store.upsert_occurrence(occ)

    from collections import Counter

    counts = Counter(change_kinds.values())
    stats = {
        "blocks": len(blocks),
        "unit_blocks": sum(1 for b in blocks if b.kind.startswith("unit")),
        "unextracted_blocks": sum(1 for b in blocks if b.kind == "unextracted"),
        "edited_units": sum(1 for b in blocks if b.kind == "unit_edited"),
        "occurrences": len(occurrences),
        "repeat_displays": counts.get(CHANGE_REPEAT, 0),
        "modified_displays": counts.get(CHANGE_MODIFIED, 0),
    }
    for name, value in stats.items():
        store.add_metric(media_id, "blocks", name, float(value))
    log.info("block_builder 完了: %s", stats)
    return stats
