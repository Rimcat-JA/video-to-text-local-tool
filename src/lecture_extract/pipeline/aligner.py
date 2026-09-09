"""表示期間と発話の同期 (設計 8)。

overlap(B, U) = max(0, min(B.end, U.end) - max(B.start, U.start))

対応づけは「同時に表示・発話された」という事実だけを表す。
発話が前の画面を説明していても、時刻を変更して合わせない。
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

from ..db.store import Store
from ..models import Alignment, ScreenOccurrence, Utterance
from ..util.timeutil import overlap_us

log = logging.getLogger(__name__)


def align_occurrences(
    occurrences: Sequence[ScreenOccurrence], utterances: Sequence[Utterance]
) -> list[Alignment]:
    """時刻順に走査し、重複時間が正の組み合わせだけを関連づける。"""
    occs = sorted(occurrences, key=lambda o: (o.start_us, o.end_us))
    utts = sorted(utterances, key=lambda u: (u.start_us, u.end_us))
    alignments: list[Alignment] = []
    active: list[Utterance] = []
    i = 0
    for occ in occs:
        while i < len(utts) and utts[i].start_us < occ.end_us:
            active.append(utts[i])
            i += 1
        active = [u for u in active if u.end_us > occ.start_us]
        for utt in active:
            ov = overlap_us(occ.start_us, occ.end_us, utt.start_us, utt.end_us)
            if ov > 0:
                alignments.append(Alignment(occurrence_id=occ.id, utterance_id=utt.id, overlap_us=ov))
    return _mark_primary(alignments, occs)


def _mark_primary(alignments: list[Alignment], occs: Sequence[ScreenOccurrence]) -> list[Alignment]:
    """全文を表示する主ブロックを決める (設計 8.2)。

    重複時間が最大のブロック。同率なら先に始まるものを選ぶ。
    """
    order = {occ.id: (occ.start_us, occ.end_us) for occ in occs}
    by_utt: dict[str, list[Alignment]] = {}
    for a in alignments:
        by_utt.setdefault(a.utterance_id, []).append(a)
    for utt_id, group in by_utt.items():
        best = min(group, key=lambda a: (-a.overlap_us, order.get(a.occurrence_id, (0, 0))))
        for a in group:
            a.is_primary = a is best
    return alignments


def run_aligner(store: Store, media_id: str) -> dict[str, int]:
    occurrences = store.occurrences(media_id)
    utterances = store.utterances(media_id)
    alignments = align_occurrences(occurrences, utterances)
    store.replace_alignments(media_id, alignments)

    aligned_utts = {a.utterance_id for a in alignments}
    orphan_utterances = [u for u in utterances if u.id not in aligned_utts]
    aligned_occs = {a.occurrence_id for a in alignments}
    silent_occurrences = [o for o in occurrences if o.id not in aligned_occs]

    stats = {
        "occurrences": len(occurrences),
        "utterances": len(utterances),
        "alignments": len(alignments),
        "utterances_without_screen": len(orphan_utterances),
        "occurrences_without_speech": len(silent_occurrences),
        "primary_assignments": sum(1 for a in alignments if a.is_primary),
    }
    for name, value in stats.items():
        store.add_metric(media_id, "align", name, float(value))
    log.info("align 完了: %s", stats)
    return stats


def utterances_for_occurrence(
    store: Store, media_id: str
) -> tuple[dict[str, list[Alignment]], dict[str, Utterance]]:
    utt_map = {u.id: u for u in store.utterances(media_id)}
    by_occ: dict[str, list[Alignment]] = {}
    for a in store.alignments(media_id):
        by_occ.setdefault(a.occurrence_id, []).append(a)
    for group in by_occ.values():
        group.sort(key=lambda a: (utt_map[a.utterance_id].start_us, utt_map[a.utterance_id].end_us))
    return by_occ, utt_map


def spans_without_screen(
    occurrences: Iterable[ScreenOccurrence], duration_us: int
) -> list[tuple[int, int]]:
    """画面の表示期間が無い区間 (設計 8.2: 画面文字がない発話も独立して残す)。"""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for occ in sorted(occurrences, key=lambda o: o.start_us):
        if occ.start_us > cursor:
            spans.append((cursor, occ.start_us))
        cursor = max(cursor, occ.end_us)
    if cursor < duration_us:
        spans.append((cursor, duration_us))
    return spans
