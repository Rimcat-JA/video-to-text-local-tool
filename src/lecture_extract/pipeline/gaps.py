"""未抽出区間の切り分けと、再処理の優先順位づけ (設計 12.3 / 15 段階 4)。

未抽出には理由が複数ある。意図的に対象外にしたもの、処理が届かなかったもの、
試みて失敗したものを混ぜると、どこを直せばよいか分からなくなる。

全画面を無条件で再認識するのではなく、重要な区間を選んで再処理できるように、
「その区間に固有の内容がありそうか」を既存の記録だけで見積もる。
モデルは動かさない。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..db.store import Store
from ..models import ScreenContent, ScreenOccurrence
from ..util.timeutil import format_timestamp, overlap_us

log = logging.getLogger(__name__)

# 未抽出の理由
REASON_SKIPPED = "skipped_short"  # 短時間表示・遷移として意図的に対象外
REASON_BLANK = "blank"  # 空画面
REASON_FAILED = "failed"  # 抽出を試みて失敗した
REASON_NOT_ATTEMPTED = "not_attempted"  # まだ処理していない


@dataclass
class Gap:
    """本文が未抽出の連続区間。"""

    start_us: int
    end_us: int
    occurrence_ids: list[str]
    reason: str
    duration_us: int
    utterance_count: int
    speech_chars: int
    neighbours_differ: bool  # 前後の画面本文が違う＝間に固有の内容がありうる
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": format_timestamp(self.start_us),
            "end": format_timestamp(self.end_us),
            "start_us": self.start_us,
            "end_us": self.end_us,
            "reason": self.reason,
            "duration_us": self.duration_us,
            "occurrence_ids": self.occurrence_ids,
            "utterance_count": self.utterance_count,
            "speech_chars": self.speech_chars,
            "neighbours_differ": self.neighbours_differ,
            "score": round(self.score, 3),
        }


def _reason_for(occ: ScreenOccurrence, failed_ids: set[str]) -> str:
    if occ.id in failed_ids:
        return REASON_FAILED
    if occ.state_kind == "blank":
        return REASON_BLANK
    if "short_or_transition" in occ.quality_flags:
        return REASON_SKIPPED
    return REASON_NOT_ATTEMPTED


def find_gaps(store: Store, media_id: str, min_duration_us: int = 2_000_000) -> list[Gap]:
    """未抽出の連続区間を集め、再処理の優先度をつける。

    優先度は「この区間に固有の内容がありそうか」の見積もりで、
    抽出できなかった文字の量ではない。実際の内容は読むまで分からない。
    """
    occurrences = store.occurrences(media_id)
    contents: dict[str, ScreenContent] = {c.id: c for c in store.all_contents()}
    utterances = store.utterances(media_id)
    failed_ids = {
        r.target_ref
        for r in store.reviews(media_id)
        if r.reason == "vision:extraction_failed"
    }

    gaps: list[Gap] = []
    i = 0
    while i < len(occurrences):
        if occurrences[i].content_id:
            i += 1
            continue
        j = i
        while j < len(occurrences) and not occurrences[j].content_id:
            j += 1
        run = occurrences[i:j]

        # 前後の画面本文。同じなら、この区間は同じ画面のままだった可能性が高い。
        before = next(
            (o for o in reversed(occurrences[:i]) if o.content_id), None
        )
        after = next((o for o in occurrences[j:] if o.content_id), None)
        b_hash = contents[before.content_id].text_hash if before and before.content_id in contents else None
        a_hash = contents[after.content_id].text_hash if after and after.content_id in contents else None
        neighbours_differ = bool(b_hash and a_hash and b_hash != a_hash)

        start_us, end_us = run[0].start_us, run[-1].end_us
        duration = end_us - start_us
        overlapping = [
            u for u in utterances if overlap_us(u.start_us, u.end_us, start_us, end_us) > 0
        ]
        speech_chars = sum(len(u.text_raw) for u in overlapping)

        reasons = {_reason_for(o, failed_ids) for o in run}
        if REASON_FAILED in reasons:
            reason = REASON_FAILED
        elif REASON_NOT_ATTEMPTED in reasons:
            reason = REASON_NOT_ATTEMPTED
        elif reasons == {REASON_BLANK}:
            reason = REASON_BLANK
        else:
            reason = REASON_SKIPPED

        if duration < min_duration_us and reason == REASON_SKIPPED:
            i = j
            continue

        # 優先度: 長く表示され、説明が多く、前後で画面が変わっている区間ほど
        # 固有の内容を含む見込みが高い。
        score = 0.0
        score += min(duration / 10_000_000, 3.0)  # 表示の長さ (10 秒で 1.0、上限 3)
        score += min(speech_chars / 500, 3.0)  # 説明の量
        if neighbours_differ:
            score += 2.0
        if reason == REASON_FAILED:
            score += 3.0  # 失敗は必ず見直す
        elif reason == REASON_NOT_ATTEMPTED:
            score += 1.0
        if reason == REASON_BLANK:
            score = 0.0  # 空画面は回収するものがない

        gaps.append(
            Gap(
                start_us=start_us,
                end_us=end_us,
                occurrence_ids=[o.id for o in run],
                reason=reason,
                duration_us=duration,
                utterance_count=len(overlapping),
                speech_chars=speech_chars,
                neighbours_differ=neighbours_differ,
                score=score,
            )
        )
        i = j

    gaps.sort(key=lambda g: -g.score)
    return gaps


def summarize(gaps: list[Gap]) -> dict[str, Any]:
    from collections import Counter

    by_reason: Counter = Counter(g.reason for g in gaps)
    time_by_reason: Counter = Counter()
    for g in gaps:
        time_by_reason[g.reason] += g.duration_us
    return {
        "gaps": len(gaps),
        "by_reason": dict(by_reason),
        "time_by_reason_us": dict(time_by_reason),
        "with_speech": sum(1 for g in gaps if g.utterance_count > 0),
        "neighbours_differ": sum(1 for g in gaps if g.neighbours_differ),
    }
