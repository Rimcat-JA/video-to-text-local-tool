"""発話の不確かさを記録する (設計 7.2 / 14.2)。

音声認識の結果そのものからは精度を判断できない。画面に表示されていた語と
突き合わせ、専門語が「画面で確認できたか」を区別して残す。

原文は書き換えない。判定はフラグとしてだけ付ける。画面に語があるという理由で
話していない語を挿入することもしない (設計 7.2)。
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

from ..db.store import Store, new_id
from ..models import ReviewItem, ScreenContent
from ..util.timeutil import overlap_us

log = logging.getLogger(__name__)

# 専門語らしい形。日本語話者の講義でも、技術用語は英数字で現れることが多い。
_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]{3,}")

# 一般語。用語として扱わない。
_COMMON = {
    "about", "after", "again", "also", "another", "because", "before", "being",
    "between", "both", "could", "different", "does", "doing", "down", "each",
    "even", "every", "first", "from", "going", "have", "here", "into", "just",
    "know", "like", "little", "look", "make", "many", "more", "most", "much",
    "need", "now", "only", "other", "over", "part", "really", "right", "same",
    "say", "see", "should", "some", "start", "still", "such", "sure", "take",
    "than", "that", "their", "them", "then", "there", "these", "they", "thing",
    "think", "this", "those", "through", "time", "under", "very", "want",
    "well", "what", "when", "where", "which", "while", "will", "with", "work",
    "would", "your", "actually", "basically", "something", "everything",
}

# 発話の周辺で画面に出ていれば「確認済み」とみなす時間幅
NEARBY_US = 120_000_000  # 前後 2 分

# 何を根拠に確認したかが分かる名前にする。これらは「画面との用語照合結果」であり、
# 音声認識が正しいかどうかの判定ではない。画面に無い語を話すことも、
# 誤認識した語がたまたま画面に存在することもある (設計 7.2)。
FLAG_CONFIRMED = "term_match:screen_nearby"  # 同時期の画面に同じ語があった
FLAG_ELSEWHERE = "term_match:screen_elsewhere"  # 講義の別の箇所の画面にあった
FLAG_UNVERIFIED = "term_match:not_on_screen"  # 画面には見当たらない
FLAG_NO_TERMS = "term_match:no_terms"  # 照合対象の専門語が無い

# 音声そのものとの照合は行っていない。行った場合に使う値を予約しておく。
FLAG_AUDIO_UNCHECKED = "audio_verify:unchecked"


# 技術語らしい形。記号や大文字混じり、数字を含む語は一般語ではない。
_TECHNICAL_SHAPE = re.compile(r"[_.\-0-9]|[a-z][A-Z]")


def _terms(text: str, vocabulary: set[str] | None = None) -> set[str]:
    """発話・画面から専門語らしい語を拾う。

    普通の英単語を専門語として数えると、未確認が大量に出て判断材料にならない。
    「記号や数字を含む」「大文字小文字が混在する」語か、画面に出ている語だけを
    専門語として扱う。
    """
    out = set()
    for m in _TERM_RE.findall(text or ""):
        low = m.lower()
        if low in _COMMON or len(low) < 4:
            continue
        technical = bool(_TECHNICAL_SHAPE.search(m))
        if technical or (vocabulary is not None and low in vocabulary):
            out.add(low)
    return out


def _plain_words(text: str) -> set[str]:
    """画面に出ている語（一般語も含む）。発話側の照合対象を決めるために使う。"""
    out = set()
    for m in _TERM_RE.findall(text or ""):
        low = m.lower()
        if len(low) >= 4 and low not in _COMMON:
            out.add(low)
    return out


def run_speech_check(store: Store, media_id: str) -> dict[str, Any]:
    """発話中の専門語を、同時期に画面へ出ていた語と突き合わせる。"""
    contents: dict[str, ScreenContent] = {c.id: c for c in store.all_contents()}
    occurrences = store.occurrences(media_id)

    # 画面に出た語を、その表示期間とともに集める。
    screen_terms: list[tuple[int, int, set[str]]] = []
    all_screen_terms: set[str] = set()
    for occ in occurrences:
        content = contents.get(occ.content_id or "")
        if content is None:
            continue
        terms = _terms(content.body_text, vocabulary=None) | _plain_words(content.body_text)
        if not terms:
            continue
        screen_terms.append((occ.start_us, occ.end_us, terms))
        all_screen_terms |= terms

    stats = {
        "utterances": 0,
        "with_terms": 0,
        "confirmed": 0,
        "unverified": 0,
        "screen_vocabulary": len(all_screen_terms),
    }
    unverified_counter: Counter = Counter()
    store.clear_reviews_by_reason_prefix(media_id, "asr:terms")

    for utt in store.utterances(media_id):
        stats["utterances"] += 1
        # 発話側は「技術語らしい語」か「画面に出ている語」だけを対象にする。
        terms = _terms(utt.text_raw, vocabulary=all_screen_terms)
        flags = [
            f
            for f in utt.quality_flags
            if not f.startswith("term_match:")
            and not f.startswith("terms_")
            and not f.startswith("audio_verify:")
        ]
        # 元音声との照合は未実施。未検査であることを明示する。
        flags.append(FLAG_AUDIO_UNCHECKED)
        if not terms:
            utt.quality_flags = sorted(set(flags) | {FLAG_NO_TERMS})
            store.upsert_utterance(utt)
            continue

        stats["with_terms"] += 1
        nearby: set[str] = set()
        for s, e, t in screen_terms:
            if overlap_us(utt.start_us - NEARBY_US, utt.end_us + NEARBY_US, s, e) > 0:
                nearby |= t

        confirmed = {t for t in terms if t in nearby}
        # 近くの画面に無くても、講義のどこかで表示されていれば「別の箇所で確認」。
        elsewhere = {t for t in terms - confirmed if t in all_screen_terms}
        unknown = terms - confirmed - elsewhere

        if confirmed:
            flags.append(FLAG_CONFIRMED)
            stats["confirmed"] += 1
        if unknown:
            flags.append(FLAG_UNVERIFIED)
            stats["unverified"] += 1
            unverified_counter.update(unknown)
        if elsewhere:
            flags.append(FLAG_ELSEWHERE)
        utt.quality_flags = sorted(set(flags))
        # 判定の根拠を残す。原文は変更しない。
        utt.tokens = [t for t in utt.tokens if t.get("kind") != "term_check"] + [
            {
                "kind": "term_check",
                "confirmed": sorted(confirmed),
                "seen_elsewhere": sorted(elsewhere),
                "not_found": sorted(unknown),
            }
        ]
        store.upsert_utterance(utt)

    # 画面に一度も出ない語が繰り返し現れる場合は、誤認の可能性が高い。
    for term, n in unverified_counter.most_common(30):
        if n < 5:
            break
        store.add_review(
            ReviewItem(
                id=new_id("rev"),
                media_id=media_id,
                target_kind="term",
                target_ref=term,
                reason="asr:terms_not_on_screen",
                detail=(
                    f"「{term}」が発話に {n} 回現れますが、画面には一度も表示されていません。"
                    "専門語の誤認の可能性があります。"
                ),
            )
        )

    for name, value in stats.items():
        store.add_metric(media_id, "speech_check", name, float(value))
    log.info("speech_check 完了: %s", stats)
    return stats
