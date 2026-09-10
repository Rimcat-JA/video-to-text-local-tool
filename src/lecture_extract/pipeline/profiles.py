"""用途別の出力 (保存用 / AI 参照用 / 学習ノートの骨組み)。

設計 1.2 の「抽出と解釈の分離」を保つため、原本 (lecture.md と JSONL) は
一切変更しない。ここで作るのは、そこから機械的に導いた派生物であり、
すべての節が元の ID と時刻を持ち、原資料へ戻れるようにする。

- material.md      : AI に読ませる教材。重複を省き、教材単位ごとにまとめる
- study_outline.md : 日本語の学習ノートの骨組み。章立てと用語と出典
                     （解説文の生成は別工程。ここでは書かない）
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ..config import RunConfig
from ..db.store import Store
from ..models import (
    ROLE_MATERIAL_BODY,
    ROLE_MATERIAL_CONTEXT,
    ReadingBlock,
    ScreenContent,
    ScreenOccurrence,
    Utterance,
)
from ..util.atomic import atomic_write_text
from ..util.timeutil import format_timestamp
from .exporter import escape_markdown_text, fence_for

log = logging.getLogger(__name__)

CODE_KINDS = {"code", "terminal_output"}

# 用語らしさの判定に使う語形。画面に出ている語と発話を突き合わせるために使う。
_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]{2,}")
# 一般的な会話語。用語一覧に載せない。専門語との区別がつかなくなるため。
_STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "you", "your", "from", "are",
    "was", "were", "has", "have", "not", "but", "can", "will", "how", "what",
    "when", "where", "which", "into", "out", "get", "set", "use", "using",
    "all", "any", "more", "than", "then", "one", "two", "new", "our",
    "about", "after", "again", "also", "another", "back", "because", "been",
    "before", "being", "both", "come", "could", "did", "does", "doing", "done",
    "down", "each", "else", "even", "every", "far", "few", "find", "first",
    "give", "going", "good", "great", "had", "here", "his", "her", "hey",
    "important", "its", "just", "keep", "kind", "know", "last", "least",
    "left", "less", "let", "like", "little", "long", "look", "lot", "make",
    "many", "may", "mean", "might", "most", "much", "must", "need", "never",
    "next", "nice", "now", "off", "okay", "once", "only", "other", "over",
    "own", "part", "put", "quite", "rather", "really", "right", "said", "same",
    "say", "see", "seen", "should", "show", "side", "simple", "since", "some",
    "sort", "start", "still", "stuff", "such", "sure", "take", "tell", "them",
    "there", "these", "they", "thing", "things", "think", "those", "though",
    "three", "through", "time", "times", "too", "took", "under", "until",
    "very", "want", "watch", "way", "well", "went", "were", "whole", "why",
    "without", "work", "would", "yes", "yet", "actually", "basically",
    "something", "everything", "anything", "nothing", "someone", "everyone",
    "here", "there", "again", "always", "already", "almost", "enough",
}

# 専門語らしい形。記号・数字・大文字小文字の混在を含む語は一般語ではない。
_TECHNICAL_SHAPE = re.compile(r"[_.\-0-9]|[a-z][A-Z]")


class ProfileExporter:
    def __init__(self, store: Store, cfg: RunConfig, media: dict[str, Any]):
        self.store = store
        self.cfg = cfg
        self.media = media
        self.media_id = media["id"]
        self.out_dir = Path(cfg.out_dir)
        self.occurrences = store.occurrences(self.media_id)
        referenced = {o.content_id for o in self.occurrences if o.content_id}
        self.contents: dict[str, ScreenContent] = {
            c.id: c for c in store.all_contents() if c.id in referenced
        }
        self.occ_by_id = {o.id: o for o in self.occurrences}
        self.blocks = store.blocks(self.media_id)
        self.utt_by_id = {u.id: u for u in store.utterances(self.media_id)}
        self.by_occ: dict[str, list] = {}
        for a in store.alignments(self.media_id):
            self.by_occ.setdefault(a.occurrence_id, []).append(a)

    # ------------------------------------------------------------- helpers
    def _members(self, block: ReadingBlock) -> list[ScreenOccurrence]:
        return [self.occ_by_id[o] for o in block.occurrence_ids if o in self.occ_by_id]

    def _change_kind(self, occ: ScreenOccurrence) -> str:
        for f in occ.quality_flags:
            if f.startswith("change:"):
                return f.split(":", 1)[1]
        return "unextracted" if not occ.content_id else "first"

    def _utterances(self, occs: list[ScreenOccurrence]) -> list[Utterance]:
        seen: set[str] = set()
        out: list[Utterance] = []
        for occ in occs:
            for a in self.by_occ.get(occ.id, []):
                if a.utterance_id in seen:
                    continue
                seen.add(a.utterance_id)
                utt = self.utt_by_id.get(a.utterance_id)
                if utt is not None:
                    out.append(utt)
        return sorted(out, key=lambda u: u.start_us)

    def _screen_versions(self, occs: list[ScreenOccurrence]) -> list[tuple[ScreenOccurrence, ScreenContent]]:
        """この単位で実際に内容が変わった時点だけを返す（再表示は省く）。"""
        out: list[tuple[ScreenOccurrence, ScreenContent]] = []
        last_hash: str | None = None
        for occ in occs:
            content = self.contents.get(occ.content_id or "")
            if content is None:
                continue
            if content.text_hash == last_hash:
                continue
            out.append((occ, content))
            last_hash = content.text_hash
        return out

    # ---------------------------------------------------------- material.md
    def write_material(self) -> Path:
        """AI に読ませる教材。重複を省き、教材単位ごとにまとめる。

        発話は結合するだけで書き換えない。画面本文も原文のまま。
        各節に元の ID と時刻を残し、原資料へ戻れるようにする。
        """
        path = self.out_dir / "material.md"
        lines: list[str] = [
            f"# 教材: {Path(self.media['source_path']).name}",
            "",
            "画面に表示された原文と発話を、教材単位ごとにまとめたものです。",
            "要約・翻訳・書き換えは行っていません。発話は連結しただけです。",
            "画面本文が未抽出の区間も、音声の説明は収録しています。",
            "画面本文に未解決の品質問題がある場合は、その単元の冒頭に注記を付けています。",
            "",
            f"- 元動画: `{self.media['source_path']}`",
            f"- 再生時間: {format_timestamp(self.media['duration_us'])}",
            "",
            "---",
            "",
        ]
        written = 0
        audio_only = 0
        covered: set[str] = set()
        index = 0
        while index < len(self.blocks):
            block = self.blocks[index]
            occs = self._members(block)
            versions = self._screen_versions(occs)

            # 画面本文が無い区間が続く場合、まとめて「音声説明のみ」の単元にする。
            # 画面を抽出できなかったことを、説明の欠落にしない (P0-1)。
            if not versions:
                run = [block]
                j = index + 1
                while j < len(self.blocks):
                    nxt = self._members(self.blocks[j])
                    if self._screen_versions(nxt):
                        break
                    run.append(self.blocks[j])
                    j += 1
                run_occs = [o for b in run for o in self._members(b)]
                run_utts = self._utterances(run_occs)
                index = j
                if not run_utts:
                    continue
                audio_only += 1
                lines.append(
                    f"## {format_timestamp(run[0].start_us)} – {format_timestamp(run[-1].end_us)}"
                    " （画面本文なし・音声説明あり）"
                )
                lines.append("")
                lines.append(
                    "> この区間の画面本文は抽出していません。表示期間と境界は"
                    " `screen_occurrences.jsonl` に残っています。"
                )
                lines.append("")
                lines.append("**説明**")
                lines.append("")
                lines.append(
                    escape_markdown_text(" ".join(u.text_raw.strip() for u in run_utts))
                )
                lines.append("")
                covered.update(u.id for u in run_utts)
                lines.append(
                    f"<!-- blocks={','.join(b.id for b in run)} screens=none "
                    f"utterances={','.join(u.id for u in run_utts)} "
                    f"t={run[0].start_us}-{run[-1].end_us} -->"
                )
                lines.append("")
                continue

            index += 1
            utts = self._utterances(occs)
            written += 1
            title = block.title_hint.strip()
            heading = f"## {format_timestamp(block.start_us)} – {format_timestamp(block.end_us)}"
            if title:
                heading += f" {escape_markdown_text(title)}"
            lines.append(heading)
            lines.append("")

            # 未解決の品質問題を、この単元の冒頭に示す (P0-2)。
            notes = self._quality_notes(versions)
            if notes:
                lines.append("> **この単元の画面本文には未解決の問題があります**")
                for note in notes:
                    lines.append(f"> - {note}")
                lines.append("")

            first_occ, first_content = versions[0]
            lines.extend(self._content_lines(first_content))
            for occ, content in versions[1:]:
                lines.append(
                    f"**{format_timestamp(occ.start_us)} 時点の変更後**"
                )
                lines.append("")
                lines.extend(self._content_lines(content))

            if utts:
                lines.append("**説明**")
                lines.append("")
                lines.append(escape_markdown_text(" ".join(u.text_raw.strip() for u in utts)))
                lines.append("")
                covered.update(u.id for u in utts)

            refs = ", ".join(v[1].id for v in versions[:4])
            lines.append(
                f"<!-- block={block.id} screens={refs} "
                f"t={block.start_us}-{block.end_us} -->"
            )
            lines.append("")

        # 収録できなかった発話が残っていないか確かめ、残っていれば末尾に収める。
        missing = [u for u in sorted(self.utt_by_id.values(), key=lambda u: u.start_us)
                   if u.id not in covered]
        if missing:
            lines.append("## どの単元にも入らなかった発話")
            lines.append("")
            lines.append(
                "画面の表示期間と重ならない発話です。欠落させないためここに収めます。"
            )
            lines.append("")
            for u in missing:
                lines.append(
                    f"- `{format_timestamp(u.start_us)} – {format_timestamp(u.end_us)}` "
                    f"{escape_markdown_text(u.text_raw)} <!-- {u.id} -->"
                )
            lines.append("")
        log.info(
            "material.md: 画面つき %d 単元 / 音声のみ %d 単元 / 収録発話 %d 件 (全 %d 件)",
            written, audio_only, len(covered) + len(missing), len(self.utt_by_id),
        )
        atomic_write_text(path, "\n".join(lines) + "\n")
        return path

    _FLAG_NOTES = {
        "truncated": "モデルの出力が長さ制限で切れています。本文が途中までの可能性があります。",
        "tile_join_unverified": "分割して読んだ範囲の継ぎ目が照合できていません。重複や欠落の可能性があります。",
        "fallback_split_read": "全画面の読み取りが成立せず、分割して読み直した結果です。",
        "unreadable": "判読できない文字が含まれます。推測では埋めていません。",
        "possible_duplicate_regions": "よく似た領域が複数あります。同じ範囲を二重に読んだ可能性があります。",
        "duplicate_regions_removed": "完全に同じ内容の領域を取り除いています。",
        "low_resolution": "解像度が不足しており、文字の形が不確かです。",
        "not_extracted": "一部の領域を抽出できていません。",
    }
    # 取り除き済みで、読み手の判断に影響しないもの
    _FLAG_RESOLVED = {"duplicate_regions_removed"}

    def _quality_notes(self, versions: list[tuple[ScreenOccurrence, ScreenContent]]) -> list[str]:
        """この単元が参照する画面本文の、未解決の品質問題を並べる (P0-2)。"""
        seen: dict[str, set[str]] = {}
        for occ, content in versions:
            flags = set(content.quality_flags)
            for region in content.regions:
                flags |= {f for f in region.flags if f in self._FLAG_NOTES}
            for flag in flags:
                if flag in self._FLAG_RESOLVED or flag not in self._FLAG_NOTES:
                    continue
                seen.setdefault(flag, set()).add(content.id)
        out = []
        for flag, ids in sorted(seen.items()):
            out.append(
                f"{self._FLAG_NOTES[flag]}（`{flag}` / 該当本文: {', '.join(sorted(ids)[:3])}）"
            )
        return out

    def _content_lines(self, content: ScreenContent) -> list[str]:
        lines: list[str] = []
        body = [
            r for r in sorted(content.regions, key=lambda r: r.reading_order)
            if r.role in (ROLE_MATERIAL_BODY, ROLE_MATERIAL_CONTEXT)
        ]
        for region in body:
            text = region.text.strip()
            if not text:
                continue
            if region.kind in CODE_KINDS:
                fence = fence_for(text)
                lines.append(f"{fence}{region.language_hint or ''}")
                lines.extend(text.split("\n"))
                lines.append(fence)
            elif region.kind == "heading":
                lines.append(f"### {escape_markdown_text(text.splitlines()[0])}")
            else:
                lines.append(escape_markdown_text(text))
            lines.append("")
            if region.candidates:
                lines.append(
                    "> ※ 判読できない箇所の候補（原文ではありません）: "
                    + escape_markdown_text(", ".join(region.candidates[:5]))
                )
                lines.append("")
        for note in content.structure_notes or []:
            lines.extend(self._structure_note_lines(note))
        return lines

    def _structure_note_lines(self, note: dict[str, Any]) -> list[str]:
        """図の関係。原文とは別の解釈であることを明示する (設計 6.4)。"""
        text = str(note.get("note", "")).strip()
        if not text:
            return []
        arrow = {"one_way": "→", "two_way": "↔", "none": "—"}.get(str(note.get("direction", "")), "")
        parts = [f"> 図の関係（原文ではなく解釈）: {escape_markdown_text(text)}"]
        if note.get("from") and note.get("to"):
            parts.append(
                f"> {escape_markdown_text(str(note['from']))} {arrow or '-'} "
                f"{escape_markdown_text(str(note['to']))}"
            )
        if note.get("group"):
            parts.append("> まとまり: " + escape_markdown_text(", ".join(map(str, note["group"][:8]))))
        if note.get("relation_unknown"):
            parts.append("> 関係不明")
        return parts + [""]

    # ------------------------------------------------------ study_outline.md
    def write_study_outline(self) -> Path:
        """日本語の学習ノートの骨組み。

        解説文はここでは書かない。章立て・出現する用語・出典だけを並べ、
        後段の生成工程が原資料へ戻れるようにする。
        """
        path = self.out_dir / "study_outline.md"
        terms = self._collect_terms()
        lines: list[str] = [
            f"# 学習ノートの骨組み: {Path(self.media['source_path']).name}",
            "",
            "章立てと、画面に現れた用語、出典の対応です。",
            "**解説文はまだ書かれていません。** ここは生成工程の入力であり、",
            "元の記録 (`lecture.md` / JSONL) を上書きするものではありません。",
            "",
            "## 用語（画面に現れ、発話でも言及された語）",
            "",
            "記号・数字を含む語か大文字で始まる語のうち、画面と発話の両方に現れたものです。",
            "コードの識別子は画面での表記をそのまま載せています。",
            "件数は照合の手がかりであり、重要度でも認識精度でもありません。",
            "",
            "| 用語 | 画面での出現(表示期間数) | 発話での言及(回数) | 初出 |",
            "|---|---|---|---|",
        ]
        for term, info in terms[:60]:
            lines.append(
                f"| `{term}` | {info['screen']} | {info['speech']} | "
                f"{format_timestamp(info['first_us'])} |"
            )
        lines.append("")
        lines.append("## 章立て")
        lines.append("")
        for block in self.blocks:
            occs = self._members(block)
            versions = self._screen_versions(occs)
            if not versions:
                continue
            title = block.title_hint.strip() or "（画面に見出しなし）"
            utts = self._utterances(occs)
            lines.append(
                f"### {format_timestamp(block.start_us)} {escape_markdown_text(title)}"
            )
            lines.append("")
            lines.append(f"- 教材単位: `{block.id}`（種別 {block.kind}）")
            lines.append(f"- 画面本文: {', '.join(v[1].id for v in versions[:4])}")
            lines.append(f"- 発話: {len(utts)} 件")
            if len(versions) > 1:
                lines.append(f"- この単位の中で内容が {len(versions) - 1} 回変わっています")
            lines.append("- 解説: （未生成）")
            lines.append("")
        atomic_write_text(path, "\n".join(lines) + "\n")
        return path

    def _collect_terms(self) -> list[tuple[str, dict[str, Any]]]:
        """画面に現れた語と、発話での言及回数を突き合わせる (設計 7.2)。"""
        # 大文字小文字の違いは同じ語として数え、表記は最も多く現れた形を使う。
        screen_counts: Counter = Counter()
        surface_counts: dict[str, Counter] = {}
        first_seen: dict[str, int] = {}
        for occ in self.occurrences:
            content = self.contents.get(occ.content_id or "")
            if content is None:
                continue
            for term in set(_TERM_RE.findall(content.body_text)):
                low = term.lower()
                if low in _STOPWORDS or len(term) < 4:
                    continue
                # 記号や数字を含まない普通の英単語は、画面と発話の両方に
                # 高頻度で現れても専門語とは限らない。形で絞る。
                if not _TECHNICAL_SHAPE.search(term) and not term[0].isupper():
                    continue
                screen_counts[low] += 1
                surface_counts.setdefault(low, Counter())[term] += 1
                first_seen.setdefault(low, occ.start_us)

        speech_text = " ".join(u.text_raw for u in self.utt_by_id.values()).lower()
        out: list[tuple[str, dict[str, Any]]] = []
        for low, screen_n in screen_counts.most_common(300):
            speech_n = speech_text.count(low)
            if speech_n == 0:
                continue  # 画面にしか出ない語は用語として拾わない
            surface = surface_counts[low].most_common(1)[0][0]
            out.append(
                (surface, {"screen": screen_n, "speech": speech_n, "first_us": first_seen[low]})
            )
        out.sort(key=lambda kv: -(kv[1]["screen"] + kv[1]["speech"]))
        return out


def write_profiles(store: Store, cfg: RunConfig, media: dict[str, Any]) -> dict[str, str]:
    exporter = ProfileExporter(store, cfg, media)
    return {
        "material.md": str(exporter.write_material()),
        "study_outline.md": str(exporter.write_study_outline()),
    }
