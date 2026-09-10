"""文書出力 (設計 10)。

- 画像を埋め込まない Markdown、JSONL、発話 SRT。
- 出力は正本 (SQLite) から再生成できる。原文の内容を変更しない。
- 原文に Markdown 記号やコードフェンスが含まれる場合、エスケープとフェンス長を調整する。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from ..config import RunConfig
from ..db.store import Store
from ..models import (
    FLAG_NOT_EXTRACTED,
    FLAG_TRUNCATED,
    FLAG_UNREADABLE,
    ROLE_BURNED_CAPTION,
    ROLE_MATERIAL_BODY,
    ROLE_MATERIAL_CONTEXT,
    ROLE_OTHER_SCREEN_TEXT,
    Region,
    ScreenContent,
    ScreenOccurrence,
    Utterance,
)
from ..util.atomic import atomic_write, atomic_write_text
from ..util.timeutil import format_srt_timestamp, format_timestamp
from .aligner import spans_without_screen, utterances_for_occurrence

log = logging.getLogger(__name__)

CODE_KINDS = {"code", "terminal_output"}
OUTPUT_FORMAT_VERSION = "1"

_LINE_LEAD = re.compile(r"^(\s*)([#>\-\+\*=\|]|\d+[\.\)])")


def escape_markdown_text(text: str) -> str:
    """原文を失わないための最小限のエスケープ。

    行頭の構造記号と、インラインで解釈される記号だけを打ち消す。
    """
    out_lines = []
    for line in text.split("\n"):
        line = line.replace("\\", "\\\\")
        for ch in ("`", "*", "_", "[", "]", "<", ">", "|"):
            line = line.replace(ch, "\\" + ch)
        line = _LINE_LEAD.sub(lambda m: f"{m.group(1)}\\{m.group(2)}", line)
        out_lines.append(line)
    return "\n".join(out_lines)


def fence_for(text: str) -> str:
    """コードフェンスの長さを内容に合わせる。"""
    longest = 0
    for match in re.finditer(r"`+", text):
        longest = max(longest, len(match.group(0)))
    return "`" * max(3, longest + 1)


def video_time_link(source_path: str, t_us: int, enabled: bool) -> str:
    if not enabled:
        return ""
    url = "file:///" + quote(str(source_path).replace("\\", "/"), safe="/:")
    return f"[{format_timestamp(t_us)}]({url}#t={t_us / 1_000_000:.3f})"


class Exporter:
    def __init__(self, store: Store, cfg: RunConfig, media: dict[str, Any]):
        self.store = store
        self.cfg = cfg
        self.media = media
        self.media_id = media["id"]
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.contents: dict[str, ScreenContent] = {c.id: c for c in store.all_contents()}
        self.occurrences: list[ScreenOccurrence] = store.occurrences(self.media_id)
        self.occ_by_id = {o.id: o for o in self.occurrences}
        self.blocks = store.blocks(self.media_id)
        self.alignments_by_occ, self.utt_by_id = utterances_for_occurrence(store, self.media_id)
        self.visual_events = store.visual_events(self.media_id)

    # ---------------------------------------------------------------- public
    def export_all(self) -> dict[str, str]:
        written = {
            "lecture.md": str(self.write_markdown()),
            "blocks.jsonl": str(self.write_blocks_jsonl()),
            "screen_contents.jsonl": str(self.write_contents_jsonl()),
            "screen_occurrences.jsonl": str(self.write_occurrences_jsonl()),
            "utterances.jsonl": str(self.write_utterances_jsonl()),
            "transcript.srt": str(self.write_srt()),
        }
        return written

    # -------------------------------------------------------------- markdown
    def write_markdown(self) -> Path:
        path = self.out_dir / "lecture.md"
        source = self.media["source_path"]
        link_enabled = self.cfg.export.video_link
        lines: list[str] = []
        lines.append(f"# 講義書き起こし: {Path(source).name}")
        lines.append("")
        lines.append(
            "この文書は画面に表示された原文と、発話の文字起こしを時系列で並べたものです。"
            "要約・翻訳・コード修正は行っていません。"
        )
        lines.append("")
        lines.append(f"- 元動画: `{source}`")
        lines.append(f"- 再生時間: {format_timestamp(self.media['duration_us'])}")
        lines.append(f"- 時刻原点 t0: {self.media['time_origin_us']} us")
        lines.append(f"- 読書ブロック数: {len(self.blocks)}")
        lines.append("")

        referenced_utterances: set[str] = set()

        index = 0
        unextracted_runs = 0
        while index < len(self.blocks):
            block = self.blocks[index]
            occs = [self.occ_by_id[o] for o in block.occurrence_ids if o in self.occ_by_id]
            if not occs:
                index += 1
                continue

            # 本文が未抽出の状態が続く区間は、1 状態ずつ節を立てると文書が読めなくなる。
            # 期間・件数・重なる発話をまとめて 1 節にする。個々の期間と根拠時刻は
            # screen_occurrences.jsonl に残っているので、情報は失われない。
            if all(o.content_id is None for o in occs):
                run_blocks = [block]
                j = index + 1
                while j < len(self.blocks):
                    nxt = [self.occ_by_id[o] for o in self.blocks[j].occurrence_ids if o in self.occ_by_id]
                    if not nxt or any(o.content_id is not None for o in nxt):
                        break
                    run_blocks.append(self.blocks[j])
                    j += 1
                unextracted_runs += 1
                lines.extend(
                    self._unextracted_run_lines(run_blocks, unextracted_runs, referenced_utterances)
                )
                index = j
                continue
            index += 1
            # 原文に見出しが無い場合は、内容の要約を無断でタイトルにしない (設計 10.1)。
            title = block.title_hint.strip()
            heading = f"## ブロック {block.index + 1}"
            if title:
                heading += f" — {escape_markdown_text(title)}"
            lines.append(heading)
            lines.append("")
            # 1. 表示開始〜終了時刻
            time_line = f"- 表示時刻: {format_timestamp(block.start_us)} – {format_timestamp(block.end_us)}"
            link = video_time_link(source, block.start_us, link_enabled)
            if link:
                time_line += f"（元動画: {link}）"
            lines.append(time_line)

            # 複数状態を含むブロックでは、最終状態の全文を本文として示す。
            # 途中の状態の全文は変化履歴に残し、原文を失わせない (設計 5.4)。
            primary_occ = next((o for o in reversed(occs) if o.content_id), occs[0])
            content = self.contents.get(primary_occ.content_id) if primary_occ.content_id else None
            # 2. 画面の題名・種類・ファイル名などの文脈
            lines.extend(self._context_lines(primary_occ, content, occs))
            lines.append("")

            # 3. 画面に表示された原文の全文
            if len(occs) > 1:
                lines.append(
                    f"**画面の原文（このブロックの最終状態 "
                    f"{format_timestamp(primary_occ.start_us)} 時点）**"
                )
                lines.append("")
            lines.extend(self._body_lines(primary_occ, content))

            # 4. 構造注記
            if self.cfg.export.include_structure_notes and content and content.structure_notes:
                lines.append("**構造注記（原文とは別の解釈）**")
                lines.append("")
                for note in content.structure_notes:
                    relation = note.get("relation", "")
                    unknown = note.get("relation_unknown")
                    suffix = "（関係不明）" if unknown else (f"（関係: {relation}）" if relation else "")
                    lines.append(f"- {escape_markdown_text(str(note.get('note', '')))}{suffix}")
                lines.append("")

            # 5. 個々の発話
            lines.extend(self._utterance_lines(occs, referenced_utterances))

            # 6. 編集中の下位状態と時刻付き差分
            if len(occs) > 1:
                lines.extend(self._sub_state_lines(occs))

            # 7. 不明箇所
            lines.extend(self._quality_lines(occs, content))
            lines.append("")

        # 画面の表示期間が無い区間の発話 (設計 8.2)
        orphan_lines = self._orphan_utterance_lines(referenced_utterances, link_enabled, source)
        if orphan_lines:
            lines.append("## 画面の表示期間に対応しない発話")
            lines.append("")
            lines.extend(orphan_lines)
            lines.append("")

        if self.cfg.export.include_ui_appendix:
            appendix = self._ui_appendix_lines()
            if appendix:
                lines.append("## 付録: 画面周辺の UI 文字")
                lines.append("")
                lines.append(
                    "本文のブロック境界には使っていません。表示時刻とともに記録しています。"
                )
                lines.append("")
                lines.extend(appendix)
                lines.append("")

        atomic_write_text(path, "\n".join(lines) + "\n")
        return path

    def _unextracted_run_lines(
        self, run_blocks: list, run_index: int, referenced: set[str]
    ) -> list[str]:
        """本文未抽出の状態が続く区間を、1 節にまとめて出す。"""
        from collections import Counter

        occs = [
            self.occ_by_id[o]
            for b in run_blocks
            for o in b.occurrence_ids
            if o in self.occ_by_id
        ]
        start_us = min(o.start_us for o in occs)
        end_us = max(o.end_us for o in occs)
        kinds = Counter(o.state_kind for o in occs)
        kind_text = " / ".join(f"{k} {n}" for k, n in kinds.most_common())

        lines = [
            f"## 未抽出区間 {run_index} — {format_timestamp(start_us)} – {format_timestamp(end_us)}",
            "",
            f"- 表示状態 {len(occs)} 件（{kind_text}）。画面本文は未抽出です。",
            "- 期間・境界・根拠時刻は `screen_occurrences.jsonl` に 1 件ずつ残しています。",
            "",
        ]

        seen: set[str] = set()
        utts: list[Utterance] = []
        for occ in occs:
            for a in self.alignments_by_occ.get(occ.id, []):
                utt = self.utt_by_id.get(a.utterance_id)
                if utt is None or utt.id in seen:
                    continue
                seen.add(utt.id)
                utts.append(utt)
        if utts:
            lines.append("**この区間に重なる発話**")
            lines.append("")
            for utt in sorted(utts, key=lambda u: u.start_us):
                referenced.add(utt.id)
                lines.append(
                    f"- `{format_timestamp(utt.start_us)} – {format_timestamp(utt.end_us)}` "
                    f"{escape_markdown_text(utt.text_raw)}"
                )
            lines.append("")
        else:
            lines.append("**発話**: この区間に重なる発話はありません。")
            lines.append("")
        return lines

    def _context_lines(
        self, occ: ScreenOccurrence, content: ScreenContent | None, occs: list[ScreenOccurrence]
    ) -> list[str]:
        lines: list[str] = []
        if content is not None:
            kind_label = content.screen_kind
            lines.append(f"- 画面種別: {kind_label}")
            ctx = content.context or {}
            if ctx.get("file_name"):
                lines.append(f"- ファイル名: `{ctx['file_name']}`")
            if ctx.get("application"):
                lines.append(f"- アプリケーション: {escape_markdown_text(str(ctx['application']))}")
            if ctx.get("tab_names"):
                names = ", ".join(f"`{t}`" for t in ctx["tab_names"])
                lines.append(f"- タブ: {names}")
            for region in sorted(content.regions, key=lambda r: r.reading_order):
                if region.role == ROLE_MATERIAL_CONTEXT and region.text.strip():
                    lines.append(f"- 文脈情報: {escape_markdown_text(region.text.strip())}")
        else:
            lines.append(f"- 画面種別: 未抽出（状態: {occ.state_kind}）")
        if len(occs) > 1:
            lines.append(f"- 含まれる表示状態: {len(occs)} 件（下に時刻付きで記載）")
        lines.append(
            "- 境界の推定幅: "
            f"開始 {format_timestamp(occ.boundary_start_lo_us)}–{format_timestamp(occ.boundary_start_hi_us)} / "
            f"終了 {format_timestamp(occs[-1].boundary_end_lo_us)}–{format_timestamp(occs[-1].boundary_end_hi_us)}"
        )
        return lines

    def _body_lines(self, occ: ScreenOccurrence, content: ScreenContent | None) -> list[str]:
        lines: list[str] = []
        if content is None:
            lines.append("> 画面本文は未抽出です。表示期間だけを記録しています。")
            lines.append(">")
            lines.append(f"> 状態: {occ.state_kind} / フラグ: {', '.join(occ.quality_flags) or 'なし'}")
            lines.append("")
            return lines
        body_regions = [
            r for r in sorted(content.regions, key=lambda r: r.reading_order) if r.role == ROLE_MATERIAL_BODY
        ]
        if not body_regions:
            lines.append("> 教材本文として抽出された領域はありません。")
            lines.append("")
            return lines
        for region in body_regions:
            lines.extend(self._region_lines(region))
        caption_regions = [r for r in content.regions if r.role == ROLE_BURNED_CAPTION]
        if caption_regions:
            lines.append("**画面に焼き込まれた字幕（音声文字起こしとは出典が異なります）**")
            lines.append("")
            for region in caption_regions:
                lines.append(f"- {escape_markdown_text(region.text.strip())}")
            lines.append("")
        return lines

    def _region_lines(self, region: Region) -> list[str]:
        lines: list[str] = []
        text = region.text
        if region.kind in CODE_KINDS:
            fence = fence_for(text)
            info = region.language_hint or ""
            lines.append(f"{fence}{info}")
            lines.extend(text.split("\n"))
            lines.append(fence)
            if region.line_numbers:
                lines.append("")
                lines.append(f"<!-- 行番号: {', '.join(region.line_numbers)} -->")
            lines.append("")
        elif region.kind == "heading":
            lines.append(f"### {escape_markdown_text(text.strip())}")
            lines.append("")
        elif region.kind == "bullet":
            for line in text.split("\n"):
                if line.strip():
                    lines.append(f"- {escape_markdown_text(line.strip())}")
            lines.append("")
        elif region.kind in ("table", "formula"):
            fence = fence_for(text)
            lines.append(f"{fence}text")
            lines.extend(text.split("\n"))
            lines.append(fence)
            lines.append("")
        else:
            lines.append(escape_markdown_text(text))
            lines.append("")
        flags = [f for f in region.flags if f]
        if flags or region.unreadable or region.candidates:
            notes = []
            if flags:
                notes.append(f"フラグ: {', '.join(sorted(set(flags)))}")
            if region.unreadable:
                notes.append(f"判読不能: {len(region.unreadable)} 箇所")
            if region.candidates:
                notes.append(f"候補（原文ではありません）: {', '.join(region.candidates[:5])}")
            lines.append(f"> ※ {escape_markdown_text(' / '.join(notes))}")
            lines.append("")
        return lines

    def _utterance_lines(
        self, occs: list[ScreenOccurrence], referenced: set[str]
    ) -> list[str]:
        lines: list[str] = []
        primary: list[tuple[Utterance, bool]] = []
        continued: list[Utterance] = []
        seen: set[str] = set()
        for occ in occs:
            for a in self.alignments_by_occ.get(occ.id, []):
                utt = self.utt_by_id.get(a.utterance_id)
                if utt is None or utt.id in seen:
                    continue
                seen.add(utt.id)
                if a.is_primary:
                    primary.append((utt, True))
                else:
                    continued.append(utt)
        if not primary and not continued:
            lines.append("**発話**: この表示期間に重なる発話はありません。")
            lines.append("")
            return lines

        if primary:
            lines.append("**発話**")
            lines.append("")
            for utt, _ in sorted(primary, key=lambda p: p[0].start_us):
                referenced.add(utt.id)
                lines.append(
                    f"- `{format_timestamp(utt.start_us)} – {format_timestamp(utt.end_us)}` "
                    f"{escape_markdown_text(utt.text_raw)}"
                    + (f" _(要確認: {', '.join(utt.quality_flags)})_" if utt.quality_flags else "")
                )
            lines.append("")
        if continued:
            lines.append("**継続・関連発話（原文は別ブロックに記載）**")
            lines.append("")
            for utt in sorted(continued, key=lambda u: u.start_us):
                referenced.add(utt.id)
                lines.append(
                    f"- `{format_timestamp(utt.start_us)} – {format_timestamp(utt.end_us)}` "
                    f"発話ID `{utt.id}`（この画面と時間が重なっています）"
                )
            lines.append("")
        return lines

    def _sub_state_lines(self, occs: list[ScreenOccurrence]) -> list[str]:
        lines = ["**このブロックに含まれる表示状態の変化履歴**", ""]
        prev_body: str | None = None
        for occ in occs:
            content = self.contents.get(occ.content_id) if occ.content_id else None
            body = content.body_text if content else None
            label = f"`{format_timestamp(occ.start_us)} – {format_timestamp(occ.end_us)}`"
            if body is None:
                lines.append(f"- {label} 状態 {occ.state_kind}（全文未確定）")
                continue
            diff = _short_diff(prev_body, body)
            lines.append(f"- {label} {escape_markdown_text(diff)}")
            # この状態の全文も残す。親ブロックへまとめたことで原文を失わせない。
            fence = fence_for(body)
            lines.append("")
            lines.append("<details><summary>この時点の画面全文</summary>")
            lines.append("")
            lines.append(f"{fence}text")
            lines.extend(body.split("\n"))
            lines.append(fence)
            lines.append("")
            lines.append("</details>")
            lines.append("")
            prev_body = body
        lines.append("")
        events = [
            ev
            for ev in self.visual_events
            if ev.start_us >= occs[0].start_us and ev.end_us <= occs[-1].end_us and ev.event_kind != "cursor"
        ]
        if events:
            lines.append("**このブロック内の視覚イベント**")
            lines.append("")
            for ev in events:
                lines.append(
                    f"- `{format_timestamp(ev.start_us)} – {format_timestamp(ev.end_us)}` "
                    f"{ev.event_kind}: {escape_markdown_text(ev.annotation)}"
                )
            lines.append("")
        return lines

    def _quality_lines(self, occs: list[ScreenOccurrence], content: ScreenContent | None) -> list[str]:
        notes: list[str] = []
        flags: set[str] = set()
        for occ in occs:
            flags |= set(occ.quality_flags)
        if content:
            flags |= set(content.quality_flags)
        if FLAG_NOT_EXTRACTED in flags:
            notes.append("この期間には全文未確定の表示状態が含まれます。")
        if FLAG_TRUNCATED in flags:
            notes.append("モデル出力が長さ制限で切れた領域があります。完成として扱っていません。")
        if FLAG_UNREADABLE in flags:
            notes.append("判読不能な文字が含まれます。推測では埋めていません。")
        if not notes:
            return []
        lines = ["**未確定箇所**", ""]
        lines.extend(f"- {n}" for n in notes)
        lines.append("")
        return lines

    def _orphan_utterance_lines(self, referenced: set[str], link_enabled: bool, source: str) -> list[str]:
        orphans = [u for u in self.utt_by_id.values() if u.id not in referenced]
        if not orphans:
            return []
        gaps = spans_without_screen(self.occurrences, self.media["duration_us"])
        lines = []
        if gaps:
            lines.append(
                "画面の表示期間が記録されていない区間があります: "
                + ", ".join(f"{format_timestamp(s)}–{format_timestamp(e)}" for s, e in gaps[:20])
            )
            lines.append("")
        for utt in sorted(orphans, key=lambda u: u.start_us):
            link = video_time_link(source, utt.start_us, link_enabled)
            lines.append(
                f"- `{format_timestamp(utt.start_us)} – {format_timestamp(utt.end_us)}` "
                f"{escape_markdown_text(utt.text_raw)}" + (f"（{link}）" if link else "")
            )
        return lines

    def _ui_appendix_lines(self) -> list[str]:
        lines: list[str] = []
        for occ in self.occurrences:
            content = self.contents.get(occ.content_id) if occ.content_id else None
            if content is None:
                continue
            ui_regions = [r for r in content.regions if r.role == ROLE_OTHER_SCREEN_TEXT and r.text.strip()]
            if not ui_regions:
                continue
            lines.append(
                f"- `{format_timestamp(occ.start_us)} – {format_timestamp(occ.end_us)}`: "
                + " / ".join(escape_markdown_text(r.text.strip().replace("\n", " ")) for r in ui_regions)
            )
        return lines

    # ----------------------------------------------------------------- jsonl
    def write_blocks_jsonl(self) -> Path:
        path = self.out_dir / "blocks.jsonl"
        with atomic_write(path) as fh:
            for block in self.blocks:
                refs = []
                seen = set()
                for occ_id in block.occurrence_ids:
                    for a in self.alignments_by_occ.get(occ_id, []):
                        if a.utterance_id in seen:
                            continue
                        seen.add(a.utterance_id)
                        refs.append(
                            {
                                "utterance_id": a.utterance_id,
                                "occurrence_id": a.occurrence_id,
                                "overlap_us": a.overlap_us,
                                "is_primary": a.is_primary,
                            }
                        )
                fh.write(
                    json.dumps(
                        {
                            "id": block.id,
                            "index": block.index,
                            "kind": block.kind,
                            "title_hint": block.title_hint,
                            "start_us": block.start_us,
                            "end_us": block.end_us,
                            "start": format_timestamp(block.start_us),
                            "end": format_timestamp(block.end_us),
                            "occurrence_ids": block.occurrence_ids,
                            "utterance_refs": refs,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return path

    def write_contents_jsonl(self) -> Path:
        path = self.out_dir / "screen_contents.jsonl"
        with atomic_write(path) as fh:
            for content in self.contents.values():
                fh.write(
                    json.dumps(
                        {
                            "id": content.id,
                            "text_hash": content.text_hash,
                            "screen_kind": content.screen_kind,
                            "context": content.context,
                            "reading_order": content.reading_order,
                            "quality_flags": content.quality_flags,
                            "structure_notes": content.structure_notes,
                            "source_attempt_ids": content.source_attempt_ids,
                            "regions": [r.to_dict() for r in content.regions],
                            "body_text": content.body_text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return path

    def write_occurrences_jsonl(self) -> Path:
        path = self.out_dir / "screen_occurrences.jsonl"
        with atomic_write(path) as fh:
            for occ in self.occurrences:
                fh.write(
                    json.dumps(
                        {
                            "id": occ.id,
                            "content_id": occ.content_id,
                            "start_us": occ.start_us,
                            "end_us": occ.end_us,
                            "start": format_timestamp(occ.start_us),
                            "end": format_timestamp(occ.end_us),
                            "boundary_start_us": [occ.boundary_start_lo_us, occ.boundary_start_hi_us],
                            "boundary_end_us": [occ.boundary_end_lo_us, occ.boundary_end_hi_us],
                            "state_kind": occ.state_kind,
                            "parent_block_id": occ.parent_block_id,
                            "change_summary": occ.change_summary,
                            "quality_flags": occ.quality_flags,
                            "evidence_refs": occ.evidence_refs,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return path

    def write_utterances_jsonl(self) -> Path:
        """原文の発話と時刻。重複参照による原文の二重保存を避ける (設計 10)。"""
        path = self.out_dir / "utterances.jsonl"
        with atomic_write(path) as fh:
            for utt in sorted(self.utt_by_id.values(), key=lambda u: u.start_us):
                occ_refs = [
                    {"occurrence_id": occ_id, "overlap_us": a.overlap_us, "is_primary": a.is_primary}
                    for occ_id, group in self.alignments_by_occ.items()
                    for a in group
                    if a.utterance_id == utt.id
                ]
                fh.write(
                    json.dumps(
                        {
                            "id": utt.id,
                            "start_us": utt.start_us,
                            "end_us": utt.end_us,
                            "start": format_timestamp(utt.start_us),
                            "end": format_timestamp(utt.end_us),
                            "text_raw": utt.text_raw,
                            "language": utt.language,
                            "source": utt.source,
                            "chunk_id": utt.chunk_id,
                            "quality_flags": utt.quality_flags,
                            "tokens": utt.tokens,
                            "occurrence_refs": occ_refs,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return path

    def write_srt(self) -> Path:
        """発話だけの字幕。画面 OCR の文を混ぜない (設計 10)。"""
        path = self.out_dir / "transcript.srt"
        with atomic_write(path) as fh:
            index = 1
            for utt in sorted(self.utt_by_id.values(), key=lambda u: (u.start_us, u.end_us)):
                end_us = max(utt.end_us, utt.start_us + 1000)
                fh.write(f"{index}\n")
                fh.write(f"{format_srt_timestamp(utt.start_us)} --> {format_srt_timestamp(end_us)}\n")
                fh.write(utt.text_raw.replace("\n", " ") + "\n\n")
                index += 1
        return path


def _short_diff(prev: str | None, current: str) -> str:
    if prev is None:
        return f"初出（{len(current.splitlines())} 行）"
    prev_lines = prev.split("\n")
    cur_lines = current.split("\n")
    added = [ln for ln in cur_lines if ln not in prev_lines]
    removed = [ln for ln in prev_lines if ln not in cur_lines]
    parts = []
    if added:
        parts.append(f"追加 {len(added)} 行: " + " / ".join(a.strip()[:60] for a in added[:3]))
    if removed:
        parts.append(f"削除 {len(removed)} 行: " + " / ".join(r.strip()[:60] for r in removed[:3]))
    if not parts:
        parts.append("行単位の差分なし（空白・記号のみの変化の可能性）")
    return "; ".join(parts)
