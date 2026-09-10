"""SQLite 正本アクセス層 (設計 9.1 / 12.2)。

- 解析結果・進捗・参照関係の正本はこの DB。
- Markdown・JSONL・SRT は、ここから再生成できる出力に過ぎない。
- 認識失敗の区間を「空欄の成功」として保存しない (設計 9.2)。
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..models import (
    Alignment,
    CoverageSpan,
    ReadingBlock,
    Region,
    ReviewItem,
    ScreenContent,
    ScreenOccurrence,
    Utterance,
    VisualEvent,
)

SCHEMA_VERSION = "1"
_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=False)


def _loads(text: str | None, default: Any) -> Any:
    if text is None or text == "":
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


class Store:
    """1 本の state.sqlite を扱う。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.execute("PRAGMA foreign_keys = ON")
        # ステージを別プロセスで並行実行する場合に備える。WAL は書き込みを 1 つに
        # 直列化するため、待たずに諦めると「database is locked」で落ちる。
        self.conn.execute("PRAGMA busy_timeout = 60000")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
        self.conn.execute(
            "INSERT OR IGNORE INTO schema_meta(key, value) VALUES ('schema_version', ?)",
            (SCHEMA_VERSION,),
        )

    def close(self) -> None:
        try:
            self.conn.commit()
        except sqlite3.Error:
            pass
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ media
    def upsert_media(self, media: dict[str, Any]) -> str:
        row = dict(media)
        row.setdefault("created_at", utcnow())
        row["stream_info"] = _dumps(row.get("stream_info", {}))
        row["timeline_flags"] = _dumps(row.get("timeline_flags", []))
        cols = list(row.keys())
        self.conn.execute(
            f"INSERT OR REPLACE INTO media ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            [row[c] for c in cols],
        )
        return row["id"]

    def get_media(self, media_id: str | None = None) -> dict[str, Any] | None:
        if media_id:
            cur = self.conn.execute("SELECT * FROM media WHERE id = ?", (media_id,))
        else:
            cur = self.conn.execute("SELECT * FROM media ORDER BY created_at LIMIT 1")
        row = cur.fetchone()
        if row is None:
            return None
        d = dict(row)
        d["stream_info"] = _loads(d["stream_info"], {})
        d["timeline_flags"] = _loads(d["timeline_flags"], [])
        return d

    def find_media_by_sha(self, sha256: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT id FROM media WHERE sha256 = ?", (sha256,)).fetchone()
        return self.get_media(row["id"]) if row else None

    # ------------------------------------------------------- frame observation
    def add_frame(self, frame: dict[str, Any]) -> str:
        row = dict(frame)
        row.setdefault("id", new_id("frm"))
        row["crop"] = _dumps(row["crop"]) if row.get("crop") is not None else None
        cols = list(row.keys())
        self.conn.execute(
            f"INSERT OR REPLACE INTO frame_observation ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            [row[c] for c in cols],
        )
        return row["id"]

    def get_frame(self, frame_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM frame_observation WHERE id = ?", (frame_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["crop"] = _loads(d["crop"], None)
        return d

    def frames_for_media(self, media_id: str, kind: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM frame_observation WHERE media_id = ?"
        args: list[Any] = [media_id]
        if kind:
            sql += " AND kind = ?"
            args.append(kind)
        sql += " ORDER BY t_us"
        out = []
        for row in self.conn.execute(sql, args):
            d = dict(row)
            d["crop"] = _loads(d["crop"], None)
            out.append(d)
        return out

    # ----------------------------------------------------- extraction attempts
    def add_attempt(self, attempt: dict[str, Any]) -> str:
        row = dict(attempt)
        row.setdefault("id", new_id("att"))
        row.setdefault("created_at", utcnow())
        cols = list(row.keys())
        self.conn.execute(
            f"INSERT OR REPLACE INTO extraction_attempt ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            [row[c] for c in cols],
        )
        return row["id"]

    def attempts_for_frame(self, frame_id: str) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self.conn.execute(
                "SELECT * FROM extraction_attempt WHERE frame_id = ? ORDER BY created_at", (frame_id,)
            )
        ]

    def attempt_stats(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) AS n FROM extraction_attempt GROUP BY status"
        ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    # ---------------------------------------------------------------- cache
    def cache_get(self, cache_key: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT payload FROM cache_entry WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        return _loads(row["payload"], None) if row else None

    def cache_put(self, cache_key: str, kind: str, payload: Any, attempt_id: str | None = None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO cache_entry (cache_key, kind, payload, attempt_id, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (cache_key, kind, _dumps(payload), attempt_id, utcnow()),
        )

    # -------------------------------------------------------- screen content
    def upsert_content(self, content: ScreenContent) -> str:
        self.conn.execute(
            "INSERT OR REPLACE INTO screen_content"
            " (id, text_hash, body_text, regions, reading_order, screen_kind, context,"
            "  structure_notes, quality_flags, source_attempt_ids)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                content.id,
                content.text_hash,
                content.body_text,
                _dumps([asdict(r) for r in content.regions]),
                _dumps(content.reading_order),
                content.screen_kind,
                _dumps(content.context),
                _dumps(content.structure_notes),
                _dumps(content.quality_flags),
                _dumps(content.source_attempt_ids),
            ),
        )
        return content.id

    def find_content_by_hash(self, text_hash: str) -> ScreenContent | None:
        row = self.conn.execute(
            "SELECT * FROM screen_content WHERE text_hash = ? LIMIT 1", (text_hash,)
        ).fetchone()
        return self._row_to_content(row) if row else None

    def get_content(self, content_id: str) -> ScreenContent | None:
        row = self.conn.execute("SELECT * FROM screen_content WHERE id = ?", (content_id,)).fetchone()
        return self._row_to_content(row) if row else None

    def all_contents(self) -> list[ScreenContent]:
        return [self._row_to_content(r) for r in self.conn.execute("SELECT * FROM screen_content")]

    @staticmethod
    def _row_to_content(row: sqlite3.Row) -> ScreenContent:
        return ScreenContent(
            id=row["id"],
            text_hash=row["text_hash"],
            body_text=row["body_text"],
            regions=[Region.from_dict(r) for r in _loads(row["regions"], [])],
            reading_order=_loads(row["reading_order"], []),
            screen_kind=row["screen_kind"],
            context=_loads(row["context"], {}),
            structure_notes=_loads(row["structure_notes"], []),
            quality_flags=_loads(row["quality_flags"], []),
            source_attempt_ids=_loads(row["source_attempt_ids"], []),
        )

    # ------------------------------------------------------ screen occurrence
    def upsert_occurrence(self, occ: ScreenOccurrence) -> str:
        self.conn.execute(
            "INSERT OR REPLACE INTO screen_occurrence"
            " (id, media_id, content_id, start_us, end_us, boundary_start_lo_us, boundary_start_hi_us,"
            "  boundary_end_lo_us, boundary_end_hi_us, state_kind, evidence_refs, parent_block_id,"
            "  change_summary, quality_flags)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                occ.id,
                occ.media_id,
                occ.content_id,
                occ.start_us,
                occ.end_us,
                occ.boundary_start_lo_us,
                occ.boundary_start_hi_us,
                occ.boundary_end_lo_us,
                occ.boundary_end_hi_us,
                occ.state_kind,
                _dumps(occ.evidence_refs),
                occ.parent_block_id,
                occ.change_summary,
                _dumps(occ.quality_flags),
            ),
        )
        return occ.id

    def occurrences(self, media_id: str) -> list[ScreenOccurrence]:
        rows = self.conn.execute(
            "SELECT * FROM screen_occurrence WHERE media_id = ? ORDER BY start_us, end_us", (media_id,)
        )
        return [self._row_to_occurrence(r) for r in rows]

    def get_occurrence(self, occ_id: str) -> ScreenOccurrence | None:
        row = self.conn.execute("SELECT * FROM screen_occurrence WHERE id = ?", (occ_id,)).fetchone()
        return self._row_to_occurrence(row) if row else None

    @staticmethod
    def _row_to_occurrence(row: sqlite3.Row) -> ScreenOccurrence:
        return ScreenOccurrence(
            id=row["id"],
            media_id=row["media_id"],
            content_id=row["content_id"],
            start_us=row["start_us"],
            end_us=row["end_us"],
            boundary_start_lo_us=row["boundary_start_lo_us"],
            boundary_start_hi_us=row["boundary_start_hi_us"],
            boundary_end_lo_us=row["boundary_end_lo_us"],
            boundary_end_hi_us=row["boundary_end_hi_us"],
            state_kind=row["state_kind"],
            evidence_refs=_loads(row["evidence_refs"], []),
            parent_block_id=row["parent_block_id"],
            change_summary=row["change_summary"],
            quality_flags=_loads(row["quality_flags"], []),
        )

    def clear_occurrences(self, media_id: str) -> None:
        self.conn.execute(
            "DELETE FROM alignment WHERE occurrence_id IN"
            " (SELECT id FROM screen_occurrence WHERE media_id = ?)",
            (media_id,),
        )
        self.conn.execute("DELETE FROM screen_occurrence WHERE media_id = ?", (media_id,))

    # ------------------------------------------------------------- utterance
    def upsert_utterance(self, utt: Utterance) -> str:
        self.conn.execute(
            "INSERT OR REPLACE INTO utterance"
            " (id, media_id, start_us, end_us, text_raw, language, tokens, quality_flags, source, chunk_id)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                utt.id,
                utt.media_id,
                utt.start_us,
                utt.end_us,
                utt.text_raw,
                utt.language,
                _dumps(utt.tokens),
                _dumps(utt.quality_flags),
                utt.source,
                utt.chunk_id,
            ),
        )
        return utt.id

    def utterances(self, media_id: str) -> list[Utterance]:
        rows = self.conn.execute(
            "SELECT * FROM utterance WHERE media_id = ? ORDER BY start_us, end_us", (media_id,)
        )
        return [
            Utterance(
                id=r["id"],
                media_id=r["media_id"],
                start_us=r["start_us"],
                end_us=r["end_us"],
                text_raw=r["text_raw"],
                language=r["language"],
                tokens=_loads(r["tokens"], []),
                quality_flags=_loads(r["quality_flags"], []),
                source=r["source"],
                chunk_id=r["chunk_id"],
            )
            for r in rows
        ]

    def delete_utterances_for_chunk(self, media_id: str, chunk_id: str) -> None:
        self.conn.execute(
            "DELETE FROM alignment WHERE utterance_id IN"
            " (SELECT id FROM utterance WHERE media_id = ? AND chunk_id = ?)",
            (media_id, chunk_id),
        )
        self.conn.execute("DELETE FROM utterance WHERE media_id = ? AND chunk_id = ?", (media_id, chunk_id))

    # ------------------------------------------------------------- alignment
    def replace_alignments(self, media_id: str, alignments: Iterable[Alignment]) -> int:
        self.conn.execute(
            "DELETE FROM alignment WHERE occurrence_id IN"
            " (SELECT id FROM screen_occurrence WHERE media_id = ?)",
            (media_id,),
        )
        n = 0
        for a in alignments:
            self.conn.execute(
                "INSERT OR REPLACE INTO alignment (occurrence_id, utterance_id, overlap_us, is_primary)"
                " VALUES (?,?,?,?)",
                (a.occurrence_id, a.utterance_id, a.overlap_us, int(a.is_primary)),
            )
            n += 1
        return n

    def alignments(self, media_id: str) -> list[Alignment]:
        rows = self.conn.execute(
            "SELECT a.* FROM alignment a JOIN screen_occurrence o ON o.id = a.occurrence_id"
            " WHERE o.media_id = ?",
            (media_id,),
        )
        return [
            Alignment(
                occurrence_id=r["occurrence_id"],
                utterance_id=r["utterance_id"],
                overlap_us=r["overlap_us"],
                is_primary=bool(r["is_primary"]),
            )
            for r in rows
        ]

    # ---------------------------------------------------------- reading block
    def replace_blocks(self, media_id: str, blocks: Sequence[ReadingBlock]) -> None:
        self.conn.execute("DELETE FROM reading_block WHERE media_id = ?", (media_id,))
        for b in blocks:
            self.conn.execute(
                "INSERT INTO reading_block"
                " (id, media_id, block_index, start_us, end_us, occurrence_ids, kind, title_hint)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (b.id, b.media_id, b.index, b.start_us, b.end_us, _dumps(b.occurrence_ids), b.kind, b.title_hint),
            )
        for b in blocks:
            for occ_id in b.occurrence_ids:
                self.conn.execute(
                    "UPDATE screen_occurrence SET parent_block_id = ? WHERE id = ?", (b.id, occ_id)
                )

    def blocks(self, media_id: str) -> list[ReadingBlock]:
        rows = self.conn.execute(
            "SELECT * FROM reading_block WHERE media_id = ? ORDER BY block_index", (media_id,)
        )
        return [
            ReadingBlock(
                id=r["id"],
                media_id=r["media_id"],
                index=r["block_index"],
                start_us=r["start_us"],
                end_us=r["end_us"],
                occurrence_ids=_loads(r["occurrence_ids"], []),
                kind=r["kind"],
                title_hint=r["title_hint"],
            )
            for r in rows
        ]

    # ---------------------------------------------------------- visual event
    def add_visual_event(self, ev: VisualEvent) -> str:
        self.conn.execute(
            "INSERT OR REPLACE INTO visual_event"
            " (id, media_id, start_us, end_us, region_ref, event_kind, annotation) VALUES (?,?,?,?,?,?,?)",
            (ev.id, ev.media_id, ev.start_us, ev.end_us, ev.region_ref, ev.event_kind, ev.annotation),
        )
        return ev.id

    def visual_events(self, media_id: str) -> list[VisualEvent]:
        rows = self.conn.execute(
            "SELECT * FROM visual_event WHERE media_id = ? ORDER BY start_us", (media_id,)
        )
        return [
            VisualEvent(
                id=r["id"],
                media_id=r["media_id"],
                start_us=r["start_us"],
                end_us=r["end_us"],
                region_ref=r["region_ref"],
                event_kind=r["event_kind"],
                annotation=r["annotation"],
            )
            for r in rows
        ]

    def clear_visual_events(self, media_id: str) -> None:
        self.conn.execute("DELETE FROM visual_event WHERE media_id = ?", (media_id,))

    # ----------------------------------------------------------- review item
    def add_review(self, item: ReviewItem) -> str:
        self.conn.execute(
            "INSERT OR REPLACE INTO review_item"
            " (id, media_id, target_kind, target_ref, reason, detail, resolution, review_status, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (
                item.id,
                item.media_id,
                item.target_kind,
                item.target_ref,
                item.reason,
                item.detail,
                item.resolution,
                item.review_status,
                utcnow(),
            ),
        )
        return item.id

    def reviews(self, media_id: str, status: str | None = None) -> list[ReviewItem]:
        sql = "SELECT * FROM review_item WHERE media_id = ?"
        args: list[Any] = [media_id]
        if status:
            sql += " AND review_status = ?"
            args.append(status)
        sql += " ORDER BY created_at"
        return [
            ReviewItem(
                id=r["id"],
                media_id=r["media_id"],
                target_kind=r["target_kind"],
                target_ref=r["target_ref"],
                reason=r["reason"],
                detail=r["detail"],
                resolution=r["resolution"],
                review_status=r["review_status"],
            )
            for r in self.conn.execute(sql, args)
        ]

    def clear_reviews_for_target(self, media_id: str, target_prefix: str) -> None:
        """特定の対象についてだけ、未解決の確認項目を消す。

        再解析する対象の分だけ作り直すために使う。ステージ全体を消すと、
        今回처理しない過去の対象の確認項目まで失われる。
        """
        self.conn.execute(
            "DELETE FROM review_item WHERE media_id = ? AND review_status = 'open'"
            " AND target_ref LIKE ?",
            (media_id, target_prefix + "%"),
        )

    def clear_reviews_by_reason_prefix(self, media_id: str, prefix: str) -> None:
        """再解析時に、そのステージが作った未解決項目だけを作り直す。

        解決済み (resolved/wontfix) は人の作業結果なので消さない。
        """
        self.conn.execute(
            "DELETE FROM review_item WHERE media_id = ? AND review_status = 'open' AND reason LIKE ?",
            (media_id, prefix + "%"),
        )

    # -------------------------------------------------------------- coverage
    def replace_coverage(self, media_id: str, track: str, spans: Iterable[CoverageSpan]) -> None:
        self.conn.execute("DELETE FROM coverage_span WHERE media_id = ? AND track = ?", (media_id, track))
        for s in spans:
            self.conn.execute(
                "INSERT INTO coverage_span (id, media_id, track, start_us, end_us, state, detail)"
                " VALUES (?,?,?,?,?,?,?)",
                (s.id, s.media_id, s.track, s.start_us, s.end_us, s.state, s.detail),
            )

    def coverage(self, media_id: str, track: str | None = None) -> list[CoverageSpan]:
        sql = "SELECT * FROM coverage_span WHERE media_id = ?"
        args: list[Any] = [media_id]
        if track:
            sql += " AND track = ?"
            args.append(track)
        sql += " ORDER BY track, start_us"
        return [
            CoverageSpan(
                id=r["id"],
                media_id=r["media_id"],
                track=r["track"],
                start_us=r["start_us"],
                end_us=r["end_us"],
                state=r["state"],
                detail=r["detail"],
            )
            for r in self.conn.execute(sql, args)
        ]

    # ------------------------------------------------------------------- job
    def get_job(self, media_id: str | None, stage: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM job WHERE media_id IS ? AND stage = ?", (media_id, stage)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["checkpoint"] = _loads(d["checkpoint"], {})
        return d

    def start_job(self, media_id: str | None, stage: str, input_hash: str, config_hash: str) -> str:
        existing = self.get_job(media_id, stage)
        job_id = existing["id"] if existing else new_id("job")
        checkpoint = existing["checkpoint"] if existing else {}
        # 入力・設定が変わっていればチェックポイントを引き継がない (設計 12.2)。
        if existing and (existing["input_hash"] != input_hash or existing["config_hash"] != config_hash):
            checkpoint = {}
        self.conn.execute(
            "INSERT OR REPLACE INTO job"
            " (id, media_id, stage, input_hash, config_hash, status, checkpoint, started_at, finished_at, error)"
            " VALUES (?,?,?,?,?,?,?,?,NULL,NULL)",
            (job_id, media_id, stage, input_hash, config_hash, "running", _dumps(checkpoint), utcnow()),
        )
        return job_id

    def update_checkpoint(self, job_id: str, checkpoint: dict[str, Any]) -> None:
        self.conn.execute("UPDATE job SET checkpoint = ? WHERE id = ?", (_dumps(checkpoint), job_id))

    def finish_job(self, job_id: str, status: str = "done", error: str | None = None) -> None:
        self.conn.execute(
            "UPDATE job SET status = ?, finished_at = ?, error = ? WHERE id = ?",
            (status, utcnow(), error, job_id),
        )

    def jobs(self) -> list[dict[str, Any]]:
        out = []
        for row in self.conn.execute("SELECT * FROM job ORDER BY started_at"):
            d = dict(row)
            d["checkpoint"] = _loads(d["checkpoint"], {})
            out.append(d)
        return out

    def stage_is_done(self, media_id: str | None, stage: str, input_hash: str, config_hash: str) -> bool:
        job = self.get_job(media_id, stage)
        return bool(
            job
            and job["status"] == "done"
            and job["input_hash"] == input_hash
            and job["config_hash"] == config_hash
        )

    def invalidate_stage(self, media_id: str | None, stage: str) -> None:
        self.conn.execute("DELETE FROM job WHERE media_id IS ? AND stage = ?", (media_id, stage))

    # ---------------------------------------------------------------- metric
    def add_metric(self, media_id: str | None, stage: str, name: str, value: float, unit: str = "") -> None:
        self.conn.execute(
            "INSERT INTO metric (media_id, stage, name, value, unit, created_at) VALUES (?,?,?,?,?,?)",
            (media_id, stage, name, float(value), unit, utcnow()),
        )

    def metrics(self, media_id: str | None = None) -> list[dict[str, Any]]:
        if media_id:
            rows = self.conn.execute(
                "SELECT * FROM metric WHERE media_id = ? ORDER BY id", (media_id,)
            )
        else:
            rows = self.conn.execute("SELECT * FROM metric ORDER BY id")
        return [dict(r) for r in rows]

    # ------------------------------------------------------------ correction
    def add_correction(
        self, target_kind: str, target_ref: str, field: str, original: str, corrected: str, author: str = "user"
    ) -> str:
        cid = new_id("cor")
        self.conn.execute(
            "INSERT INTO correction"
            " (id, target_kind, target_ref, field, original_value, corrected_value, author, created_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (cid, target_kind, target_ref, field, original, corrected, author, utcnow()),
        )
        return cid

    def corrections(self, target_kind: str | None = None) -> list[dict[str, Any]]:
        if target_kind:
            rows = self.conn.execute(
                "SELECT * FROM correction WHERE target_kind = ? ORDER BY created_at", (target_kind,)
            )
        else:
            rows = self.conn.execute("SELECT * FROM correction ORDER BY created_at")
        return [dict(r) for r in rows]
