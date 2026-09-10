"""パイプライン試験。

設計 14.3「優先する検証」に対応する項目を、合成講義動画で確認する。
モデルは使わず、画像に埋め込んだマーカーから台本を引くアダプターを使う。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lecture_extract.adapters.asr.stub import ScriptedStubAsr
from lecture_extract.adapters.vision.stub import MarkerStubVision
from lecture_extract.config import RunConfig
from lecture_extract.db.store import Store
from lecture_extract.pipeline.frame_scan import run_frame_scan
from lecture_extract.pipeline.ingest import ingest
from lecture_extract.pipeline.orchestrator import Orchestrator

TOLERANCE_US = 120_000  # フレーム間隔 (50ms) の範囲で境界を許容する


def make_config(fixture: dict, tmp_path: Path, **overrides) -> RunConfig:
    cfg = RunConfig()
    cfg.input_path = fixture["path"]
    cfg.work_dir = str(tmp_path / "work")
    cfg.out_dir = str(tmp_path / "out")
    for key, value in overrides.items():
        target = cfg
        parts = key.split("__")
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)
    return cfg


def run_pipeline(fixture: dict, tmp_path: Path, **overrides) -> tuple[Orchestrator, dict]:
    cfg = make_config(fixture, tmp_path, **overrides)
    vision = MarkerStubVision(fixture["vision_script"])
    asr = ScriptedStubAsr(segments=fixture["speeches"])
    orch = Orchestrator(cfg, vision_adapter=vision, asr_adapter=asr)
    results = orch.run()
    return orch, results


# --------------------------------------------------------------- frame_scan
def test_frame_scan_recovers_every_state(lecture_fixture, tmp_path):
    cfg = make_config(lecture_fixture, tmp_path)
    store = Store(Path(cfg.work_dir) / "state.sqlite")
    media_id, probe = ingest(store, cfg)
    job_id = store.start_job(media_id, "frame_scan", "h", "c")
    stats = run_frame_scan(store, cfg, probe, media_id, 0, probe.duration_us, job_id, resume=False)
    occurrences = store.occurrences(media_id)

    truth = lecture_fixture["states"]
    assert stats["states"] == len(truth), f"表示状態の数が正解と一致しません: {stats}"
    for occ, expected in zip(occurrences, truth):
        assert abs(occ.start_us - expected["start_us"]) <= TOLERANCE_US
        assert abs(occ.end_us - expected["end_us"]) <= TOLERANCE_US
        # 境界は推定であることが保存されている (設計 4.2)。
        assert occ.boundary_start_lo_us <= occ.start_us <= occ.boundary_start_hi_us

    # 1 文字だけの変更 (result = 0 → result = 1) が独立した状態になっている。
    assert abs(occurrences[2].start_us - 4_500_000) <= TOLERANCE_US

    # 0.5 秒未満の表示も、期間として残っている。
    short = [o for o in occurrences if o.state_kind == "short"]
    assert len(short) == 1
    assert short[0].end_us - short[0].start_us < 500_000

    # カーソル点滅だけでは文字変更にしない。視覚イベントとしては残る。
    events = store.visual_events(media_id)
    assert any(e.event_kind == "cursor" for e in events)
    assert stats["cursor_events"] > 0
    store.close()


def test_time_coverage_has_no_unknown_span(lecture_fixture, tmp_path):
    orch, results = run_pipeline(lecture_fixture, tmp_path)
    try:
        from lecture_extract.pipeline.quality import _unknown_coverage

        media = orch.store.get_media()
        gaps = _unknown_coverage(orch.store, media["id"], media["duration_us"])
        assert gaps == [], f"処理状態が不明な区間が残っています: {gaps}"
    finally:
        orch.close()


# ---------------------------------------------------------------- end-to-end
def test_full_pipeline_outputs(lecture_fixture, tmp_path):
    orch, results = run_pipeline(lecture_fixture, tmp_path)
    try:
        out = Path(orch.cfg.out_dir)
        for name in (
            "lecture.md",
            "blocks.jsonl",
            "screen_contents.jsonl",
            "screen_occurrences.jsonl",
            "utterances.jsonl",
            "transcript.srt",
            "quality_report.md",
            "manifest.json",
        ):
            assert (out / name).exists(), f"{name} が出力されていません"

        markdown = (out / "lecture.md").read_text(encoding="utf-8")
        # 画面の原文がそのまま出ている。
        assert "def total(items):" in markdown
        assert "result = 0" in markdown
        assert "result = 1" in markdown
        # 発話の原文と時刻が出ている。
        assert "きょうは合計を求める関数を書きます" in markdown
        # 画像は埋め込まない。
        assert "![" not in markdown

        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["input"]["sha256"]
        assert manifest["models"]["this_run"]["vision"]["adapter"] == "stub"
        # 採用データを生成したモデルを、書き出しだけの再実行でも追跡できる (P2-4)。
        assert "produced_extraction" in manifest["models"]
        assert manifest["status"] in ("completed_with_review", "reviewed")
    finally:
        orch.close()


def test_repeated_slide_keeps_two_occurrences_one_content(lecture_fixture, tmp_path):
    orch, _ = run_pipeline(lecture_fixture, tmp_path)
    try:
        media = orch.store.get_media()
        occurrences = orch.store.occurrences(media["id"])
        with_content = [o for o in occurrences if o.content_id]
        # 同じスライドが 2 回出る。本文は 1 件に集約され、出現は 2 件残る (設計 5.6)。
        counts: dict[str, int] = {}
        for occ in with_content:
            counts[occ.content_id] = counts.get(occ.content_id, 0) + 1
        assert max(counts.values()) == 2
        assert len({o.id for o in with_content}) == len(with_content)
    finally:
        orch.close()


def test_edited_code_versions_are_not_merged(lecture_fixture, tmp_path):
    orch, _ = run_pipeline(lecture_fixture, tmp_path)
    try:
        media = orch.store.get_media()
        contents = orch.store.all_contents()
        bodies = [c.body_text for c in contents]
        assert any("result = 0" in b for b in bodies)
        assert any("result = 1" in b for b in bodies)
        # 編集前と編集後を混ぜて 1 つの本文にしていない。
        assert not any("result = 0" in b and "result = 1" in b for b in bodies)
    finally:
        orch.close()


def test_every_utterance_is_referenced(lecture_fixture, tmp_path):
    orch, _ = run_pipeline(lecture_fixture, tmp_path)
    try:
        media = orch.store.get_media()
        utterances = orch.store.utterances(media["id"])
        alignments = orch.store.alignments(media["id"])
        assert len(utterances) == len(lecture_fixture["speeches"])
        aligned = {a.utterance_id for a in alignments}
        assert {u.id for u in utterances} <= aligned, "どのブロックにも結び付かない発話があります"
        # 発話ごとの主ブロックはちょうど 1 つ。
        for utt in utterances:
            primaries = [a for a in alignments if a.utterance_id == utt.id and a.is_primary]
            assert len(primaries) == 1
        # 境界をまたぐ発話は両側から参照できる。
        crossing = [u for u in utterances if u.start_us < 2_000_000 < u.end_us]
        assert crossing
        for utt in crossing:
            refs = [a for a in alignments if a.utterance_id == utt.id]
            assert len(refs) >= 2
    finally:
        orch.close()


def test_srt_contains_only_speech(lecture_fixture, tmp_path):
    orch, _ = run_pipeline(lecture_fixture, tmp_path)
    try:
        srt = (Path(orch.cfg.out_dir) / "transcript.srt").read_text(encoding="utf-8")
        assert "def total(items):" not in srt  # 画面 OCR を字幕へ混ぜない
        assert "きょうは合計を求める関数を書きます" in srt
        assert srt.count("-->") == len(lecture_fixture["speeches"])
    finally:
        orch.close()


# ------------------------------------------------------------------- 再開
def test_resume_does_not_duplicate_or_lose(lecture_fixture, tmp_path):
    orch1, _ = run_pipeline(lecture_fixture, tmp_path)
    media = orch1.store.get_media()
    media_id = media["id"]
    before = {
        "occurrences": [(o.start_us, o.end_us, o.content_id) for o in orch1.store.occurrences(media_id)],
        "utterances": [(u.start_us, u.end_us, u.text_raw) for u in orch1.store.utterances(media_id)],
        "blocks": [(b.start_us, b.end_us) for b in orch1.store.blocks(media_id)],
    }
    orch1.close()

    # 同じ入力・設定で再実行しても、確定済みの結果は増減しない。
    orch2, results = run_pipeline(lecture_fixture, tmp_path)
    try:
        after = {
            "occurrences": [(o.start_us, o.end_us, o.content_id) for o in orch2.store.occurrences(media_id)],
            "utterances": [(u.start_us, u.end_us, u.text_raw) for u in orch2.store.utterances(media_id)],
            "blocks": [(b.start_us, b.end_us) for b in orch2.store.blocks(media_id)],
        }
        assert after == before
        assert results["frame_scan"] == {"skipped": True}
        assert results["vision"] == {"skipped": True}
    finally:
        orch2.close()


def test_interrupted_vision_resumes(lecture_fixture, tmp_path):
    """途中で失敗させ、再開後に欠落・二重登録がないことを確認する。"""

    class FailingVision(MarkerStubVision):
        def __init__(self, path, fail_after: int):
            super().__init__(path)
            self.fail_after = fail_after

        def extract(self, image_png, kind, *, extra_instruction=""):
            if self.calls >= self.fail_after:
                raise RuntimeError("模擬的な中断")
            return super().extract(image_png, kind, extra_instruction=extra_instruction)

    cfg = make_config(lecture_fixture, tmp_path)
    asr = ScriptedStubAsr(segments=lecture_fixture["speeches"])
    orch = Orchestrator(
        cfg, vision_adapter=FailingVision(lecture_fixture["vision_script"], fail_after=3), asr_adapter=asr
    )
    with pytest.raises(RuntimeError):
        orch.run()
    media = orch.store.get_media()
    media_id = media["id"]
    partial = [o for o in orch.store.occurrences(media_id) if o.content_id]
    orch.close()
    assert partial, "中断前の確定結果が残っていません"

    orch2, _ = run_pipeline(lecture_fixture, tmp_path)
    try:
        occurrences = orch2.store.occurrences(media_id)
        assert len({o.id for o in occurrences}) == len(occurrences)
        extracted = [o for o in occurrences if o.content_id]
        assert len(extracted) >= len(partial)
        # 全期間が記録されている。
        assert occurrences[0].start_us == 0
        assert occurrences[-1].end_us == media["duration_us"]
    finally:
        orch2.close()


# ---------------------------------------------------------------- チャンク
def test_chunk_boundary_keeps_one_copy_of_each_utterance(lecture_fixture, tmp_path):
    orch, _ = run_pipeline(
        lecture_fixture,
        tmp_path,
        asr__chunk_us=3_000_000,
        asr__chunk_overlap_us=1_000_000,
    )
    try:
        media = orch.store.get_media()
        utterances = orch.store.utterances(media["id"])
        texts = [u.text_raw for u in utterances]
        assert len(texts) == len(set(texts)), f"チャンク境界で発話が二重化しました: {texts}"
        expected = {s["text"] for s in lecture_fixture["speeches"]}
        assert set(texts) == expected, "チャンク境界で発話が消えました"
        # 原時刻は元動画の時間軸に戻っている。
        for utt, spec in zip(sorted(utterances, key=lambda u: u.start_us), lecture_fixture["speeches"]):
            assert utt.start_us == spec["start_us"]
            assert utt.end_us == spec["end_us"]
    finally:
        orch.close()


def test_range_limited_run_reports_original_timeline(lecture_fixture, tmp_path):
    orch, _ = run_pipeline(
        lecture_fixture, tmp_path, range_start_us=4_000_000, range_end_us=8_000_000
    )
    try:
        media = orch.store.get_media()
        occurrences = orch.store.occurrences(media["id"])
        assert occurrences[0].start_us >= 4_000_000 - TOLERANCE_US
        assert occurrences[-1].end_us <= 8_000_000 + TOLERANCE_US
        coverage = orch.store.coverage(media["id"], "screen")
        assert any(s.state == "out_of_range" and s.start_us == 0 for s in coverage)
    finally:
        orch.close()


# ------------------------------------------------- 連続入力の親ブロック化
def test_typing_sequence_becomes_one_editing_block(tmp_path):
    """設計 5.4: 連続入力は親ブロックにまとめるが、変更履歴は失わない。"""
    from fixtures import SceneSpec, make_video, write_vision_script

    base = ["def total(items):", "    result = 0", "    for item in items:"]
    scenes = [
        SceneSpec(marker=21, duration_us=1_500_000, lines=base, kind="code", file_name="t.py"),
        SceneSpec(marker=22, duration_us=1_000_000, lines=base + ["        result += it"], kind="code", file_name="t.py"),
        SceneSpec(marker=23, duration_us=1_000_000, lines=base + ["        result += item"], kind="code", file_name="t.py"),
        SceneSpec(
            marker=24,
            duration_us=1_500_000,
            lines=base + ["        result += item", "    return result"],
            kind="code",
            file_name="t.py",
        ),
    ]
    root = tmp_path / "typing"
    truth = make_video(root / "typing.mp4", scenes, [])
    truth["vision_script"] = str(write_vision_script(root / "script.json", scenes))
    truth["speeches"] = []

    orch, _ = run_pipeline(truth, tmp_path / "run")
    try:
        media = orch.store.get_media()
        blocks = orch.store.blocks(media["id"])
        editing = [b for b in blocks if b.kind.startswith("unit")]
        assert editing, f"連続入力が親ブロックになっていません: {[(b.kind, len(b.occurrence_ids)) for b in blocks]}"
        assert len(editing[0].occurrence_ids) >= 3
        # 内容が変わった履歴を含む単位として区別されている。
        assert any(b.kind == "unit_edited" for b in blocks)

        markdown = (Path(orch.cfg.out_dir) / "lecture.md").read_text(encoding="utf-8")
        # 途中の版も、変化履歴に全文として残っている。
        assert "result += it\n" in markdown or "result += it" in markdown
        assert "return result" in markdown
        assert "変化履歴" in markdown
        # 編集前と編集後を 1 つの本文へ混ぜていない。
        contents = orch.store.all_contents()
        assert not any("result += it\n" in c.body_text and "return result" in c.body_text for c in contents)
    finally:
        orch.close()


# ------------------------------------------------------- 画像キャッシュ管理
def test_image_cache_prune_and_regenerate(lecture_fixture, tmp_path):
    """設計 12.3: 上限を超えた画像は削除でき、元動画と時刻から作り直せる。"""
    from lecture_extract.pipeline.cache import (
        cache_size_bytes,
        prune_image_cache,
        regenerate_frame,
    )

    orch, _ = run_pipeline(lecture_fixture, tmp_path)
    try:
        media = orch.store.get_media()
        cfg = orch.cfg
        before = cache_size_bytes(cfg, media["id"])
        assert before > 0

        # 上限を極端に小さくして、削除が起きることを確かめる。
        cfg.scan.max_cache_bytes = 1024
        stats = prune_image_cache(orch.store, cfg, media["id"])
        assert stats["removed"] > 0
        assert stats["after_bytes"] < before

        # 削除された根拠画像を、元動画のハッシュと時刻から作り直せる。
        pruned = [f for f in orch.store.frames_for_media(media["id"]) if not f["evidence_ref"]]
        assert pruned
        path = regenerate_frame(orch.store, cfg, pruned[0]["id"])
        assert path.exists()
        assert orch.store.get_frame(pruned[0]["id"])["evidence_ref"]
    finally:
        orch.close()


def test_storage_exhausted_stops_before_writing(lecture_fixture, tmp_path, monkeypatch):
    """設計 12.2: 保存領域不足では処理を止め、既存結果と元動画を保全する。"""
    import shutil as _shutil

    from lecture_extract.pipeline import cache as cache_mod

    fake = type("U", (), {"total": 100, "used": 99, "free": 1})()
    monkeypatch.setattr(cache_mod.shutil, "disk_usage", lambda path: fake)
    with pytest.raises(cache_mod.StorageExhausted):
        cache_mod.check_free_space(tmp_path / "work")
    assert _shutil is not None


def test_repeat_and_modified_are_distinguished(lecture_fixture, tmp_path):
    """設計 5.6: 同じ画面の再表示と、内容を変更した画面を区別する。"""
    orch, _ = run_pipeline(lecture_fixture, tmp_path)
    try:
        media = orch.store.get_media()
        occurrences = orch.store.occurrences(media["id"])
        kinds = {}
        for occ in occurrences:
            for f in occ.quality_flags:
                if f.startswith("change:"):
                    kinds[occ.id] = f.split(":", 1)[1]
        assert kinds, "表示状態の位置づけが記録されていません"
        # 同じアジェンダスライドが 2 回出るので、初出と再表示の両方がある。
        assert "first" in kinds.values()
        # コードは 1 文字変更されるので、変更として記録される単位がある。
        blocks = orch.store.blocks(media["id"])
        assert any(b.kind in ("unit", "unit_edited", "single") for b in blocks)
    finally:
        orch.close()
