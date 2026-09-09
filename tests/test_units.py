"""単体試験: 時刻演算、同期、正規化、出力の脱落防止。"""

from __future__ import annotations

from fractions import Fraction

import pytest

from lecture_extract.adapters.vision.llama_server import parse_completion, validate_payload
from lecture_extract.models import ScreenOccurrence, Utterance
from lecture_extract.pipeline.aligner import align_occurrences
from lecture_extract.pipeline.exporter import escape_markdown_text, fence_for
from lecture_extract.pipeline.vision_extract import merge_tile_lines
from lecture_extract.util.textnorm import normalize_for_compare, normalize_for_dedup_utterance
from lecture_extract.util.timeutil import (
    contains_half_open,
    format_srt_timestamp,
    format_timestamp,
    overlap_us,
    pts_to_us,
)


def test_pts_to_us_uses_time_base_not_frame_number():
    # 可変フレームレートでも PTS x time_base で時刻を求める (設計 4.1)。
    assert pts_to_us(90_000, Fraction(1, 90_000)) == 1_000_000
    assert pts_to_us(1001, Fraction(1, 30_000)) == 33_367
    assert pts_to_us(90_000, Fraction(1, 90_000), t0_us=500_000) == 500_000


def test_format_timestamps():
    assert format_timestamp(3_723_456_000) == "1:02:03.456"
    assert format_srt_timestamp(3_723_456_000) == "01:02:03,456"
    assert format_srt_timestamp(-5) == "00:00:00,000"


def test_overlap_and_half_open_boundary():
    assert overlap_us(0, 100, 50, 150) == 50
    assert overlap_us(0, 100, 100, 150) == 0  # 半開区間なので境界は重ならない
    assert contains_half_open(0, 100, 0)
    assert not contains_half_open(0, 100, 100)


def _occ(occ_id: str, start: int, end: int) -> ScreenOccurrence:
    return ScreenOccurrence(
        id=occ_id,
        media_id="m",
        content_id=None,
        start_us=start,
        end_us=end,
        boundary_start_lo_us=start,
        boundary_start_hi_us=start,
        boundary_end_lo_us=end,
        boundary_end_hi_us=end,
    )


def _utt(utt_id: str, start: int, end: int, text: str = "x") -> Utterance:
    return Utterance(id=utt_id, media_id="m", start_us=start, end_us=end, text_raw=text)


def test_alignment_crossing_boundary_has_single_primary():
    occs = [_occ("o1", 0, 1000), _occ("o2", 1000, 3000)]
    utts = [_utt("u1", 800, 2500)]
    alignments = align_occurrences(occs, utts)
    assert {a.occurrence_id for a in alignments} == {"o1", "o2"}
    primaries = [a for a in alignments if a.is_primary]
    assert len(primaries) == 1
    # 重複時間が最大のブロックが主になる。
    assert primaries[0].occurrence_id == "o2"


def test_alignment_ties_go_to_earlier_block():
    occs = [_occ("o1", 0, 1000), _occ("o2", 1000, 2000)]
    utts = [_utt("u1", 500, 1500)]
    alignments = align_occurrences(occs, utts)
    primaries = [a for a in alignments if a.is_primary]
    assert len(primaries) == 1
    assert primaries[0].occurrence_id == "o1"


def test_alignment_loses_no_utterance():
    occs = [_occ(f"o{i}", i * 1000, (i + 1) * 1000) for i in range(5)]
    utts = [_utt(f"u{i}", i * 700, i * 700 + 400) for i in range(7)]
    alignments = align_occurrences(occs, utts)
    aligned = {a.utterance_id for a in alignments}
    expected = {u.id for u in utts if u.start_us < occs[-1].end_us}
    assert expected <= aligned


def test_code_normalization_keeps_indent_and_symbols():
    a = "def f():\n    return 1"
    b = "def f():\n        return 1"
    assert normalize_for_compare(a, is_code=True) != normalize_for_compare(b, is_code=True)
    # 行末空白の違いだけは同一視する。
    assert normalize_for_compare("x = 1  ", is_code=True) == normalize_for_compare("x = 1", is_code=True)
    # 1 文字違いは別物として扱う。
    assert normalize_for_compare("result = 0", is_code=True) != normalize_for_compare("result = 1", is_code=True)


def test_utterance_dedup_normalization():
    assert normalize_for_dedup_utterance("これは、テストです。") == normalize_for_dedup_utterance("これはテストです")


def test_merge_tile_lines_uses_overlap():
    acc = ["a", "b", "c"]
    new = ["b", "c", "d"]
    merged, ok = merge_tile_lines(acc, new, 3)
    assert merged == ["a", "b", "c", "d"]
    assert ok


def test_merge_tile_lines_reports_unverified_join():
    merged, ok = merge_tile_lines(["a", "b"], ["x", "y"], 3)
    assert merged == ["a", "b", "x", "y"]
    assert not ok  # 重複が一致しなかったことを隠さない


def test_markdown_escaping_preserves_content():
    text = "# not a heading\n- not a list\n`code` *star*"
    escaped = escape_markdown_text(text)
    assert "\\#" in escaped and "\\-" in escaped and "\\`" in escaped
    # 文字そのものは失われていない。
    for ch in "#-`*":
        assert ch in escaped


def test_code_fence_length_adapts():
    assert fence_for("plain") == "```"
    assert fence_for("a ``` b") == "````"
    assert fence_for("a ````` b") == "``````"


def test_truncated_completion_is_not_success():
    data = {
        "choices": [{"finish_reason": "length", "message": {"content": '{"lines":["a"]'}}],
    }
    result = parse_completion(data, "crop", 10)
    assert result.status == "truncated"
    assert not result.ok


def test_output_complete_false_is_not_success():
    data = {
        "choices": [
            {"finish_reason": "stop", "message": {"content": '{"lines":["a"],"output_complete":false}'}}
        ]
    }
    result = parse_completion(data, "crop", 10)
    assert result.status == "truncated"


def test_invalid_json_is_schema_invalid():
    data = {"choices": [{"finish_reason": "stop", "message": {"content": "not json"}}]}
    assert parse_completion(data, "crop", 10).status == "schema_invalid"


def test_validate_payload_requires_bbox():
    bad = {"screen_kind": "slide", "regions": [{"text": "x", "bbox": [0, 0, 1]}]}
    assert validate_payload(bad, "full")
    good = {"screen_kind": "slide", "regions": [{"text": "x", "bbox": [0, 0, 1, 1]}]}
    assert validate_payload(good, "full") == ""


def test_loopback_only():
    from lecture_extract.adapters.vision.llama_server import VisionSetupError, _ensure_loopback

    _ensure_loopback("http://127.0.0.1:8080")
    with pytest.raises(VisionSetupError):
        _ensure_loopback("http://example.com:8080")
