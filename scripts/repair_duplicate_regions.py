"""保存済みの画面本文から、重複した領域を取り除き直す。

モデルが同じ範囲を複数の領域として繰り返し出力することがある。抽出時の
判定をあとから強化した場合に、モデルを再実行せずに既存の結果へ適用する。

原文は書き換えない。重複した領域を取り除き、取り除いた事実をフラグに残すだけ。

使い方:
    python scripts/repair_duplicate_regions.py <作業フォルダ> [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lecture_extract.db.store import Store  # noqa: E402
from lecture_extract.pipeline.vision_extract import (  # noqa: E402
    _drop_duplicate_regions,
    _flag_possible_duplicates,
    build_body_text,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="画面本文の重複領域を取り除き直す")
    parser.add_argument("work_dir", help="作業フォルダ (state.sqlite があるところ)")
    parser.add_argument("--dry-run", action="store_true", help="変更せず件数だけ表示する")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    store = Store(Path(args.work_dir) / "state.sqlite")
    try:
        media = store.get_media()
        if media is None:
            print("取り込み済みの動画がありません。")
            return 1
        referenced = {o.content_id for o in store.occurrences(media["id"]) if o.content_id}

        changed = 0
        removed_total = 0
        flagged_total = 0
        chars_before = 0
        chars_after = 0
        for content in store.all_contents():
            if content.id not in referenced:
                continue  # 過去の実行の残骸は触らない
            regions, dropped = _drop_duplicate_regions(list(content.regions))
            marked = _flag_possible_duplicates(regions)
            if not dropped and not marked:
                continue
            if marked:
                content.quality_flags = sorted(
                    set(content.quality_flags) | {"possible_duplicate_regions"}
                )
                flagged_total += 1
            chars_before += len(content.body_text)
            changed += 1
            removed_total += dropped
            if args.dry_run:
                chars_after += len(build_body_text(regions))
                continue
            content.regions = regions
            content.reading_order = [r.region_id for r in sorted(regions, key=lambda r: r.reading_order)]
            content.body_text = build_body_text(regions)
            if dropped:
                content.quality_flags = sorted(
                    set(content.quality_flags) | {"duplicate_regions_removed"}
                )
            chars_after += len(content.body_text)
            store.upsert_content(content)

        print(f"対象の画面本文: {len(referenced)} 件")
        print(f"重複を取り除いた本文: {changed} 件 / 取り除いた領域 {removed_total} 個")
        print(f"よく似た領域として印を付けた本文: {flagged_total} 件")
        if chars_before:
            print(
                f"本文の文字数: {chars_before} → {chars_after} "
                f"({(1 - chars_after / chars_before) * 100:.1f}% 減)"
            )
        if args.dry_run:
            print("（--dry-run のため保存していません）")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
