"""段階 0 の動作確認: 画像 1 枚を llama-server へ渡し、抽出結果をそのまま表示する。

使い方:
    python scripts/try_vision.py path/to/screenshot.png
    python scripts/try_vision.py path/to/screenshot.png --url http://127.0.0.1:8080 --kind full

パイプラインを通さず、モデルの読み取り精度だけを確認するための道具。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lecture_extract.adapters.vision.llama_server import LlamaServerVision  # noqa: E402
from lecture_extract.config import VisionConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="llama-server に画像 1 枚を読ませて結果を表示する")
    parser.add_argument("image")
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--kind", choices=["full", "crop", "tile"], default="full")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--raw", action="store_true", help="解析前の原応答も表示する")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    cfg = VisionConfig(server_url=args.url, max_tokens=args.max_tokens)
    adapter = LlamaServerVision(cfg, work_dir=Path("tmp"))
    if not adapter.health():
        print(f"{args.url} に接続できません。llama-server を起動してから実行してください。")
        return 1

    image = Path(args.image).read_bytes()
    result = adapter.extract(image, args.kind)
    print(f"status        : {result.status}")
    print(f"finish_reason : {result.finish_reason}")
    print(f"latency       : {result.latency_ms} ms")
    if result.error:
        print(f"error         : {result.error}")
    print()
    if args.raw or result.payload is None:
        print("--- 原応答 ---")
        print(result.raw_text[:20000])
        print()
    if result.payload is None:
        return 2

    payload = result.payload
    if args.kind == "full":
        print(f"screen_kind   : {payload.get('screen_kind')}")
        print(f"context       : {json.dumps(payload.get('context') or {}, ensure_ascii=False)}")
        print()
        for region in sorted(payload.get("regions") or [], key=lambda r: r.get("reading_order", 0)):
            print(f"[{region.get('reading_order')}] kind={region.get('kind')} role={region.get('role')}")
            print(f"    bbox={region.get('bbox')} flags={region.get('flags') or []}")
            if region.get("unreadable"):
                print(f"    判読不能: {json.dumps(region['unreadable'], ensure_ascii=False)}")
            if region.get("candidates"):
                print(f"    候補(原文ではない): {region['candidates']}")
            text = region.get("text", "")
            for line in text.split("\n"):
                print(f"    | {line}")
            print()
    else:
        for line in payload.get("lines") or []:
            print(f"| {line}")
        print()
        print(f"flags: {payload.get('flags') or []}")
    print(f"output_complete: {payload.get('output_complete')}")
    adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
