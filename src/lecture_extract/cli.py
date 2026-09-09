"""コマンドライン操作 (設計 3.1: 最初の操作方式は CLI)。"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

from .config import RunConfig
from .db.store import Store
from .util.logging_setup import setup_logging
from .util.timeutil import US, format_timestamp

log = logging.getLogger("lecture_extract")


def parse_time(value: str | None) -> int | None:
    """"12.5" / "1:23" / "0:01:23.456" を マイクロ秒へ変換する。"""
    if value is None or value == "":
        return None
    value = value.strip()
    if ":" not in value:
        return int(round(float(value) * US))
    parts = value.split(":")
    if len(parts) > 3:
        raise argparse.ArgumentTypeError(f"時刻の形式が不正です: {value}")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return int(round(seconds * US))


def build_config(args: argparse.Namespace) -> RunConfig:
    cfg = RunConfig.load(args.config) if getattr(args, "config", None) else RunConfig()
    if getattr(args, "input", None):
        cfg.input_path = str(Path(args.input).resolve())
    if getattr(args, "work", None):
        cfg.work_dir = str(Path(args.work).resolve())
    elif not getattr(args, "config", None):
        cfg.work_dir = str((Path(args.out).resolve() / "work") if getattr(args, "out", None) else Path("work").resolve())
    if getattr(args, "out", None):
        cfg.out_dir = str(Path(args.out).resolve())

    if getattr(args, "mode", None):
        cfg.scan.mode = args.mode
    if getattr(args, "fast_fps", None):
        cfg.scan.fast_fps = args.fast_fps
    if getattr(args, "start", None) is not None:
        cfg.range_start_us = parse_time(args.start)
    if getattr(args, "end", None) is not None:
        cfg.range_end_us = parse_time(args.end)

    if getattr(args, "vision_adapter", None):
        cfg.vision.adapter = args.vision_adapter
    if getattr(args, "vision_url", None):
        cfg.vision.server_url = args.vision_url
    if getattr(args, "vision_model", None):
        cfg.vision.model_path = str(Path(args.vision_model).resolve())
    if getattr(args, "mmproj", None):
        cfg.vision.mmproj_path = str(Path(args.mmproj).resolve())
    if getattr(args, "manage_vision_server", False):
        cfg.vision.manage_server = True
    if getattr(args, "n_ctx", None):
        cfg.vision.n_ctx = args.n_ctx
    if getattr(args, "ngl", None) is not None:
        cfg.vision.n_gpu_layers = args.ngl

    if getattr(args, "asr_adapter", None):
        cfg.asr.adapter = args.asr_adapter
    if getattr(args, "asr_model", None):
        cfg.asr.model_path = str(Path(args.asr_model).resolve())
    if getattr(args, "asr_binary", None):
        cfg.asr.binary = args.asr_binary
    if getattr(args, "language", None):
        cfg.asr.language = args.language
    if getattr(args, "channel", None):
        cfg.asr.channel = args.channel
    if getattr(args, "word_timestamps", False):
        cfg.asr.word_timestamps = True
    return cfg


# --------------------------------------------------------------------- run
def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline.orchestrator import Orchestrator

    cfg = build_config(args)
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_level, Path(cfg.work_dir) / "logs" / "run.log")
    if args.save_config:
        cfg.save(args.save_config)
        log.info("設定を保存しました: %s", args.save_config)

    orch = Orchestrator(
        cfg,
        resume=not args.no_resume,
        only_stages=args.only,
        force_stages=args.force,
    )
    try:
        results = orch.run()
    finally:
        orch.close()

    print()
    print("=== 実行結果 ===")
    for stage, result in results.items():
        if stage == "status":
            continue
        print(f"  {stage}: {json.dumps(result, ensure_ascii=False)[:300]}")
    print(f"  完了状態: {results.get('status')}")
    print(f"  出力先: {cfg.out_dir}")
    return 0


# ------------------------------------------------------------------- probe
def cmd_probe(args: argparse.Namespace) -> int:
    from .pipeline.probe import check_frame_timestamps, probe_media

    setup_logging(args.log_level)
    probe = probe_media(args.input)
    data = probe.to_dict()
    data["frame_timestamp_check"] = check_frame_timestamps(args.input)
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


# ------------------------------------------------------------------ doctor
def cmd_doctor(args: argparse.Namespace) -> int:
    """設計 13.1: 必要ファイルが欠けていれば、解析前に不足を一覧表示する。"""
    setup_logging(args.log_level)
    cfg = build_config(args)
    problems: list[str] = []
    ok: list[str] = []

    for name, env in (("ffmpeg", "LECTURE_EXTRACT_FFMPEG"), ("ffprobe", "LECTURE_EXTRACT_FFPROBE")):
        from .pipeline.ffmpeg_tools import ToolNotFound, resolve_tool

        try:
            ok.append(f"{name}: {resolve_tool(name, env)}")
        except ToolNotFound as exc:
            problems.append(str(exc))

    for module in ("av", "cv2", "numpy", "requests"):
        try:
            __import__(module)
            ok.append(f"python module {module}: 利用可能")
        except ImportError as exc:
            problems.append(f"python module {module} が読み込めません: {exc}")

    if cfg.vision.adapter == "llama_server":
        binary = shutil.which(cfg.vision.server_binary)
        if binary:
            ok.append(f"llama-server: {binary}")
        else:
            problems.append(
                f"llama-server が見つかりません ({cfg.vision.server_binary})。docs/SETUP.md を参照してください。"
            )
        for label, path in (("画像モデル", cfg.vision.model_path), ("mmproj", cfg.vision.mmproj_path)):
            if not path:
                problems.append(f"{label} のパスが未設定です (--vision-model / --mmproj)。")
            elif not Path(path).exists():
                problems.append(f"{label} が見つかりません: {path}")
            else:
                size_gb = Path(path).stat().st_size / 1024**3
                ok.append(f"{label}: {path} ({size_gb:.2f} GiB)")

    if cfg.asr.adapter == "whisper_cpp":
        from .adapters.asr.whisper_cpp import AsrSetupError, find_binary

        try:
            ok.append(f"whisper.cpp: {find_binary(cfg.asr.binary)}")
        except AsrSetupError as exc:
            problems.append(str(exc))
        if not cfg.asr.model_path:
            problems.append("音声モデルのパスが未設定です (--asr-model)。")
        elif not Path(cfg.asr.model_path).exists():
            problems.append(f"音声モデルが見つかりません: {cfg.asr.model_path}")
        else:
            size_gb = Path(cfg.asr.model_path).stat().st_size / 1024**3
            ok.append(f"音声モデル: {cfg.asr.model_path} ({size_gb:.2f} GiB)")

    print("=== 利用可能 ===")
    for item in ok:
        print(f"  OK  {item}")
    print()
    if problems:
        print("=== 不足 ===")
        for item in problems:
            print(f"  NG  {item}")
        print()
        print("不足があるため、この構成では解析を開始できません。")
        return 1
    print("不足はありません。")
    return 0


# ------------------------------------------------------------------ status
def cmd_status(args: argparse.Namespace) -> int:
    setup_logging(args.log_level)
    store = Store(Path(args.work) / "state.sqlite")
    try:
        media = store.get_media()
        if media is None:
            print("取り込み済みの動画がありません。")
            return 1
        print(f"元動画: {media['source_path']}")
        print(f"  sha256: {media['sha256']}")
        print(f"  再生時間: {format_timestamp(media['duration_us'])}")
        print()
        print("ステージ:")
        for job in store.jobs():
            checkpoint = json.dumps(job["checkpoint"], ensure_ascii=False)
            if len(checkpoint) > 120:
                checkpoint = checkpoint[:117] + "..."
            print(f"  {job['stage']:<15} {job['status']:<8} {checkpoint}")
        print()
        occurrences = store.occurrences(media["id"])
        utterances = store.utterances(media["id"])
        reviews = store.reviews(media["id"], status="open")
        print(f"表示期間: {len(occurrences)} 件 (本文抽出済み {sum(1 for o in occurrences if o.content_id)} 件)")
        print(f"発話: {len(utterances)} 件")
        print(f"未解決の確認項目: {len(reviews)} 件")
        return 0
    finally:
        store.close()


# ------------------------------------------------------------------ export
def cmd_export(args: argparse.Namespace) -> int:
    from .pipeline.exporter import Exporter
    from .pipeline.quality import completion_status, write_manifest, write_quality_report

    setup_logging(args.log_level)
    work = Path(args.work)
    store = Store(work / "state.sqlite")
    try:
        media = store.get_media()
        if media is None:
            print("取り込み済みの動画がありません。")
            return 1
        cfg = RunConfig()
        config_path = work / "run_config.json"
        if config_path.exists():
            cfg = RunConfig.load(config_path)
        cfg.work_dir = str(work)
        cfg.out_dir = str(Path(args.out).resolve())
        exporter = Exporter(store, cfg, media)
        written = exporter.export_all()
        status = completion_status(store, media["id"], all_stages_done=True)
        write_quality_report(store, cfg, media, status, verification_scope="出力の再生成のみ")
        write_manifest(store, cfg, media, {"adapter": "regenerated"}, {"adapter": "regenerated"}, status, {})
        for name, path in written.items():
            print(f"  {name}: {path}")
        return 0
    finally:
        store.close()


# ------------------------------------------------------------------ review
def cmd_review(args: argparse.Namespace) -> int:
    setup_logging(args.log_level)
    store = Store(Path(args.work) / "state.sqlite")
    try:
        media = store.get_media()
        if media is None:
            print("取り込み済みの動画がありません。")
            return 1
        if args.review_command == "list":
            items = store.reviews(media["id"], status=None if args.all else "open")
            for item in items:
                print(f"{item.id}  [{item.review_status}] {item.reason}  {item.target_kind}={item.target_ref}")
                if item.detail:
                    print(f"    {item.detail[:300]}")
            print(f"\n{len(items)} 件")
            return 0
        if args.review_command == "resolve":
            store.conn.execute(
                "UPDATE review_item SET review_status = ?, resolution = ? WHERE id = ?",
                (args.status, args.note, args.id),
            )
            print(f"{args.id} を {args.status} にしました。")
            return 0
    finally:
        store.close()
    return 1


# ------------------------------------------------------------------- cache
def cmd_cache(args: argparse.Namespace) -> int:
    """画像キャッシュの容量確認・整理・根拠画像の再生成 (設計 12.3)。"""
    from .pipeline.cache import cache_size_bytes, prune_image_cache, regenerate_frame

    setup_logging(args.log_level)
    work = Path(args.work)
    store = Store(work / "state.sqlite")
    try:
        media = store.get_media()
        if media is None:
            print("取り込み済みの動画がありません。")
            return 1
        config_path = work / "run_config.json"
        cfg = RunConfig.load(config_path) if config_path.exists() else RunConfig()
        cfg.work_dir = str(work)

        if args.restore:
            path = regenerate_frame(store, cfg, args.restore)
            print(f"再生成しました: {path}")
            return 0

        size = cache_size_bytes(cfg, media["id"])
        print(f"画像キャッシュ: {size / 1024**3:.3f} GiB / 上限 {cfg.scan.max_cache_bytes / 1024**3:.1f} GiB")
        if args.prune:
            stats = prune_image_cache(store, cfg, media["id"])
            print(
                f"  削除 {stats['removed']} 件 / 解放 {stats['freed_bytes'] / 1024**3:.3f} GiB"
                f" / 整理後 {stats['after_bytes'] / 1024**3:.3f} GiB"
            )
            print("  削除した画像は、元動画と時刻から `cache --restore <frame_id>` で作り直せます。")
        return 0
    finally:
        store.close()


# ----------------------------------------------------------------- correct
def cmd_correct(args: argparse.Namespace) -> int:
    """手修正を記録する。元の抽出結果は残したまま訂正版を作る (設計 9.1)。"""
    setup_logging(args.log_level)
    store = Store(Path(args.work) / "state.sqlite")
    try:
        content = store.get_content(args.content_id)
        if content is None:
            print(f"画面本文が見つかりません: {args.content_id}")
            return 1
        target = next((r for r in content.regions if r.region_id == args.region_id), None)
        if target is None:
            print(f"領域が見つかりません: {args.region_id}")
            return 1
        new_text = Path(args.text_file).read_text(encoding="utf-8") if args.text_file else args.text
        if new_text is None:
            print("--text か --text-file で訂正後の原文を指定してください。")
            return 1
        store.add_correction("region", f"{content.id}:{target.region_id}", "text", target.text, new_text)
        target.evidence["corrected_from"] = target.text
        target.text = new_text
        target.flags = sorted(set(target.flags) | {"user_corrected"})
        content.quality_flags = sorted(set(content.quality_flags) | {"user_corrected"})
        store.upsert_content(content)
        print("訂正を記録しました。元の抽出結果は correction テーブルに残っています。")
        return 0
    finally:
        store.close()


# -------------------------------------------------------------------- main
def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])


def _force_utf8_console() -> None:
    """Windows の既定コードページでも日本語が化けないようにする。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, OSError):
            pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8_console()
    parser = argparse.ArgumentParser(
        prog="lecture-extract",
        description="ローカル講義動画テキスト化システム（画面全文 + 発話、時刻同期）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="動画を解析して Markdown / JSONL / SRT を出力する")
    _add_common(p_run)
    p_run.add_argument("input", help="入力動画 (mp4 / mkv / webm など)")
    p_run.add_argument("-o", "--out", default="out", help="出力先フォルダ")
    p_run.add_argument("-w", "--work", default=None, help="作業フォルダ (既定: <out>/work)")
    p_run.add_argument("--config", default=None, help="設定 JSON")
    p_run.add_argument("--save-config", default=None, help="使用した設定を JSON に保存")
    p_run.add_argument("--mode", choices=["preserve", "fast"], default=None, help="解析モード")
    p_run.add_argument("--fast-fps", type=float, default=None)
    p_run.add_argument("--start", default=None, help="解析開始時刻 (例 0:10:00)")
    p_run.add_argument("--end", default=None, help="解析終了時刻")
    p_run.add_argument("--vision-adapter", choices=["llama_server", "stub"], default=None)
    p_run.add_argument("--vision-url", default=None, help="llama-server の URL (ループバックのみ)")
    p_run.add_argument("--vision-model", default=None, help="GGUF 画像モデル")
    p_run.add_argument("--mmproj", default=None, help="マルチモーダル投影ファイル")
    p_run.add_argument("--manage-vision-server", action="store_true", help="llama-server を起動・停止まで行う")
    p_run.add_argument("--n-ctx", type=int, default=None)
    p_run.add_argument("--ngl", type=int, default=None, help="GPU へ載せる層数")
    p_run.add_argument("--asr-adapter", choices=["whisper_cpp", "stub"], default=None)
    p_run.add_argument("--asr-model", default=None, help="whisper.cpp の ggml モデル")
    p_run.add_argument("--asr-binary", default=None, help="whisper-cli の場所")
    p_run.add_argument("--language", default=None, help="音声の言語 (既定 auto)")
    p_run.add_argument("--channel", default=None, help="mix / left / right / チャンネル番号")
    p_run.add_argument("--word-timestamps", action="store_true", help="単語時刻も保存する (experimental)")
    p_run.add_argument("--only", nargs="*", default=None, help="実行するステージを限定する")
    p_run.add_argument("--force", nargs="*", default=None, help="完了済みでも再実行するステージ")
    p_run.add_argument("--no-resume", action="store_true", help="チェックポイントを使わず最初から実行する")
    p_run.set_defaults(func=cmd_run)

    p_probe = sub.add_parser("probe", help="時刻基準とストリーム情報を確認する")
    _add_common(p_probe)
    p_probe.add_argument("input")
    p_probe.set_defaults(func=cmd_probe)

    p_doctor = sub.add_parser("doctor", help="必要なソフトウェアとモデルの不足を一覧表示する")
    _add_common(p_doctor)
    p_doctor.add_argument("--config", default=None)
    p_doctor.add_argument("--vision-adapter", choices=["llama_server", "stub"], default=None)
    p_doctor.add_argument("--vision-model", default=None)
    p_doctor.add_argument("--mmproj", default=None)
    p_doctor.add_argument("--asr-adapter", choices=["whisper_cpp", "stub"], default=None)
    p_doctor.add_argument("--asr-model", default=None)
    p_doctor.add_argument("--asr-binary", default=None)
    p_doctor.set_defaults(func=cmd_doctor)

    p_status = sub.add_parser("status", help="ステージの進捗と件数を表示する")
    _add_common(p_status)
    p_status.add_argument("-w", "--work", default="work")
    p_status.set_defaults(func=cmd_status)

    p_export = sub.add_parser("export", help="正本から出力を作り直す")
    _add_common(p_export)
    p_export.add_argument("-w", "--work", default="work")
    p_export.add_argument("-o", "--out", default="out")
    p_export.set_defaults(func=cmd_export)

    p_review = sub.add_parser("review", help="未確定箇所の確認")
    _add_common(p_review)
    p_review.add_argument("-w", "--work", default="work")
    review_sub = p_review.add_subparsers(dest="review_command", required=True)
    r_list = review_sub.add_parser("list")
    r_list.add_argument("--all", action="store_true", help="解決済みも表示する")
    r_resolve = review_sub.add_parser("resolve")
    r_resolve.add_argument("id")
    r_resolve.add_argument("--status", default="resolved", choices=["resolved", "wontfix", "open"])
    r_resolve.add_argument("--note", default="")
    p_review.set_defaults(func=cmd_review)

    p_cache = sub.add_parser("cache", help="画像キャッシュの容量確認・整理・根拠画像の再生成")
    _add_common(p_cache)
    p_cache.add_argument("-w", "--work", default="work")
    p_cache.add_argument("--prune", action="store_true", help="上限を超えた分を優先度の低い順に削除する")
    p_cache.add_argument("--restore", default=None, metavar="FRAME_ID", help="根拠画像を元動画から作り直す")
    p_cache.set_defaults(func=cmd_cache)

    p_correct = sub.add_parser("correct", help="領域の原文を手修正する（元の抽出結果は残す）")
    _add_common(p_correct)
    p_correct.add_argument("-w", "--work", default="work")
    p_correct.add_argument("content_id")
    p_correct.add_argument("region_id")
    p_correct.add_argument("--text", default=None)
    p_correct.add_argument("--text-file", default=None)
    p_correct.set_defaults(func=cmd_correct)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\n中断しました。次回は同じコマンドで再開できます。", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI では要点だけ表示する
        log.error("%s", exc)
        if args.log_level == "DEBUG":
            raise
        print(f"\nエラー: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
