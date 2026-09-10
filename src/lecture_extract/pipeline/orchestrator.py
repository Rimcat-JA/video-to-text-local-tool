"""処理順序と中断・再開の管理 (設計 11 / 12.2)。

初期の実行順序: 取り込み・時間軸確認 → 軽量画面走査 → 音声認識 → 画像認識 → 同期 → 文書出力。
画像認識中に音声モデルを常駐させず、GPU メモリを再利用する。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

from ..config import RunConfig
from ..db.store import Store
from ..util.gpulock import GpuLock
from ..util.hashing import config_hash
from .aligner import run_aligner
from .asr_run import run_asr
from .block_builder import run_block_builder
from .cache import prune_image_cache
from .exporter import Exporter
from .frame_scan import run_frame_scan
from .ingest import analysis_range, ingest, input_hash
from .probe import MediaProbe, probe_media
from .profiles import write_profiles
from .quality import completion_status, write_manifest, write_quality_report
from .screen_tracker import run_screen_tracker
from .speech_check import run_speech_check
from .vision_extract import run_vision

log = logging.getLogger(__name__)

STAGE_ORDER = [
    "ingest",
    "frame_scan",
    "asr",
    "vision",
    "screen_tracker",
    "align",
    "blocks",
    "export",
]

# あるステージを実行したら、下流のステージは作り直す。
STAGE_DOWNSTREAM = {
    "ingest": ["frame_scan", "asr", "vision", "screen_tracker", "align", "blocks", "export"],
    "frame_scan": ["vision", "screen_tracker", "align", "blocks", "export"],
    "asr": ["align", "blocks", "export"],
    "vision": ["screen_tracker", "align", "blocks", "export"],
    "screen_tracker": ["align", "blocks", "export"],
    "align": ["blocks", "export"],
    "blocks": ["export"],
    "export": [],
}

# 設定ハッシュを引くときのステージ名の対応。
_CONFIG_STAGE = {
    "ingest": "ingest",
    "frame_scan": "frame_scan",
    "asr": "asr",
    "vision": "vision",
    "screen_tracker": "vision",
    "align": "align",
    "blocks": "blocks",
    "export": "export",
}


class _NullLock:
    """ロックを取らない場合の入れ物。"""

    def __enter__(self) -> "_NullLock":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class Orchestrator:
    def __init__(
        self,
        cfg: RunConfig,
        *,
        vision_adapter=None,
        asr_adapter=None,
        resume: bool = True,
        only_stages: list[str] | None = None,
        force_stages: list[str] | None = None,
    ):
        self.cfg = cfg
        self.work_dir = Path(cfg.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.store = Store(self.work_dir / "state.sqlite")
        # 出力を作り直すときに同じ設定を使えるよう、作業フォルダへ残す。
        cfg.save(self.work_dir / "run_config.json")
        self.vision_adapter = vision_adapter
        self.asr_adapter = asr_adapter
        self.resume = resume
        self.only_stages = set(only_stages) if only_stages else None
        self.force_stages = set(force_stages or [])
        self.results: dict[str, Any] = {}
        self.media: dict[str, Any] | None = None
        self.probe: MediaProbe | None = None
        self.vision_info: dict[str, Any] = {"adapter": "not_run"}
        self.asr_info: dict[str, Any] = {"adapter": "not_run"}

    # ------------------------------------------------------------------ util
    def close(self) -> None:
        self.store.close()

    def _wants(self, stage: str) -> bool:
        return self.only_stages is None or stage in self.only_stages

    def _run_stage(
        self,
        stage: str,
        media_id: str | None,
        in_hash: str,
        fn: Callable[[str], Any],
    ) -> Any:
        cfg_hash = self.cfg.stage_config_hash(_CONFIG_STAGE[stage])
        forced = stage in self.force_stages
        if not self._wants(stage):
            log.info("[%s] 対象外のため実行しません。", stage)
            return None
        if not forced and self.resume and self.store.stage_is_done(media_id, stage, in_hash, cfg_hash):
            log.info("[%s] 入力・設定が同じで完了済みです。再解析しません。", stage)
            self.results[stage] = {"skipped": True}
            return None
        job_id = self.store.start_job(media_id, stage, in_hash, cfg_hash)
        started = time.time()
        try:
            result = fn(job_id)
        except Exception as exc:  # noqa: BLE001 - 失敗を記録してから再送出する
            self.store.finish_job(job_id, "failed", str(exc))
            raise
        elapsed = time.time() - started
        self.store.finish_job(job_id, "done")
        self.store.add_metric(media_id, stage, "elapsed_s", elapsed, "s")
        for downstream in STAGE_DOWNSTREAM.get(stage, []):
            self.store.invalidate_stage(media_id, downstream)
        log.info("[%s] 完了 (%.1fs)", stage, elapsed)
        self.results[stage] = result if isinstance(result, dict) else {"ok": True}
        return result

    # ------------------------------------------------------------------- run
    def run(self) -> dict[str, Any]:
        cfg = self.cfg
        store = self.store

        # --- ingest / timeline ---
        media_id = self._ingest()
        assert self.probe is not None and self.media is not None
        probe = self.probe
        start_us, end_us = analysis_range(cfg, probe.duration_us)
        in_hash = input_hash(cfg, self.media["sha256"])

        # --- frame_scan ---
        self._run_stage(
            "frame_scan",
            media_id,
            in_hash,
            lambda job_id: run_frame_scan(
                store, cfg, probe, media_id, start_us, end_us, job_id, resume=self.resume
            ),
        )

        # --- asr (先に実行し、終わったらモデルを解放する) ---
        if self._wants("asr"):
            self._run_asr(media_id, in_hash, probe, start_us, end_us)

        # --- vision ---
        if self._wants("vision"):
            self._run_vision(media_id, in_hash, probe)

        # 画像キャッシュを上限内へ収める (設計 3.3 / 12.3)。
        if self._wants("vision"):
            self.results["cache"] = prune_image_cache(store, cfg, media_id)

        # --- screen_tracker ---
        self._run_stage(
            "screen_tracker", media_id, in_hash, lambda job_id: run_screen_tracker(store, cfg, media_id)
        )

        # --- align ---
        self._run_stage("align", media_id, in_hash, lambda job_id: run_aligner(store, media_id))

        # 発話の専門語を画面文字と突き合わせ、確認済みかどうかを残す (設計 7.2)。
        if self._wants("align"):
            self.results['speech_check'] = run_speech_check(store, media_id)

        # --- blocks ---
        self._run_stage("blocks", media_id, in_hash, lambda job_id: run_block_builder(store, cfg, media_id))

        # --- export ---
        self._run_stage("export", media_id, in_hash, lambda job_id: self._export(media_id))

        status = completion_status(
            store, media_id, all_stages_done=self._all_core_stages_done(media_id, in_hash)
        )
        if self._wants("export"):
            write_quality_report(
                store,
                cfg,
                self.media,
                status,
                verification_scope=self._verification_scope(),
            )
            write_manifest(store, cfg, self.media, self.vision_info, self.asr_info, status, self.results)
        self.results["status"] = status
        return self.results

    # -------------------------------------------------------------- stages
    def _ingest(self) -> str:
        cfg = self.cfg
        store = self.store
        source = Path(cfg.input_path)
        stat = source.stat() if source.exists() else None
        quick_hash = config_hash(
            {
                "path": str(source.resolve()) if source.exists() else str(source),
                "size": stat.st_size if stat else None,
                "mtime_ns": stat.st_mtime_ns if stat else None,
                "range": [cfg.range_start_us, cfg.range_end_us],
            }
        )
        cfg_hash = cfg.stage_config_hash("ingest")
        job = store.get_job(None, "ingest")
        if (
            self.resume
            and "ingest" not in self.force_stages
            and job
            and job["status"] == "done"
            and job["input_hash"] == quick_hash
            and job["config_hash"] == cfg_hash
            and job["checkpoint"].get("media_id")
        ):
            media_id = job["checkpoint"]["media_id"]
            media = store.get_media(media_id)
            if media is not None:
                self.media = media
                self.probe = probe_media(cfg.input_path)
                log.info("[ingest] 既存の取り込み結果を使います (media_id=%s)", media_id)
                self.results["ingest"] = {"skipped": True, "media_id": media_id}
                return media_id

        job_id = store.start_job(None, "ingest", quick_hash, cfg_hash)
        try:
            media_id, probe = ingest(store, cfg)
        except Exception as exc:  # noqa: BLE001
            store.finish_job(job_id, "failed", str(exc))
            raise
        store.update_checkpoint(job_id, {"media_id": media_id})
        store.finish_job(job_id, "done")
        self.media = store.get_media(media_id)
        self.probe = probe
        self.results["ingest"] = {"media_id": media_id, "duration_us": probe.duration_us}
        return media_id

    def _run_asr(self, media_id: str, in_hash: str, probe: MediaProbe, start_us: int, end_us: int) -> None:
        adapter = self.asr_adapter or build_asr_adapter(self.cfg, self.work_dir)
        cfg_hash = self.cfg.stage_config_hash("asr")
        if (
            "asr" not in self.force_stages
            and self.resume
            and self.store.stage_is_done(media_id, "asr", in_hash, cfg_hash)
        ):
            log.info("[asr] 完了済みです。再解析しません。")
            self.results["asr"] = {"skipped": True}
            self.asr_info = adapter.describe() if hasattr(adapter, "describe") else {}
            return
        if hasattr(adapter, "ensure_ready"):
            adapter.ensure_ready()
        self.asr_info = adapter.describe()
        # CPU ビルドの ASR は GPU を使わないので、ロックを取らない。
        lock = GpuLock(self.work_dir / "gpu.lock") if self.cfg.asr.uses_gpu else _NullLock()
        try:
            with lock:
                self._run_stage(
                    "asr",
                    media_id,
                    in_hash,
                    lambda job_id: run_asr(
                        self.store,
                        self.cfg,
                        probe,
                        adapter,
                        media_id,
                        job_id,
                        start_us,
                        end_us,
                        resume=self.resume,
                    ),
                )
        finally:
            # 音声認識が終わったらモデルを解放する (設計 3.2)。
            adapter.close()

    def _run_vision(self, media_id: str, in_hash: str, probe: MediaProbe) -> None:
        adapter = self.vision_adapter or build_vision_adapter(self.cfg, self.work_dir)
        cfg_hash = self.cfg.stage_config_hash("vision")
        if (
            "vision" not in self.force_stages
            and self.resume
            and self.store.stage_is_done(media_id, "vision", in_hash, cfg_hash)
        ):
            log.info("[vision] 完了済みです。再解析しません。")
            self.results["vision"] = {"skipped": True}
            self.vision_info = adapter.describe()
            return
        if hasattr(adapter, "ensure_ready"):
            adapter.ensure_ready()
        self.vision_info = adapter.describe()
        lock = GpuLock(self.work_dir / "gpu.lock")
        try:
            with lock:
                self._run_stage(
                    "vision",
                    media_id,
                    in_hash,
                    lambda job_id: run_vision(
                        self.store,
                        self.cfg,
                        adapter,
                        media_id,
                        job_id,
                        probe.duration_us,
                        resume=self.resume,
                    ),
                )
        finally:
            adapter.close()

    def _export(self, media_id: str) -> dict[str, Any]:
        media = self.store.get_media(media_id)
        assert media is not None
        exporter = Exporter(self.store, self.cfg, media)
        written = exporter.export_all()
        # 用途別の派生物 (設計 1.2: 原本は上書きしない)
        written.update(write_profiles(self.store, self.cfg, media))
        return {"files": written}

    # ------------------------------------------------------------- reporting
    def _all_core_stages_done(self, media_id: str, in_hash: str) -> bool:
        for stage in STAGE_ORDER:
            if stage == "ingest":
                job = self.store.get_job(None, "ingest")
            else:
                job = self.store.get_job(media_id, stage)
            if not job or job["status"] != "done":
                return False
        return True

    def _verification_scope(self) -> str:
        if self.cfg.scan.mode == "fast":
            return (
                "高速モード。観測の間に出て消えた表示を取り逃がしている可能性があります。"
                "全文保全を検証済みとは表示しません（設計 5.3）。"
            )
        return (
            "保全モード。全フレームを復号して変化候補を拾い、候補を VLM で確認しました。"
            "全文が正しいことを人が確認した状態ではありません。"
        )


def build_vision_adapter(cfg: RunConfig, work_dir: Path):
    if cfg.vision.adapter == "stub":
        from ..adapters.vision.stub import ScriptedStubVision

        return ScriptedStubVision()
    from ..adapters.vision.llama_server import LlamaServerVision

    return LlamaServerVision(cfg.vision, work_dir)


def build_asr_adapter(cfg: RunConfig, work_dir: Path):
    if cfg.asr.adapter == "stub":
        from ..adapters.asr.stub import ScriptedStubAsr

        return ScriptedStubAsr(segments=[])
    from ..adapters.asr.whisper_cpp import WhisperCppAsr

    return WhisperCppAsr(cfg.asr, work_dir)
