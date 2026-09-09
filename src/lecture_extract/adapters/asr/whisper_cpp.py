"""whisper.cpp を使う音声認識アダプター (設計 3.1 / 7)。

- 元の言語で文字起こしし、発話単位の開始・終了時刻を保存する。
- 単語レベル時刻は公式 README で experimental とされているため、任意扱いにする (設計 7.3)。
- 実行ファイルとモデルが無い場合は、解析前に不足を示して停止する (設計 13.1)。
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from ...config import AsrConfig
from ...util.hashing import config_hash, sha256_file_cached
from .base import AsrResult, AsrSegment

log = logging.getLogger(__name__)

# whisper.cpp の実行ファイル名は版によって異なる。
BINARY_CANDIDATES = ["whisper-cli", "whisper-cli.exe", "main", "main.exe", "whisper", "whisper.exe"]


class AsrSetupError(RuntimeError):
    pass


def find_binary(configured: str) -> str:
    if configured:
        if Path(configured).exists():
            return str(Path(configured))
        found = shutil.which(configured)
        if found:
            return found
    for name in BINARY_CANDIDATES:
        found = shutil.which(name)
        if found:
            return found
    raise AsrSetupError(
        "whisper.cpp の実行ファイルが見つかりません。docs/SETUP.md の手順で導入し、"
        "--asr-binary で場所を指定してください。"
    )


class WhisperCppAsr:
    def __init__(self, cfg: AsrConfig, work_dir: str | Path):
        self.cfg = cfg
        self.work_dir = Path(work_dir)
        self.hash_cache = self.work_dir / "model_hashes.json"
        self._binary: str | None = None
        self._describe_cache: dict[str, Any] | None = None

    # --------------------------------------------------------------- setup
    def ensure_ready(self) -> None:
        self._binary = find_binary(self.cfg.binary)
        model = Path(self.cfg.model_path)
        if not self.cfg.model_path or not model.exists():
            raise AsrSetupError(
                "音声モデルが見つかりません。解析前に取得してください:\n  - "
                f"{self.cfg.model_path or '(未設定)'}\n"
                "取得手順は docs/SETUP.md を参照してください。"
            )

    @property
    def binary(self) -> str:
        if self._binary is None:
            self._binary = find_binary(self.cfg.binary)
        return self._binary

    def describe(self) -> dict[str, Any]:
        if self._describe_cache is not None:
            return self._describe_cache
        model_sha = ""
        if self.cfg.model_path and Path(self.cfg.model_path).exists():
            model_sha = sha256_file_cached(self.cfg.model_path, self.hash_cache)
        params = {
            "language": self.cfg.language,
            "word_timestamps": self.cfg.word_timestamps,
            "extra_args": list(self.cfg.extra_args),
            "initial_prompt": self.cfg.initial_prompt,
        }
        info = {
            "adapter": "whisper_cpp",
            "binary": self.binary,
            "model_path": self.cfg.model_path,
            "model_sha256": model_sha,
            "model_revision": f"{Path(self.cfg.model_path).name}@{model_sha[:16]}",
            "runtime_version": self._runtime_version(),
            "params": params,
            "params_hash": config_hash(params),
        }
        self._describe_cache = info
        return info

    def _runtime_version(self) -> str:
        try:
            proc = subprocess.run(
                [self.binary, "--help"],
                capture_output=True,
                text=True,
                timeout=30,
                encoding="utf-8",
                errors="replace",
            )
            text = (proc.stdout or "") + (proc.stderr or "")
            for line in text.splitlines():
                if "whisper" in line.lower() and ("usage" in line.lower() or "version" in line.lower()):
                    return line.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            return f"unknown ({exc})"
        return "whisper.cpp (version unreported)"

    # ----------------------------------------------------------- transcribe
    def transcribe(self, wav_path: str, *, out_prefix: str, chunk_start_us: int = 0) -> AsrResult:
        # chunk_start_us はログと再現用の情報。モデルには時刻を渡さない (設計 6.5 / 7)。
        threads = self.cfg.threads or max(1, (os.cpu_count() or 4) - 1)
        cmd = [
            self.binary,
            "-m",
            str(self.cfg.model_path),
            "-f",
            str(wav_path),
            "-of",
            str(out_prefix),
            "-t",
            str(threads),
            "--output-json-full" if self.cfg.word_timestamps else "--output-json",
        ]
        if self.cfg.language and self.cfg.language != "auto":
            cmd += ["-l", self.cfg.language]
        else:
            cmd += ["-l", "auto"]
        if self.cfg.initial_prompt:
            cmd += ["--prompt", self.cfg.initial_prompt]
        cmd += list(self.cfg.extra_args)

        started = time.time()
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        latency_ms = int((time.time() - started) * 1000)
        json_path = Path(f"{out_prefix}.json")
        if proc.returncode != 0 or not json_path.exists():
            return AsrResult(
                status="error",
                error=f"whisper.cpp が失敗しました (code={proc.returncode}): {(proc.stderr or '')[-2000:]}",
                latency_ms=latency_ms,
            )
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return AsrResult(status="error", error=f"出力 JSON を読めません: {exc}", latency_ms=latency_ms)
        segments, language = parse_whisper_json(data)
        return AsrResult(
            status="ok",
            segments=segments,
            raw_ref=str(json_path),
            latency_ms=latency_ms,
            language=language,
        )

    def close(self) -> None:
        return None


def parse_whisper_json(data: dict[str, Any]) -> tuple[list[AsrSegment], str]:
    """whisper.cpp の JSON 出力を共通形式へ変換する。"""
    language = ""
    result = data.get("result")
    if isinstance(result, dict):
        language = result.get("language", "") or ""
    params = data.get("params")
    if not language and isinstance(params, dict):
        language = params.get("language", "") or ""

    segments: list[AsrSegment] = []
    for item in data.get("transcription") or []:
        offsets = item.get("offsets") or {}
        start_ms = offsets.get("from")
        end_ms = offsets.get("to")
        if start_ms is None or end_ms is None:
            continue
        text = (item.get("text") or "").strip()
        if not text:
            continue
        tokens: list[dict[str, Any]] = []
        for tok in item.get("tokens") or []:
            tok_offsets = tok.get("offsets") or {}
            if tok_offsets.get("from") is None:
                continue
            tokens.append(
                {
                    "text": tok.get("text", ""),
                    "start_us": int(tok_offsets["from"]) * 1000,
                    "end_us": int(tok_offsets.get("to", tok_offsets["from"])) * 1000,
                    "experimental": True,  # 設計 7.3: 単語レベル時刻は experimental
                }
            )
        segments.append(
            AsrSegment(
                start_us=int(start_ms) * 1000,
                end_us=int(end_ms) * 1000,
                text=text,
                language=language,
                tokens=tokens,
            )
        )
    return segments, language
