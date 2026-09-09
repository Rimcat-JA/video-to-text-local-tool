"""FFmpeg / ffprobe の薄いラッパー。

外部通信は行わない。実行ファイルは PATH か環境変数で解決する (設計 13)。
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

log = logging.getLogger(__name__)


class ToolNotFound(RuntimeError):
    pass


class ToolFailed(RuntimeError):
    def __init__(self, cmd: Sequence[str], returncode: int, stderr: str):
        self.cmd = list(cmd)
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"command failed ({returncode}): {' '.join(map(str, cmd))}\n{stderr[-4000:]}")


def resolve_tool(name: str, env_var: str) -> str:
    override = os.environ.get(env_var)
    if override:
        if Path(override).exists() or shutil.which(override):
            return override
        raise ToolNotFound(f"{env_var}={override} が見つかりません")
    found = shutil.which(name)
    if not found:
        raise ToolNotFound(
            f"{name} が PATH にありません。環境変数 {env_var} で実行ファイルの場所を指定してください。"
        )
    return found


def ffmpeg_path() -> str:
    return resolve_tool("ffmpeg", "LECTURE_EXTRACT_FFMPEG")


def ffprobe_path() -> str:
    return resolve_tool("ffprobe", "LECTURE_EXTRACT_FFPROBE")


def run(cmd: Sequence[str], *, timeout: int | None = None, check: bool = True) -> subprocess.CompletedProcess:
    log.debug("run: %s", " ".join(map(str, cmd)))
    proc = subprocess.run(
        [str(c) for c in cmd],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        raise ToolFailed(cmd, proc.returncode, proc.stderr or "")
    return proc


def ffprobe_json(args: Sequence[str], *, timeout: int = 300) -> dict[str, Any]:
    cmd = [ffprobe_path(), "-v", "error", "-print_format", "json", *args]
    proc = run(cmd, timeout=timeout)
    return json.loads(proc.stdout or "{}")


def tool_version(path_getter) -> str:
    try:
        proc = run([path_getter(), "-version"], check=False, timeout=30)
        return (proc.stdout or "").splitlines()[0].strip()
    except Exception as exc:  # noqa: BLE001 - 版情報は取得できなくても致命的ではない
        return f"unknown ({exc})"
