"""同一 GPU への要求をロックする (設計 12.2)。

重い推論は GPU 1 枚につき 1 件。予期しない同時推論を避けるため、
プロセスをまたいで使えるファイルロックにする。
"""

from __future__ import annotations

import errno
import logging
import os
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)


def _process_alive(pid: int) -> bool:
    """プロセスが生きているか。

    Windows の os.kill は signal 0 でも TerminateProcess を呼ぶため、生存確認には
    使えない (相手を終了させてしまう)。OS ごとに安全な方法で確認する。
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        SYNCHRONIZE = 0x00100000
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        WAIT_OBJECT_0 = 0x0
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(
            SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, False, wintypes.DWORD(pid)
        )
        if not handle:
            return False  # 開けない = 既に存在しない
        try:
            # 終了済みのプロセスハンドルはシグナル状態になる。
            return kernel32.WaitForSingleObject(handle, 0) != WAIT_OBJECT_0
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


class GpuLock:
    def __init__(self, path: str | Path, *, timeout_s: float = 3600.0, poll_s: float = 1.0):
        self.path = Path(path)
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self._fd: int | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + self.timeout_s
        warned = False
        while True:
            try:
                self._fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(self._fd, str(os.getpid()).encode("ascii"))
                return
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                if self._holder_is_dead():
                    log.warning("残っていた GPU ロックを回収します: %s", self.path)
                    self.path.unlink(missing_ok=True)
                    continue
                if not warned:
                    log.info("他の推論が GPU を使用中です。待機します: %s", self.path)
                    warned = True
                if time.time() > deadline:
                    raise TimeoutError(f"GPU ロックを取得できませんでした: {self.path}")
                time.sleep(self.poll_s)

    def _holder_is_dead(self) -> bool:
        try:
            pid = int(self.path.read_text(encoding="ascii").strip() or "0")
        except (OSError, ValueError):
            return False
        if pid <= 0:
            return False
        if pid == os.getpid():
            return False
        return not _process_alive(pid)

    def release(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            finally:
                self._fd = None
        self.path.unlink(missing_ok=True)

    def __enter__(self) -> "GpuLock":
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
