"""同一 GPU への要求をロックする (設計 12.2)。

重い推論は GPU 1 枚につき 1 件。予期しない同時推論を避けるため、
プロセスをまたいで使えるファイルロックにする。
"""

from __future__ import annotations

import errno
import logging
import os
import time
from pathlib import Path

log = logging.getLogger(__name__)


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
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        except OSError:
            return False
        return False

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
