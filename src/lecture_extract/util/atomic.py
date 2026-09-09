"""一時書き込みと確定を分ける (設計 12.2)。途中ファイルを正本にしない。"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def atomic_write(path: str | Path, mode: str = "w", encoding: str | None = "utf-8") -> Iterator:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 一時ファイル名はプロセスごとに分ける（並行実行時の衝突を避ける）。
    tmp = path.with_name(f"{path.name}.{os.getpid()}.partial")
    kwargs = {} if "b" in mode else {"encoding": encoding, "newline": "\n"}
    fh = open(tmp, mode, **kwargs)
    try:
        yield fh
        fh.flush()
        os.fsync(fh.fileno())
    except BaseException:
        fh.close()
        tmp.unlink(missing_ok=True)
        raise
    else:
        fh.close()
        os.replace(tmp, path)


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    with atomic_write(path, "wb", encoding=None) as fh:
        fh.write(data)
    return Path(path)


def atomic_write_text(path: str | Path, text: str) -> Path:
    with atomic_write(path, "w") as fh:
        fh.write(text)
    return Path(path)
