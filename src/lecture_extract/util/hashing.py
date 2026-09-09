"""ハッシュ計算。入力・設定・モデルの同一性判定に使う (設計 12.1)。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CHUNK = 1024 * 1024


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_json(obj: Any) -> str:
    """辞書順・区切り固定の JSON 文字列。設定ハッシュの入力に使う。"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def config_hash(obj: Any) -> str:
    return sha256_text(stable_json(obj))


def sha256_file_cached(path: str | Path, cache_path: str | Path) -> str:
    """大きなモデルファイル用。サイズと更新時刻が同じならキャッシュを使う。"""
    import json as _json

    path = Path(path)
    cache_path = Path(cache_path)
    stat = path.stat()
    key = str(path.resolve())
    cache: dict[str, Any] = {}
    if cache_path.exists():
        try:
            cache = _json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}
    entry = cache.get(key)
    if entry and entry.get("size") == stat.st_size and entry.get("mtime_ns") == stat.st_mtime_ns:
        return entry["sha256"]
    digest = sha256_file(path)
    cache[key] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".partial")
    tmp.write_text(_json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(cache_path)
    return digest
