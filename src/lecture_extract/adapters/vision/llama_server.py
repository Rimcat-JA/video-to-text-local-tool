"""llama.cpp の llama-server を使う画像読み取りアダプター (設計 3.1 / 13.1)。

- 127.0.0.1 のループバック接続だけを使う。外部推論サービスへ切り替えない。
- モデル本体と mmproj を管理し、欠けていれば解析前に停止する。
- 出力上限に到達した応答や途中で切れた JSON を成功扱いにしない (設計 6.2)。
"""

from __future__ import annotations

import base64
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from ...config import VisionConfig
from ...util.hashing import config_hash, sha256_file_cached
from .base import STATUS_ERROR, STATUS_OK, STATUS_SCHEMA_INVALID, STATUS_TRUNCATED, VisionResult
from .prompts import PROMPT_VERSION, instruction_for, schema_for

log = logging.getLogger(__name__)

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}


class VisionSetupError(RuntimeError):
    pass


def _ensure_loopback(url: str) -> None:
    host = urlparse(url).hostname or ""
    if host not in LOOPBACK_HOSTS:
        raise VisionSetupError(
            f"推論サーバーの接続先がローカルではありません: {url}. "
            "設計 13.1 により、通常処理でローカル以外へ接続しません。"
        )


class LlamaServerVision:
    def __init__(self, cfg: VisionConfig, work_dir: str | Path):
        self.cfg = cfg
        if not cfg.model_alias:
            cfg.model_alias = Path(cfg.model_path).stem.lower() or "vlm"
        self.work_dir = Path(work_dir)
        self.hash_cache = self.work_dir / "model_hashes.json"
        _ensure_loopback(cfg.server_url)
        self.session = requests.Session()
        self._proc: subprocess.Popen | None = None
        self._describe_cache: dict[str, Any] | None = None

    # ---------------------------------------------------------------- server
    def ensure_ready(self) -> None:
        if self.health():
            log.info("llama-server は既に応答しています: %s", self.cfg.server_url)
            return
        if not self.cfg.manage_server:
            raise VisionSetupError(
                f"{self.cfg.server_url} に接続できません。llama-server を起動するか "
                "--manage-vision-server を付けて実行してください。"
            )
        self.start_server()

    def start_server(self) -> None:
        cfg = self.cfg
        model = Path(cfg.model_path)
        mmproj = Path(cfg.mmproj_path)
        missing = [str(p) for p in (model, mmproj) if not p.exists()]
        if missing:
            # 設計 13.1: 必要ファイルが欠けていれば解析前に不足を一覧表示して停止する。
            raise VisionSetupError(
                "画像モデルのファイルが不足しています。解析前に取得してください:\n  - "
                + "\n  - ".join(missing)
            )
        binary = shutil.which(cfg.server_binary) or cfg.server_binary
        port = urlparse(cfg.server_url).port or 8080
        cmd = [
            binary,
            "-m",
            str(model),
            "--mmproj",
            str(mmproj),
            "-c",
            str(cfg.n_ctx),
            "-ngl",
            str(cfg.n_gpu_layers),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--parallel",
            str(max(1, cfg.concurrency)),
            "--alias",
            cfg.model_alias,
        ]
        if cfg.image_min_tokens > 0:
            # Qwen-VL は画像トークンが少ないと読み取り精度が落ちるとされている。
            cmd += ["--image-min-tokens", str(cfg.image_min_tokens)]
        log_path = self.work_dir / "logs" / "llama-server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("llama-server を起動します: %s", " ".join(cmd))
        self._log_fh = open(log_path, "ab")
        self._proc = subprocess.Popen(cmd, stdout=self._log_fh, stderr=subprocess.STDOUT)
        deadline = time.time() + 300
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise VisionSetupError(
                    f"llama-server が終了しました (code={self._proc.returncode})。ログ: {log_path}"
                )
            if self.health():
                log.info("llama-server の準備ができました。")
                return
            time.sleep(1.0)
        self.close()
        raise VisionSetupError(f"llama-server が時間内に応答しませんでした。ログ: {log_path}")

    def health(self) -> bool:
        try:
            resp = self.session.get(f"{self.cfg.server_url}/health", timeout=3)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def close(self) -> None:
        """GPU メモリを解放する (設計 11: 画像認識中に音声モデルを常駐させない)。"""
        if self._proc is not None:
            log.info("llama-server を停止します。")
            self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:  # pragma: no cover
                self._proc.kill()
            self._proc = None
            try:
                self._log_fh.close()
            except Exception:  # noqa: BLE001
                pass
        self.session.close()

    # -------------------------------------------------------------- describe
    def describe(self) -> dict[str, Any]:
        if self._describe_cache is not None:
            return self._describe_cache
        info: dict[str, Any] = {
            "adapter": "llama_server",
            "model_alias": self.cfg.model_alias,
            "model_path": self.cfg.model_path,
            "mmproj_path": self.cfg.mmproj_path,
            "prompt_version": PROMPT_VERSION,
        }
        for key, path in (("model_sha256", self.cfg.model_path), ("mmproj_sha256", self.cfg.mmproj_path)):
            if path and Path(path).exists():
                info[key] = sha256_file_cached(path, self.hash_cache)
            else:
                info[key] = ""
        info["runtime_version"] = self._runtime_version()
        info["server_props"] = self._server_props()
        info["params"] = {
            "temperature": self.cfg.temperature,
            "seed": self.cfg.seed,
            "max_tokens": self.cfg.max_tokens,
            "crop_max_tokens": self.cfg.crop_max_tokens,
            "repeat_penalty": self.cfg.repeat_penalty,
            "n_ctx": self.cfg.n_ctx,
            "max_image_long_side": self.cfg.max_image_long_side,
            "image_min_tokens": self.cfg.image_min_tokens,
        }
        info["params_hash"] = config_hash(info["params"])
        info["model_revision"] = f"{self.cfg.model_alias}@{info['model_sha256'][:16]}"
        self._describe_cache = info
        return info

    def _runtime_version(self) -> str:
        binary = shutil.which(self.cfg.server_binary) or self.cfg.server_binary
        try:
            proc = subprocess.run(
                [binary, "--version"], capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace"
            )
            text = (proc.stdout or "") + (proc.stderr or "")
            for line in text.splitlines():
                if line.strip():
                    return line.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            return f"unknown ({exc})"
        return "unknown"

    def _server_props(self) -> dict[str, Any]:
        try:
            resp = self.session.get(f"{self.cfg.server_url}/props", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    k: data.get(k)
                    for k in ("model_path", "n_ctx", "build_info", "chat_template")
                    if k in data
                }
        except (requests.RequestException, ValueError):
            pass
        return {}

    # --------------------------------------------------------------- extract
    def extract(self, image_png: bytes, kind: str, *, extra_instruction: str = "") -> VisionResult:
        instruction = instruction_for(kind)
        if extra_instruction:
            instruction = f"{instruction}\n\n{extra_instruction}"
        schema = schema_for(kind)
        b64 = base64.b64encode(image_png).decode("ascii")
        payload = {
            "model": self.cfg.model_alias,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }
            ],
            "temperature": self.cfg.temperature,
            "seed": self.cfg.seed,
            # クロップは 1 領域分しか転記しないので上限を絞る。
            # 繰り返しループに入ったときの損失時間を抑える目的も兼ねる。
            "max_tokens": self.cfg.max_tokens if kind == "full" else self.cfg.crop_max_tokens,
            "repeat_penalty": self.cfg.repeat_penalty,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "screen_extraction", "schema": schema, "strict": True},
            },
        }
        started = time.time()
        try:
            resp = self.session.post(
                f"{self.cfg.server_url}/v1/chat/completions",
                json=payload,
                timeout=self.cfg.request_timeout_s,
            )
        except requests.RequestException as exc:
            return VisionResult(
                status=STATUS_ERROR,
                payload=None,
                raw_text="",
                error=f"request failed: {exc}",
                latency_ms=int((time.time() - started) * 1000),
            )
        latency_ms = int((time.time() - started) * 1000)
        if resp.status_code != 200:
            return VisionResult(
                status=STATUS_ERROR,
                payload=None,
                raw_text=resp.text[:20000],
                error=f"HTTP {resp.status_code}",
                latency_ms=latency_ms,
            )
        try:
            data = resp.json()
        except ValueError as exc:
            return VisionResult(
                status=STATUS_ERROR, payload=None, raw_text=resp.text[:20000], error=str(exc), latency_ms=latency_ms
            )
        return parse_completion(data, kind, latency_ms)


def parse_completion(data: dict[str, Any], kind: str, latency_ms: int) -> VisionResult:
    """OpenAI 互換応答を検証する。切り詰めや不正 JSON を成功扱いにしない。"""
    choices = data.get("choices") or []
    if not choices:
        return VisionResult(
            status=STATUS_ERROR,
            payload=None,
            raw_text=json.dumps(data, ensure_ascii=False)[:20000],
            error="choices が空です",
            latency_ms=latency_ms,
        )
    choice = choices[0]
    finish_reason = choice.get("finish_reason") or ""
    content = (choice.get("message") or {}).get("content") or ""
    usage = data.get("usage") or {}

    if finish_reason == "length":
        return VisionResult(
            status=STATUS_TRUNCATED,
            payload=None,
            raw_text=content,
            finish_reason=finish_reason,
            error="出力上限に到達しました",
            latency_ms=latency_ms,
            usage=usage,
        )
    try:
        parsed = json.loads(content)
    except (ValueError, TypeError) as exc:
        return VisionResult(
            status=STATUS_SCHEMA_INVALID,
            payload=None,
            raw_text=content,
            finish_reason=finish_reason,
            error=f"JSON として解釈できません: {exc}",
            latency_ms=latency_ms,
            usage=usage,
        )
    problem = validate_payload(parsed, kind)
    if problem:
        return VisionResult(
            status=STATUS_SCHEMA_INVALID,
            payload=parsed if isinstance(parsed, dict) else None,
            raw_text=content,
            finish_reason=finish_reason,
            error=problem,
            latency_ms=latency_ms,
            usage=usage,
        )
    if parsed.get("output_complete") is False:
        return VisionResult(
            status=STATUS_TRUNCATED,
            payload=parsed,
            raw_text=content,
            finish_reason=finish_reason,
            error="モデルが出力未完了と申告しました",
            latency_ms=latency_ms,
            usage=usage,
        )
    return VisionResult(
        status=STATUS_OK,
        payload=parsed,
        raw_text=content,
        finish_reason=finish_reason,
        latency_ms=latency_ms,
        usage=usage,
    )


def validate_payload(payload: Any, kind: str) -> str:
    """必要最小限のスキーマ検証。足りない場合は理由を返す。"""
    if not isinstance(payload, dict):
        return "オブジェクトではありません"
    if kind == "full":
        if not isinstance(payload.get("regions"), list):
            return "regions が配列ではありません"
        for i, region in enumerate(payload["regions"]):
            if not isinstance(region, dict):
                return f"regions[{i}] がオブジェクトではありません"
            if not isinstance(region.get("text"), str):
                return f"regions[{i}].text が文字列ではありません"
            bbox = region.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox)):
                return f"regions[{i}].bbox が 4 個の数値ではありません"
        if not isinstance(payload.get("screen_kind"), str):
            return "screen_kind がありません"
    else:
        lines = payload.get("lines")
        if not isinstance(lines, list) or not all(isinstance(v, str) for v in lines):
            return "lines が文字列配列ではありません"
    if "output_complete" in payload and not isinstance(payload["output_complete"], bool):
        return "output_complete が真偽値ではありません"
    return ""
