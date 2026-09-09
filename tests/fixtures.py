"""検証用の合成講義動画を作る。

設計 14.1 の評価用データに相当する最小構成を、手元で再現できる形で用意する:
スライド、コード、1 文字変更、短時間表示、カーソル点滅、同一スライドの再登場、
境界をまたぐ発話、無音。
"""

from __future__ import annotations

import json
import math
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lecture_extract.adapters.vision.stub import encode_marker

WIDTH = 960
HEIGHT = 540
FPS = 20
SAMPLE_RATE = 16_000


@dataclass
class SceneSpec:
    marker: int
    duration_us: int
    lines: list[str]
    kind: str = "slide"  # 'slide' | 'code'
    title: str = ""
    caret: bool = False  # カーソル点滅を描く
    file_name: str = ""


@dataclass
class SpeechSpec:
    start_us: int
    end_us: int
    text: str


def render_scene(scene: SceneSpec, caret_on: bool) -> np.ndarray:
    bg = (28, 28, 30) if scene.kind == "code" else (250, 248, 245)
    fg = (235, 235, 235) if scene.kind == "code" else (20, 20, 20)
    img = np.full((HEIGHT, WIDTH, 3), bg, dtype=np.uint8)
    y = 120
    if scene.title:
        cv2.putText(img, scene.title, (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, fg, 2, cv2.LINE_AA)
    for line in scene.lines:
        cv2.putText(img, line, (60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, fg, 1, cv2.LINE_AA)
        y += 34
    if scene.caret and caret_on:
        # 細いキャレット。文字内容の変化ではない。
        cv2.rectangle(img, (60, y - 24), (63, y - 4), fg, -1)
    return encode_marker(img, scene.marker)


def make_audio(path: Path, speeches: list[SpeechSpec], total_us: int) -> None:
    n = int(total_us / 1_000_000 * SAMPLE_RATE)
    samples = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(7)
    for speech in speeches:
        s0 = max(0, int(speech.start_us / 1_000_000 * SAMPLE_RATE))
        s1 = min(n, int(speech.end_us / 1_000_000 * SAMPLE_RATE))
        if s1 <= s0:
            continue
        t = np.arange(s1 - s0) / SAMPLE_RATE
        tone = 0.22 * np.sin(2 * math.pi * 220 * t) + 0.05 * rng.standard_normal(s1 - s0)
        samples[s0:s1] += tone.astype(np.float32)
    pcm = np.clip(samples, -1.0, 1.0)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((pcm * 32767).astype(np.int16).tobytes())


def make_video(
    out_path: Path,
    scenes: list[SceneSpec],
    speeches: list[SpeechSpec] | None = None,
    *,
    fps: int = FPS,
    with_audio: bool = True,
) -> dict[str, Any]:
    """合成動画を作り、正解データ (状態の開始・終了時刻) を返す。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    speeches = speeches or []
    frame_us = 1_000_000 // fps

    truth: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    t = 0
    caret_period_frames = max(1, fps // 2)
    for scene in scenes:
        count = max(1, round(scene.duration_us / frame_us))
        start = t
        for i in range(count):
            caret_on = (i // caret_period_frames) % 2 == 0
            frames.append(render_scene(scene, caret_on))
            t += frame_us
        truth.append(
            {
                "marker": scene.marker,
                "start_us": start,
                "end_us": t,
                "kind": scene.kind,
                "lines": list(scene.lines),
                "title": scene.title,
                "file_name": scene.file_name,
            }
        )

    total_us = t
    audio_path = out_path.with_suffix(".wav")
    if with_audio:
        make_audio(audio_path, speeches, total_us)

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(fps),
        "-i",
        "-",
    ]
    if with_audio:
        cmd += ["-i", str(audio_path)]
    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "0",  # 輝度は可逆。1 文字変更の検証を圧縮で潰さない。
        "-pix_fmt",
        "yuv420p",
    ]
    if with_audio:
        cmd += ["-c:a", "aac", "-b:a", "96k", "-shortest"]
    cmd += [str(out_path)]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    stderr = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg に失敗しました: {stderr}")

    return {
        "path": str(out_path),
        "total_us": total_us,
        "fps": fps,
        "frame_us": frame_us,
        "states": truth,
        "speeches": [{"start_us": s.start_us, "end_us": s.end_us, "text": s.text} for s in speeches],
    }


def write_vision_script(path: Path, scenes: list[SceneSpec]) -> Path:
    """マーカー ID から抽出結果を引く台本を書く。"""
    script: dict[str, Any] = {}
    for scene in scenes:
        regions = []
        if scene.title:
            regions.append(
                {
                    "kind": "heading",
                    "role": "material_body",
                    "bbox": [0.05, 0.08, 0.95, 0.18],
                    "reading_order": 0,
                    "text": scene.title,
                }
            )
        body_kind = "code" if scene.kind == "code" else "paragraph"
        regions.append(
            {
                "kind": body_kind,
                "role": "material_body",
                "bbox": [0.05, 0.2, 0.95, 0.9],
                "reading_order": len(regions),
                "text": "\n".join(scene.lines),
                "stub_region_id": f"scene{scene.marker}",
            }
        )
        if scene.file_name:
            regions.append(
                {
                    "kind": "label",
                    "role": "material_context",
                    "bbox": [0.05, 0.02, 0.5, 0.06],
                    "reading_order": len(regions),
                    "text": scene.file_name,
                }
            )
        regions.append(
            {
                "kind": "ui",
                "role": "other_screen_text",
                "bbox": [0.8, 0.95, 1.0, 1.0],
                "reading_order": len(regions),
                "text": "File  Edit  View",
            }
        )
        script[str(scene.marker)] = {
            "screen_kind": "code_editor" if scene.kind == "code" else "slide",
            "context": {"title": scene.title, "file_name": scene.file_name},
            "regions": regions,
            "structure_notes": [],
            "output_complete": True,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def default_scenes() -> list[SceneSpec]:
    """評価用データの最小セット。"""
    code_v1 = [
        "def total(items):",
        "    result = 0",
        "    for item in items:",
        "        result += item",
        "    return result",
    ]
    code_v2 = list(code_v1)
    code_v2[1] = "    result = 1"  # 1 文字だけの変更
    return [
        SceneSpec(marker=11, duration_us=2_000_000, lines=["Agenda", "1. sum", "2. loops"], title="Lecture 1"),
        SceneSpec(
            marker=12,
            duration_us=2_500_000,
            lines=code_v1,
            kind="code",
            title="",
            caret=True,
            file_name="total.py",
        ),
        SceneSpec(
            marker=13,
            duration_us=2_500_000,
            lines=code_v2,
            kind="code",
            title="",
            caret=True,
            file_name="total.py",
        ),
        SceneSpec(marker=14, duration_us=100_000, lines=["popup: saved"], title="Notice"),
        SceneSpec(marker=11, duration_us=2_000_000, lines=["Agenda", "1. sum", "2. loops"], title="Lecture 1"),
    ]


def default_speeches() -> list[SpeechSpec]:
    return [
        SpeechSpec(500_000, 1_800_000, "きょうは合計を求める関数を書きます"),
        # 画面境界 (2.0s) をまたぐ発話
        SpeechSpec(1_900_000, 2_900_000, "ここでコードエディタに移ります"),
        SpeechSpec(3_200_000, 4_400_000, "初期値をゼロにしています"),
        SpeechSpec(5_000_000, 6_200_000, "初期値を一に変えるとどうなるでしょうか"),
        SpeechSpec(7_500_000, 8_500_000, "もう一度アジェンダに戻ります"),
    ]
