from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fixtures import (  # noqa: E402
    default_scenes,
    default_speeches,
    make_video,
    write_vision_script,
)


@pytest.fixture(scope="session")
def lecture_fixture(tmp_path_factory) -> dict:
    """合成講義動画と、それに対応する台本を一度だけ作る。"""
    root = tmp_path_factory.mktemp("lecture")
    scenes = default_scenes()
    speeches = default_speeches()
    truth = make_video(root / "lecture.mp4", scenes, speeches)
    truth["vision_script"] = str(write_vision_script(root / "vision_script.json", scenes))
    truth["root"] = str(root)
    truth["scenes"] = scenes
    truth["speech_specs"] = speeches
    return truth
