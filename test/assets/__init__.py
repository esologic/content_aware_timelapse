"""Uses pathlib to make referencing test assets by path easier."""

from pathlib import Path
from typing import List

_ASSETS_DIRECTORY = Path(__file__).parent.resolve()

ASSETS_DIRECTORY_PATH = str(_ASSETS_DIRECTORY)

SAMPLE_TIMELAPSE_INPUT_PATH = _ASSETS_DIRECTORY / "sample_timelapse_input.mp4"
LONG_TEST_VIDEO_PATH = _ASSETS_DIRECTORY / "long_video.mp4"

BORING_IMAGES_PATHS: List[Path] = [
    _ASSETS_DIRECTORY / "boring_1.jpg",
    _ASSETS_DIRECTORY / "boring_2.jpg",
]

INTERESTING_IMAGES_PATHS: List[Path] = [
    _ASSETS_DIRECTORY / "interesting_1.jpg",
    _ASSETS_DIRECTORY / "interesting_2.jpg",
]
