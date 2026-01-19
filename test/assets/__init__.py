"""Uses pathlib to make referencing test assets by path easier."""

from pathlib import Path
from typing import List

_ASSETS_DIRECTORY = Path(__file__).parent.resolve()

ASSETS_DIRECTORY_PATH = str(_ASSETS_DIRECTORY)

SAMPLE_TIMELAPSE_INPUT_PATH = _ASSETS_DIRECTORY / "sample_timelapse_input.mp4"
SAMPLE_AUDIO_PATH = _ASSETS_DIRECTORY / "sample_audio.mp3"
LONG_TEST_VIDEO_PATH = _ASSETS_DIRECTORY / "long_video.mp4"

BORING_IMAGES_PATHS: List[Path] = [
    _ASSETS_DIRECTORY / "boring_1.jpg",
    _ASSETS_DIRECTORY / "boring_2.jpg",
]

MIDDLE_IMAGES_PATHS: List[Path] = [
    _ASSETS_DIRECTORY / "middle_1.jpg",
]

INTERESTING_IMAGES_PATHS: List[Path] = [
    _ASSETS_DIRECTORY / "interesting_1.jpg",
    _ASSETS_DIRECTORY / "interesting_2.jpg",
    _ASSETS_DIRECTORY / "interesting_3.jpg",
]

SORTED_BENCH_SCENES_PATHS: List[Path] = [
    _ASSETS_DIRECTORY / "bench_scene_out_of_focus.png",
    _ASSETS_DIRECTORY / "bench_scene_in_focus.png",
]

SORTED_STREAM_SOFTWARE_PATHS: List[Path] = [
    _ASSETS_DIRECTORY / "stream_software_obscured_1.png",
    _ASSETS_DIRECTORY / "stream_software.png",
]

IRL_VS_SOFTWARE_1: List[Path] = [
    _ASSETS_DIRECTORY / "stream_software_obscured_1.png",
    _ASSETS_DIRECTORY / "bench_scene_in_focus.png",
]

STREAM_SOFTWARE = _ASSETS_DIRECTORY / "stream_software.png"

IRL_VS_SOFTWARE_2: List[Path] = [
    STREAM_SOFTWARE,
    _ASSETS_DIRECTORY / "bench_scene_in_focus.png",
]

IRL_VS_SOFTWARE_3: List[Path] = [
    _ASSETS_DIRECTORY / "stream_cad_1.png",
    _ASSETS_DIRECTORY / "bench_scene_in_focus.png",
]

SORTED_STREAM_GENERIC_PATHS: List[Path] = [
    _ASSETS_DIRECTORY / "stream_software_obscured_1.png",
    STREAM_SOFTWARE,
    _ASSETS_DIRECTORY / "stream_cad_1.png",
    _ASSETS_DIRECTORY / "bench_scene_in_focus.png",
]
