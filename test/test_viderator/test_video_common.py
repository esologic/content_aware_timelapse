"""
Tests of video interaction code.
"""

import subprocess
from itertools import tee
from pathlib import Path
from test.assets import LONG_TEST_VIDEO_PATH, SAMPLE_TIMELAPSE_INPUT_PATH
from test.test_viderator import viderator_test_common
from typing import Optional

import numpy as np
import pytest

from content_aware_timelapse import timeout_common
from content_aware_timelapse.viderator import video_common
from content_aware_timelapse.viderator.image_common import image_resolution
from content_aware_timelapse.viderator.viderator_types import ImageResolution


@pytest.mark.parametrize("video_path", [SAMPLE_TIMELAPSE_INPUT_PATH, LONG_TEST_VIDEO_PATH])
@pytest.mark.parametrize("video_fps", [None])
@pytest.mark.parametrize("_reduce_fps_to", [None, 25, 15])
@pytest.mark.parametrize(
    "width_height",
    [None],
)
@pytest.mark.parametrize("starting_frame", [None, 0, 30])
def test_frames_in_video(
    video_path: Path,
    video_fps: Optional[float],
    _reduce_fps_to: Optional[float],
    width_height: Optional[ImageResolution],
    starting_frame: Optional[int],
) -> None:
    """
    Compares the outputs of the canonical video -> frame source given a variety of arguments.
    There isn't really a ground truth check here, but the different methods should always produce
    the same result.

    :param video_path: Passed to functions.
    :param video_fps: Passed to functions.
    :param _reduce_fps_to: Passed to functions.
    :param width_height: Passed to functions.
    :param starting_frame: Passed to functions.
    :return: None
    """

    opencv_reader = video_common.frames_in_video_opencv(
        video_path=video_path,
        video_fps=video_fps,
        width_height=width_height,
        starting_frame=starting_frame,
    )

    ffmpeg_reader = video_common.frames_in_video_ffmpeg(
        video_path=video_path,
        video_fps=video_fps,
        width_height=width_height,
        starting_frame=starting_frame,
    )

    assert opencv_reader.original_fps == ffmpeg_reader.original_fps
    assert opencv_reader.original_resolution == ffmpeg_reader.original_resolution
    assert opencv_reader.total_frame_count == ffmpeg_reader.total_frame_count

    ffmpeg_frames, ffmpeg_time = timeout_common.measure_execution_time(list, ffmpeg_reader.frames)
    opencv_frames, opencv_time = timeout_common.measure_execution_time(list, opencv_reader.frames)

    print(f"ffmpeg time: {ffmpeg_time}, opencv time: {opencv_time}, video: {video_path.name}")

    assert len(opencv_frames) == len(ffmpeg_frames)

    for opencv_frame, ffmpeg_frame in zip(opencv_frames, ffmpeg_frames):
        assert np.array_equal(opencv_frame, ffmpeg_frame)

        if width_height is not None:
            assert image_resolution(opencv_frame) == width_height
            assert image_resolution(ffmpeg_frame) == width_height


def is_video_valid(path: Path) -> bool:
    """
    Uses ffprobe to see if a video can be opened or not.
    :param path: Path to video.
    :return: True if the video can be opened without error, False if otherwise.
    """

    try:
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_format", "-show_streams", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


@pytest.mark.parametrize(
    "input_image_resolution",
    [
        ImageResolution(width=10, height=10),
        ImageResolution(width=1920, height=1080),
    ],
)
@pytest.mark.parametrize("count", [10, 100, 1000])
@pytest.mark.parametrize("fps", [15, 30, 60])
@pytest.mark.parametrize("high_quality", [True, False])
def test_write_source_to_disk_consume(
    tmpdir: str, input_image_resolution: ImageResolution, count: int, fps: int, high_quality: bool
) -> None:
    """
    Test to make sure the write functionality works.
    :param tmpdir: Test fixture.
    :param input_image_resolution: Resolution of video to write.
    :param count: Number of frames.
    :param fps: FPS of output.
    :param high_quality: Passed to write function.
    :return: None
    """

    output_path = Path(tmpdir) / "output.mp4"

    input_frames, test_frames = tee(
        viderator_test_common.create_black_frames_iterator(
            image_resolution=input_image_resolution, count=count
        ),
        2,
    )

    video_common.write_source_to_disk_consume(
        source=input_frames,
        video_path=output_path,
        video_fps=fps,
        high_quality=high_quality,
    )

    assert is_video_valid(output_path)

    video_frames = video_common.frames_in_video_opencv(
        video_path=output_path,
    )

    assert video_frames.original_fps == fps
    assert video_frames.total_frame_count == count
    assert video_frames.original_resolution == input_image_resolution

    for input_frame, output_frame in zip(test_frames, video_frames.frames):
        assert np.array_equal(input_frame, output_frame)
