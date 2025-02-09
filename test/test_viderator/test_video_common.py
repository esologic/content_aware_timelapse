"""
Tests of video interaction code.
"""

from pathlib import Path
from test.assets import LONG_TEST_VIDEO_PATH, SAMPLE_TIMELAPSE_INPUT_PATH
from typing import Optional

import numpy as np
import pytest

from content_aware_timelapse import timeout_common
from content_aware_timelapse.viderator import video_common
from content_aware_timelapse.viderator.image_common import ImageResolution, image_resolution


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
