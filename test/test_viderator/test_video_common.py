"""
Tests of video interaction code.
"""

import subprocess
from itertools import tee
from pathlib import Path
from test.test_viderator import viderator_test_common

import numpy as np
import pytest

from content_aware_timelapse.viderator import frames_in_video, video_common
from content_aware_timelapse.viderator.viderator_types import ImageResolution

VISUALIZATION_ENABLED = False


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

    video_frames = frames_in_video.frames_in_video_opencv(
        video_path=output_path,
    )

    assert video_frames.original_fps == fps
    assert video_frames.total_frame_count == count
    assert video_frames.original_resolution == input_image_resolution

    for input_frame, output_frame in zip(test_frames, video_frames.frames):
        assert np.array_equal(input_frame, output_frame)
