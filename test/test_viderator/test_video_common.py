"""
Tests of video interaction code.
"""

import shutil
import subprocess
from itertools import tee
from pathlib import Path
from test.assets import LONG_TEST_VIDEO_PATH, SAMPLE_TIMELAPSE_INPUT_PATH
from test.test_viderator import viderator_test_common
from typing import List

import numpy as np
import pytest

from assets import EASTERN_BOX_TURTLE_PATH
from content_aware_timelapse.viderator import frames_in_video, image_common, video_common
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


@pytest.mark.integration
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

    assert not output_path.exists()

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

    for frame_index, (input_frame, output_frame) in enumerate(
        zip(test_frames, video_frames.frames)
    ):
        assert np.array_equal(input_frame, output_frame), (
            f"Frame #{frame_index} not equal! Input Sum: {input_frame.sum()} "
            f"shape: {input_frame.shape}, Output Sum: {output_frame.sum()}, "
            f"shape: {output_frame.shape},"
        )


@pytest.mark.parametrize(
    "video_to_copy",
    [
        LONG_TEST_VIDEO_PATH,
        SAMPLE_TIMELAPSE_INPUT_PATH,
    ],
)
@pytest.mark.parametrize("num_copies", [1, 2, 3, 4])
def test_concat_videos_for_youtube(
    tmpdir: str,
    artifact_root: Path,
    video_to_copy: Path,
    num_copies: int,
) -> None:
    """
    Test to make sure video concatenation works by checking the output frame count.
    :param tmpdir: Test fixture.
    :param artifact_root: Test fixture.
    :param video_to_copy: Video that will be concatenated with itself.
    :param num_copies: Number of times to concatenate the video with itself.
    :return: None
    """

    input_frame_count = frames_in_video.frames_in_video_opencv(
        video_path=video_to_copy,
    ).total_frame_count

    output_path = artifact_root / f"youtube_concat_{num_copies}.mp4"

    tmp_dir = Path(tmpdir) / "concat_test_"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    video_paths: List[Path] = []

    try:
        for i in range(num_copies):
            tmp_path = tmp_dir / f"copy_{i}.mp4"
            shutil.copy(video_to_copy, tmp_path)
            video_paths.append(tmp_path)

        video_common.concat_videos_for_youtube(
            video_paths=tuple(video_paths),
            output_path=output_path,
        )

        output_frame_count = frames_in_video.frames_in_video_opencv(
            video_path=output_path,
        ).total_frame_count

        expected_frames = input_frame_count * num_copies

        assert output_frame_count == expected_frames

    finally:
        # Clean up temp files
        for p in video_paths:
            try:
                p.unlink()
            except FileNotFoundError:
                pass

        # Remove temp directory
        try:
            tmp_dir.rmdir()
        except OSError:
            pass


@pytest.mark.parametrize("high_quality", [True, False])
def test_write_source_to_disk_consume_turtle(artifact_root: Path, high_quality: bool) -> None:
    """
    Eyeball test using the turtle asset, where it should be obvious if the color space has been
    swapped or anything else terrible has happened.
    :param high_quality: Passed to write function.
    :return: None
    """

    output_path = Path(artifact_root) / "turtle_sense_check.mp4"
    output_path.unlink(missing_ok=True)

    num_frames = 30
    video_fps = 30

    video_common.write_source_to_disk_consume(
        source=(
            image_common.load_rgb_image(path=EASTERN_BOX_TURTLE_PATH) for _ in range(num_frames)
        ),
        video_path=output_path,
        video_fps=video_fps,
        high_quality=high_quality,
    )

    assert is_video_valid(output_path)

    video_frames = frames_in_video.frames_in_video_opencv(
        video_path=output_path,
    )

    assert video_frames.original_fps == video_fps
    assert video_frames.total_frame_count == num_frames
    assert video_frames.original_resolution == image_common.image_resolution(
        image_common.load_rgb_image(path=EASTERN_BOX_TURTLE_PATH)
    )
