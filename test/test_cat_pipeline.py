"""
End-to-end tests of timelapse creation.
"""

from pathlib import Path
from test.assets import SAMPLE_TIMELAPSE_INPUT_PATH

from content_aware_timelapse import cat_pipeline
from content_aware_timelapse.viderator import video_common


def test_create_timelapse(tmpdir: Path) -> None:
    """
    Using a test asset, runs the main `create_timelapse` function and inspects the output.
    :param tmpdir: Test fixture.
    :return: None
    """

    output_path = tmpdir / "output.mp4"
    vectors_path = tmpdir / "vectors.hdf5"

    duration = 30
    output_fps = 25

    cat_pipeline.create_timelapse(
        input_files=[SAMPLE_TIMELAPSE_INPUT_PATH],
        output_path=output_path,
        duration=duration,
        output_fps=output_fps,
        batch_size=100,
        vectors_path=vectors_path,
    )

    video_frames = video_common.frames_in_video(
        video_path=output_path,
    )

    assert output_path.exists()
    assert vectors_path.exists()

    assert video_frames.total_frame_count == duration * output_fps
