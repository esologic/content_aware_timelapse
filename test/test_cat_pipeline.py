"""
End-to-end tests of timelapse creation.
"""

from datetime import datetime
from pathlib import Path
from test.assets import LONG_TEST_VIDEO_PATH, SAMPLE_TIMELAPSE_INPUT_PATH

import pytest

from content_aware_timelapse import cat_pipeline
from content_aware_timelapse.frames_to_vectors.conversion_types import ConversionScoringFunctions
from content_aware_timelapse.frames_to_vectors.vector_computation.compute_vectors_clip import (
    CONVERT_CLIP,
)
from content_aware_timelapse.frames_to_vectors.vector_computation.compute_vectors_vit import (
    CONVERT_VIT_ATTENTION,
    CONVERT_VIT_CLS,
)
from content_aware_timelapse.viderator import video_common

CURRENT_DIRECTORY = Path(__file__).parent.resolve()


@pytest.mark.parametrize(
    "conversion_scoring_functions", [CONVERT_VIT_CLS, CONVERT_VIT_ATTENTION, CONVERT_CLIP]
)
def test_create_timelapse(
    tmpdir: str, conversion_scoring_functions: ConversionScoringFunctions
) -> None:
    """
    Using a test asset, runs the main `create_timelapse` function and inspects the output.
    :param tmpdir: Test fixture.
    :return: None
    """

    output_path = Path(tmpdir) / "output.mp4"
    vectors_path = Path(tmpdir) / "vectors.hdf5"

    duration = 30
    output_fps = 25

    cat_pipeline.create_timelapse(
        input_files=[SAMPLE_TIMELAPSE_INPUT_PATH],
        output_path=output_path,
        duration=duration,
        output_fps=output_fps,
        batch_size=100,
        conversion_scoring_functions=conversion_scoring_functions,
        vectors_path=vectors_path,
        plot_path=None,
    )

    video_frames = video_common.frames_in_video_opencv(
        video_path=output_path,
    )

    assert output_path.exists()
    assert vectors_path.exists()

    assert video_frames.total_frame_count == duration * output_fps


@pytest.mark.parametrize(
    "conversion_scoring_functions",
    [
        CONVERT_VIT_CLS,
        CONVERT_VIT_ATTENTION,
    ],
)
def test_create_timelapses_output(conversion_scoring_functions: ConversionScoringFunctions) -> None:
    """
    Create timelapses using the supported conversion functions, and write the output to a local
    folder so it can be reviewed. Obviously we don't want to commit the resulting videos, but it's
    good to visually see the results.
    :return: None
    """

    output_path = (
        CURRENT_DIRECTORY
        / f"{conversion_scoring_functions.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    )

    duration = 30
    output_fps = 30

    cat_pipeline.create_timelapse(
        input_files=[LONG_TEST_VIDEO_PATH],
        output_path=output_path,
        duration=duration,
        output_fps=output_fps,
        batch_size=100,
        conversion_scoring_functions=conversion_scoring_functions,
        vectors_path=None,
        plot_path=None,
    )

    video_frames = video_common.frames_in_video_opencv(
        video_path=output_path,
    )

    assert output_path.exists()
    assert video_frames.total_frame_count == duration * output_fps
