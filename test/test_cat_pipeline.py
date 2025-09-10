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
    CONVERT_POIS_VIT_ATTENTION,
    CONVERT_SCORE_VIT_ATTENTION,
    CONVERT_SCORE_VIT_CLS,
)
from content_aware_timelapse.viderator import frames_in_video

CURRENT_DIRECTORY = Path(__file__).parent.resolve()


@pytest.mark.parametrize("buffer_size", [0, 100])
@pytest.mark.parametrize("batch_size", [50, 100])
@pytest.mark.parametrize(
    "conversion_scoring_functions",
    [CONVERT_SCORE_VIT_CLS, CONVERT_SCORE_VIT_ATTENTION, CONVERT_CLIP],
)
def test_create_uncropped_timelapse(
    tmpdir: str,
    buffer_size: int,
    batch_size: int,
    conversion_scoring_functions: ConversionScoringFunctions,
) -> None:
    """
    Using a test asset, runs the main `create_timelapse` function and inspects the output.
    :param buffer_size: Passed to function.
    :param batch_size: Passed to function.
    :param tmpdir: Test fixture.
    :return: None
    """

    output_path = Path(tmpdir) / "output.mp4"
    vectors_path = Path(tmpdir) / "vectors.hdf5"

    duration = 30
    output_fps = 25

    cat_pipeline.create_uncropped_timelapse(
        input_files=[SAMPLE_TIMELAPSE_INPUT_PATH],
        output_path=output_path,
        duration=duration,
        output_fps=output_fps,
        batch_size=batch_size,
        conversion_scoring_functions=conversion_scoring_functions,
        vectors_path=vectors_path,
        plot_path=None,
        buffer_size=buffer_size,
    )

    video_frames = frames_in_video.frames_in_video_opencv(
        video_path=output_path,
    )

    assert output_path.exists()
    assert vectors_path.exists()

    assert video_frames.total_frame_count == duration * output_fps


def test_create_cropped_timelapse(
    tmpdir: str,
) -> None:
    """
    Using a test asset, runs the main `create_timelapse` function and inspects the output.
    :param buffer_size: Passed to function.
    :param batch_size: Passed to function.
    :param tmpdir: Test fixture.
    :return: None
    """

    output_path = Path(tmpdir) / "output.mp4"
    pois_vectors_path = Path(tmpdir) / "pios_vectors.hdf5"
    scores_vectors_path = Path(tmpdir) / "scores_vectors.hdf5"

    duration = 30
    output_fps = 25

    cat_pipeline.create_cropped_timelapse(
        input_files=[SAMPLE_TIMELAPSE_INPUT_PATH],
        output_path=output_path,
        duration=duration,
        output_fps=output_fps,
        batch_size=50,
        pois_vectors_path=pois_vectors_path,
        scores_vectors_path=scores_vectors_path,
        plot_path=None,
        buffer_size=100,
        conversion_pois_functions=CONVERT_POIS_VIT_ATTENTION,
        conversion_scoring_functions=CONVERT_SCORE_VIT_CLS,
    )

    video_frames = frames_in_video.frames_in_video_opencv(
        video_path=output_path,
    )

    assert output_path.exists()

    assert pois_vectors_path.exists()
    assert scores_vectors_path.exists()

    assert video_frames.total_frame_count == duration * output_fps


@pytest.mark.parametrize(
    "conversion_scoring_functions",
    [
        CONVERT_SCORE_VIT_CLS,
        CONVERT_SCORE_VIT_ATTENTION,
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

    cat_pipeline.create_uncropped_timelapse(
        input_files=[LONG_TEST_VIDEO_PATH],
        output_path=output_path,
        duration=duration,
        output_fps=output_fps,
        batch_size=100,
        conversion_scoring_functions=conversion_scoring_functions,
        vectors_path=None,
        plot_path=None,
        buffer_size=0,
    )

    video_frames = frames_in_video.frames_in_video_opencv(
        video_path=output_path,
    )

    assert output_path.exists()
    assert video_frames.total_frame_count == duration * output_fps
