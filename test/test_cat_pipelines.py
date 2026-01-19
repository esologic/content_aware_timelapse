"""
End-to-end tests of timelapse creation.
"""

from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from test.assets import LONG_TEST_VIDEO_PATH, SAMPLE_AUDIO_PATH, SAMPLE_TIMELAPSE_INPUT_PATH
from test.test_viderator import viderator_test_common
from typing import List, NamedTuple, Optional

import pytest

from content_aware_timelapse import cat_pipelines
from content_aware_timelapse.frames_to_vectors.conversion_types import ConversionScoringFunctions
from content_aware_timelapse.frames_to_vectors.vector_computation.compute_vectors_clip import (
    CONVERT_CLIP,
)
from content_aware_timelapse.frames_to_vectors.vector_computation.compute_vectors_vit import (
    CONVERT_POIS_VIT_ATTENTION,
    CONVERT_SCORE_VIT_ATTENTION,
    CONVERT_SCORE_VIT_CLS,
)
from content_aware_timelapse.gpu_discovery import discover_gpus
from content_aware_timelapse.viderator import frames_in_video, video_common
from content_aware_timelapse.viderator.image_common import image_resolution, load_rgb_image
from content_aware_timelapse.viderator.viderator_types import AspectRatio, ImageResolution


@pytest.mark.integration
@pytest.mark.parametrize("buffer_size", [0, 100])
@pytest.mark.parametrize("batch_size", [50, 100])
@pytest.mark.parametrize(
    "conversion_scoring_functions",
    [CONVERT_SCORE_VIT_CLS, CONVERT_SCORE_VIT_ATTENTION, CONVERT_CLIP],
)
def test_create_timelapse_score(
    tmpdir: str,
    buffer_size: int,
    batch_size: int,
    conversion_scoring_functions: ConversionScoringFunctions,
    artifact_root: Path,
) -> None:
    """
    Using a test asset, runs the main `create_timelapse` function and inspects the output.
    :param buffer_size: Passed to function.
    :param batch_size: Passed to function.
    :param tmpdir: Test fixture.
    :param artifact_root: Test fixture that provides an optionally persisted directory to write
    test assets to.
    :return: None
    """

    output_path = Path(tmpdir) / "output.mp4"
    vectors_path = Path(tmpdir) / "vectors.hdf5"

    duration = 30
    output_fps = 25

    cat_pipelines.create_timelapse_score(
        input_files=[SAMPLE_TIMELAPSE_INPUT_PATH],
        output_path=output_path,
        duration=duration,
        output_fps=output_fps,
        output_resolution=None,
        resize_inputs=False,
        batch_size=batch_size,
        conversion_scoring_functions=conversion_scoring_functions,
        vectors_path=vectors_path,
        plot_path=None,
        buffer_size=buffer_size,
        deselection_radius_frames=10,
        audio_paths=[SAMPLE_AUDIO_PATH],
        gpus=discover_gpus(),
        best_frame_path=artifact_root / "best_frame.png",
    )

    video_frames = frames_in_video.frames_in_video_opencv(
        video_path=output_path,
    )

    assert output_path.exists()
    assert vectors_path.exists()

    assert video_frames.total_frame_count == duration * output_fps

    output_path.unlink(missing_ok=True)
    vectors_path.unlink(missing_ok=True)


@pytest.mark.integration
@pytest.mark.parametrize("best_frame_enabled", [True, False])
@pytest.mark.parametrize(
    "conversion_scoring_functions",
    [
        CONVERT_SCORE_VIT_CLS,
        CONVERT_SCORE_VIT_ATTENTION,
    ],
)
def test_create_timelapse_score_output(
    conversion_scoring_functions: ConversionScoringFunctions,
    best_frame_enabled: bool,
    artifact_root: Path,
) -> None:
    """
    Create timelapses using the supported conversion functions, and write the output to a local
    folder so it can be reviewed. Obviously we don't want to commit the resulting videos, but it's
    good to visually see the results.
    :param artifact_root: Test fixture that provides an optionally persisted directory to write
    test assets to.
    :return: None
    """

    test_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = artifact_root / f"{conversion_scoring_functions.name}_{test_time}.mp4"

    best_frame_path: Optional[Path] = (
        (artifact_root / f"{conversion_scoring_functions.name}_best_frame_{test_time}.png")
        if best_frame_enabled
        else None
    )

    duration = 30
    output_fps = 30

    cat_pipelines.create_timelapse_score(
        input_files=[LONG_TEST_VIDEO_PATH],
        output_path=output_path,
        duration=duration,
        output_fps=output_fps,
        output_resolution=None,
        resize_inputs=False,
        batch_size=100,
        conversion_scoring_functions=conversion_scoring_functions,
        vectors_path=None,
        plot_path=None,
        buffer_size=0,
        deselection_radius_frames=10,
        audio_paths=[SAMPLE_AUDIO_PATH],
        gpus=discover_gpus(),
        best_frame_path=best_frame_path,
    )

    video_frames = frames_in_video.frames_in_video_opencv(
        video_path=output_path,
    )

    assert output_path.exists()
    assert video_frames.total_frame_count == duration * output_fps

    if best_frame_path is not None:
        assert best_frame_path.exists()
        assert load_rgb_image(path=best_frame_path) is not None

    output_path.unlink(missing_ok=True)


@pytest.mark.integration
def test_create_timelapse_crop_score(
    tmpdir: str,
) -> None:
    """
    Using a test asset, runs the main `create_timelapse` function and inspects the output.
    :param tmpdir: Test fixture.
    :return: None
    """

    output_path = Path(tmpdir) / "output.mp4"
    pois_vectors_path = Path(tmpdir) / "pios_vectors.hdf5"
    scores_vectors_path = Path(tmpdir) / "scores_vectors.hdf5"

    duration = 30
    output_fps = 25

    cat_pipelines.create_timelapse_crop_score(
        input_files=[SAMPLE_TIMELAPSE_INPUT_PATH],
        output_path=output_path,
        duration=duration,
        output_fps=output_fps,
        output_resolution=None,
        resize_inputs=False,
        batch_size_pois=100,
        batch_size_scores=300,
        aspect_ratio=AspectRatio(4, 3),
        scoring_deselection_radius_frames=10,
        pois_vectors_path=pois_vectors_path,
        scores_vectors_path=scores_vectors_path,
        plot_path=None,
        scaled_frames_buffer_size=10000,
        conversion_pois_functions=CONVERT_POIS_VIT_ATTENTION,
        conversion_scoring_functions=CONVERT_SCORE_VIT_CLS,
        audio_paths=[SAMPLE_AUDIO_PATH],
        gpus=discover_gpus(),
        layout_matrix=[[0]],
    )

    video_frames = frames_in_video.frames_in_video_opencv(
        video_path=output_path,
    )

    assert output_path.exists()

    assert pois_vectors_path.exists()
    assert scores_vectors_path.exists()

    assert video_frames.total_frame_count == duration * output_fps

    output_path.unlink(missing_ok=True)
    pois_vectors_path.unlink(missing_ok=True)
    scores_vectors_path.unlink(missing_ok=True)


class ResolutionFrameCount(NamedTuple):
    """
    Intermediate type for the test.
    """

    resolution: ImageResolution
    frame_count: int


@pytest.mark.parametrize(
    "video_params, resize_inputs, exception_expected",
    [
        # Two different resolutions
        (
            [
                ResolutionFrameCount(resolution=ImageResolution(1920, 1080), frame_count=10),
                ResolutionFrameCount(resolution=ImageResolution(1280, 720), frame_count=5),
            ],
            True,
            False,
        ),
        # The Same Resolution
        (
            [
                ResolutionFrameCount(resolution=ImageResolution(1280, 720), frame_count=10),
                ResolutionFrameCount(resolution=ImageResolution(1280, 720), frame_count=5),
            ],
            True,
            False,
        ),
        # The Two different resolutions, but no resize flag
        (
            [
                ResolutionFrameCount(resolution=ImageResolution(1920, 1080), frame_count=10),
                ResolutionFrameCount(resolution=ImageResolution(1280, 720), frame_count=5),
            ],
            False,
            True,
        ),
        # Two resolutions, bad aspect ratio, should raise.
        (
            [
                ResolutionFrameCount(resolution=ImageResolution(1920, 1080), frame_count=10),
                ResolutionFrameCount(resolution=ImageResolution(1280, 730), frame_count=5),
            ],
            True,
            True,
        ),
    ],
)
def test_load_input_videos(
    video_params: List[ResolutionFrameCount], resize_inputs: bool, exception_expected: bool
) -> None:
    """
    Test a few things about the `load_input_videos` function, mainly the part about resizing the
    inputs.
    :param video_params: Test creates videos for reading.
    :param resize_inputs: Function flag.
    :return: None
    """

    # Because load_input_video expects paths to videos, we need to write some video files to load
    # as a first step.

    with ExitStack() as stack:

        video_paths = [
            stack.enter_context(video_common.video_safe_temp_path())
            for _ in range(len(video_params))
        ]

        for video_path, frames in zip(
            video_paths,
            [
                viderator_test_common.create_black_frames_iterator(
                    image_resolution=video_param.resolution, count=video_param.frame_count
                )
                for video_param in video_params
            ],
        ):
            video_common.write_source_to_disk_consume(
                source=frames, video_path=video_path, video_fps=1
            )

        # Start of the actual test logic here.

        if exception_expected:
            with pytest.raises(Exception):
                cat_pipelines.load_input_videos(
                    input_files=video_paths,
                    resize_inputs=resize_inputs,
                )
        else:

            combined_videos = cat_pipelines.load_input_videos(
                input_files=video_paths,
                resize_inputs=resize_inputs,
            )

            assert combined_videos.total_frame_count == sum(
                video_param.frame_count for video_param in video_params
            )

            input_frames = list(combined_videos.frames)

            assert combined_videos.total_frame_count == len(input_frames)

            assert combined_videos.original_resolution == min(
                (video_param.resolution for video_param in video_params),
                key=lambda resolution: resolution.width * resolution.height,
            )

            for frame in input_frames:
                assert combined_videos.original_resolution == image_resolution(frame)
