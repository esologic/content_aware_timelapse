"""
End -> End Testing of the vector computation and sorting.
"""

import itertools
from functools import partial
from test.assets import (
    IRL_VS_SOFTWARE_1,
    IRL_VS_SOFTWARE_2,
    IRL_VS_SOFTWARE_3,
    SAMPLE_TIMELAPSE_INPUT_PATH,
    SORTED_BENCH_SCENES_PATHS,
    SORTED_STREAM_GENERIC_PATHS,
    SORTED_STREAM_SOFTWARE_PATHS,
)
from typing import Iterator, List

import more_itertools
import pytest

import content_aware_timelapse.frames_to_vectors.conversion
import content_aware_timelapse.frames_to_vectors.vector_computation.compute_vectors_vit
from content_aware_timelapse.frames_to_vectors import vector_scoring
from content_aware_timelapse.frames_to_vectors.conversion_types import (
    ConversionScoringFunctions,
    IndexScores,
)
from content_aware_timelapse.frames_to_vectors.vector_computation.compute_vectors_vit import (
    CONVERT_VIT_ATTENTION,
    CONVERT_VIT_CLS,
)
from content_aware_timelapse.viderator import frames_in_video, image_common, video_common
from content_aware_timelapse.viderator.viderator_types import ImageSourceType


@pytest.mark.parametrize(
    "conversion_scoring_functions",
    [
        # CONVERT_CLIP, # CLIP doesn't totally work as expected yet...
        CONVERT_VIT_CLS,
        CONVERT_VIT_ATTENTION,
    ],
)
@pytest.mark.parametrize(
    "frames",
    [
        list(map(image_common.load_rgb_image, SORTED_BENCH_SCENES_PATHS)),
        list(map(image_common.load_rgb_image, SORTED_STREAM_SOFTWARE_PATHS)),
        list(map(image_common.load_rgb_image, IRL_VS_SOFTWARE_1)),
        list(map(image_common.load_rgb_image, IRL_VS_SOFTWARE_2)),
        list(map(image_common.load_rgb_image, IRL_VS_SOFTWARE_3)),
        list(map(image_common.load_rgb_image, SORTED_STREAM_GENERIC_PATHS)),
    ],
)
def test_sorting_converted_vectors(
    conversion_scoring_functions: ConversionScoringFunctions, frames: ImageSourceType
) -> None:
    """
    Sanity check using library assets to assume that scoring and sorting images works as expected.
    :param frames: Should be sorted in order of interesting-ness, with the most interesting frame
    first.
    :return: None
    """

    vectors = content_aware_timelapse.frames_to_vectors.conversion.frames_to_vectors(
        frames=frames,
        input_signature="test signature",
        batch_size=1,
        total_input_frames=1,
        convert_batches=conversion_scoring_functions.conversion,
        intermediate_path=None,
    )

    score_indexes: List[IndexScores] = list(
        map(
            partial(
                conversion_scoring_functions.scoring,
                original_source_resolution=image_common.image_resolution(next(iter(frames))),
            ),
            enumerate(vectors),
        )
    )

    indices = vector_scoring._score_and_sort_frames(  # pylint: disable=protected-access
        score_weights=conversion_scoring_functions.weights,
        index_scores=score_indexes,
        num_output_frames=len(score_indexes) - 2,
        plot_path=None,
    )

    assert indices == list(sorted(indices, reverse=True))


@pytest.mark.skip()
@pytest.mark.parametrize(
    "conversion_scoring_functions",
    [
        CONVERT_VIT_ATTENTION,
    ],
)
def test_interesting_patches(conversion_scoring_functions: ConversionScoringFunctions) -> None:
    """
    Sanity check using library assets to assume that scoring and sorting images works as expected.
    :param frames: Should be sorted in order of interesting-ness, with the most interesting frame
    first.
    :return: None
    """

    video_frames = frames_in_video.frames_in_video_opencv(video_path=SAMPLE_TIMELAPSE_INPUT_PATH)

    analysis_frames, drawing_frames = itertools.tee(video_frames.frames, 2)

    vectors = content_aware_timelapse.frames_to_vectors.conversion.frames_to_vectors(
        frames=analysis_frames,
        input_signature="test signature",
        batch_size=1,
        total_input_frames=1,
        convert_batches=conversion_scoring_functions.conversion,
        intermediate_path=None,
    )

    score_indexes: Iterator[IndexScores] = map(
        partial(
            conversion_scoring_functions.scoring,
            original_source_resolution=video_frames.original_resolution,
        ),
        enumerate(vectors),
    )

    viz_frames_frames = (
        image_common.draw_points_on_image(score["interesting_points"], frame)
        for score, frame in zip(score_indexes, drawing_frames)
    )

    more_itertools.consume(
        video_common.display_frame_forward_opencv(
            source=viz_frames_frames, window_name="Viz Frames"
        )
    )
