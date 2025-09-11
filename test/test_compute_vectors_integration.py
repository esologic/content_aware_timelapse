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
from typing import List

import more_itertools
import pytest

import content_aware_timelapse.frames_to_vectors.conversion
import content_aware_timelapse.frames_to_vectors.vector_computation.compute_vectors_vit
from content_aware_timelapse import cat_pipeline
from content_aware_timelapse.cat_pipeline import PointFrameCount
from content_aware_timelapse.frames_to_vectors import vector_scoring
from content_aware_timelapse.frames_to_vectors.conversion_types import (
    ConversionScoringFunctions,
    IndexPointsOfInterest,
    IndexScores,
)
from content_aware_timelapse.frames_to_vectors.vector_computation.compute_vectors_vit import (
    CONVERT_POIS_VIT_ATTENTION,
    CONVERT_SCORE_VIT_ATTENTION,
    CONVERT_SCORE_VIT_CLS,
)
from content_aware_timelapse.viderator import frames_in_video, image_common, video_common
from content_aware_timelapse.viderator.viderator_types import ImageSourceType, RGBInt8ImageType


@pytest.mark.parametrize(
    "conversion_scoring_functions",
    [
        # CONVERT_CLIP, # CLIP doesn't totally work as expected yet...
        CONVERT_SCORE_VIT_CLS,
        CONVERT_SCORE_VIT_ATTENTION,
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
            conversion_scoring_functions.scoring,
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


def test_points_of_interest() -> None:
    """
    Sanity check using library assets to assume that scoring and sorting images works as expected.
    :return: None
    """

    video_frames = frames_in_video.frames_in_video_opencv(video_path=SAMPLE_TIMELAPSE_INPUT_PATH)

    analysis_frames, drawing_frames = itertools.tee(video_frames.frames, 2)

    vectors = content_aware_timelapse.frames_to_vectors.conversion.frames_to_vectors(
        frames=analysis_frames,
        input_signature="test signature",
        batch_size=1,
        total_input_frames=1,
        convert_batches=CONVERT_POIS_VIT_ATTENTION.conversion,
        intermediate_path=None,
    )

    points_of_interest: List[IndexPointsOfInterest] = list(
        map(
            partial(
                CONVERT_POIS_VIT_ATTENTION.compute_pois,
                original_source_resolution=video_frames.original_resolution,
            ),
            enumerate(vectors),
        )
    )

    winning_points_and_counts: List[PointFrameCount] = cat_pipeline.count_frames_filter(
        points_of_interest=points_of_interest,
        total_frame_count=video_frames.total_frame_count,
        drop_frame_threshold=0.7,
    )

    # TODO regions need to be like 5 pixels in x and y apart
    regions = cat_pipeline.top_regions(
        points=winning_points_and_counts,
        image_size=video_frames.original_resolution,
        region_size=(500, 500),
        top_k=2,
    )

    points_only = {points_and_counts.point for points_and_counts in winning_points_and_counts}

    def b(frame: RGBInt8ImageType, score: cat_pipeline.IndexPointsOfInterest):

        with_points = image_common.draw_points_on_image(
            points=[point for point in score["points_of_interest"] if point in points_only],
            image=frame,
        )

        return image_common.draw_regions_on_image(
            regions=regions,
            image=with_points,
        )

    vis_frames = (b(frame, score) for score, frame in zip(points_of_interest, drawing_frames))

    more_itertools.consume(
        video_common.display_frame_forward_opencv(source=vis_frames, window_name="Viz Frames")
    )
