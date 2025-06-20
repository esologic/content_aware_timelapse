"""
End -> End Testing of the vector computation and sorting.
"""

from pathlib import Path
from test.assets import (
    SORTED_BENCH_SCENES_PATHS,
    SORTED_STREAM_GENERIC_PATHS,
    SORTED_STREAM_SOFTWARE_PATHS,
)
from typing import List

import pytest

import content_aware_timelapse.frames_to_vectors.conversion
from content_aware_timelapse import vector_scoring
from content_aware_timelapse.frames_to_vectors.vector_computation.compute_vectors_vit import (
    compute_vectors_vit_cls,
)
from content_aware_timelapse.vector_scoring import IndexScores
from content_aware_timelapse.viderator import image_common
from content_aware_timelapse.viderator.image_common import ImageSourceType


@pytest.mark.parametrize(
    "frames",
    [
        list(map(image_common.load_rgb_image, SORTED_BENCH_SCENES_PATHS)),
        list(map(image_common.load_rgb_image, SORTED_STREAM_SOFTWARE_PATHS)),
        list(map(image_common.load_rgb_image, SORTED_STREAM_GENERIC_PATHS)),
    ],
)
def test_sorting_converted_vectors(frames: ImageSourceType) -> None:
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
        convert_batches=compute_vectors_vit_cls,
        intermediate_path=None,
    )

    score_indexes: List[IndexScores] = list(
        map(vector_scoring.calculate_scores, enumerate(vectors))
    )

    indices = vector_scoring._score_and_sort_frames(  # pylint: disable=protected-access
        index_scores=score_indexes,
        num_output_frames=len(score_indexes) - 2,
        plot_path=Path("./fig.png"),
    )

    assert list(sorted(indices, reverse=True)) == indices
