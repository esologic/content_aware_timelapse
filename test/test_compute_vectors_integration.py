"""
End -> End Testing of the vector computation and sorting.
"""

import itertools
from test.assets import BORING_IMAGES_PATHS, INTERESTING_IMAGES_PATHS
from test.test_viderator import viderator_test_common
from typing import List

import content_aware_timelapse.frames_to_vectors.conversion
from content_aware_timelapse import vector_scoring
from content_aware_timelapse.frames_to_vectors.vector_computation.compute_vectors_clip import (
    compute_vectors_clip,
)
from content_aware_timelapse.vector_scoring import IndexScores
from content_aware_timelapse.viderator import image_common
from content_aware_timelapse.viderator.video_common import ImageResolution


def test_sorting_converted_vectors() -> None:
    """
    Sanity check using library assets to assume that scoring and sorting images works as expected.
    :return: None
    """

    vectors = content_aware_timelapse.frames_to_vectors.conversion.frames_to_vectors(
        frames=itertools.chain.from_iterable(
            [
                viderator_test_common.create_black_frames_iterator(
                    image_resolution=ImageResolution(1620, 1080), count=1
                ),
                map(
                    image_common.load_rgb_image,
                    itertools.chain.from_iterable([BORING_IMAGES_PATHS, INTERESTING_IMAGES_PATHS]),
                ),
            ]
        ),
        input_signature="test signature",
        batch_size=1,
        total_input_frames=1,
        convert_batches=compute_vectors_clip,
        intermediate_path=None,
    )

    score_indexes: List[IndexScores] = list(
        map(vector_scoring.calculate_scores, enumerate(vectors))
    )

    print("stop")

    for score in score_indexes:
        print(score)
