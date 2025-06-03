"""
Test to make sure the frame -> vectors process works as expected.
"""

from pathlib import Path
from tempfile import NamedTemporaryFile
from test.test_viderator import viderator_test_common

import pytest

import content_aware_timelapse.frames_to_vectors.conversion
from content_aware_timelapse.frames_to_vectors.conversion_types import ConvertBatchesFunction
from content_aware_timelapse.frames_to_vectors.vector_computation.compute_vectors_clip import (
    compute_vectors_clip,
)
from content_aware_timelapse.frames_to_vectors.vector_computation.compute_vectors_vit import (
    compute_vectors_vit_forward_features,
)
from content_aware_timelapse.viderator.video_common import ImageResolution


@pytest.mark.parametrize(
    "convert_batches_function", [compute_vectors_vit_forward_features, compute_vectors_clip]
)
def test_frames_to_vectors_pipeline(convert_batches_function: ConvertBatchesFunction) -> None:
    """
    Test that sanity-checks loading of the model into the GPU and vectorizing an image.
    :return: None
    """

    with NamedTemporaryFile(suffix=".hdf5", delete=True) as tmp:

        output = next(
            content_aware_timelapse.frames_to_vectors.conversion.frames_to_vectors(
                frames=viderator_test_common.create_black_frames_iterator(
                    image_resolution=ImageResolution(1000, 1000), count=1
                ),
                intermediate_path=Path(tmp.name),
                input_signature="test signature",
                batch_size=1,
                total_input_frames=1,
                convert_batches=convert_batches_function,
            )
        )

        assert output.any() and output.all()
