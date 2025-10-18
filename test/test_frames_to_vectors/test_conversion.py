"""
Test to make sure the frame -> vectors process works as expected.
"""

from pathlib import Path
from tempfile import NamedTemporaryFile
from test.test_viderator import viderator_test_common

import pytest

import content_aware_timelapse.frames_to_vectors.conversion
from content_aware_timelapse.frames_to_vectors.conversion import IntermediateFileInfo
from content_aware_timelapse.frames_to_vectors.conversion_types import ConvertBatchesFunction
from content_aware_timelapse.frames_to_vectors.vector_computation.compute_vectors_clip import (
    CONVERT_CLIP,
)
from content_aware_timelapse.frames_to_vectors.vector_computation.compute_vectors_vit import (
    CONVERT_SCORE_VIT_ATTENTION,
    CONVERT_SCORE_VIT_CLS,
)
from content_aware_timelapse.gpu_discovery import discover_gpus
from content_aware_timelapse.viderator.viderator_types import ImageResolution


@pytest.mark.parametrize(
    "convert_batches_function",
    [
        CONVERT_SCORE_VIT_CLS.conversion,
        CONVERT_SCORE_VIT_ATTENTION.conversion,
        CONVERT_CLIP.conversion,
    ],
)
def test_frames_to_vectors_pipeline(convert_batches_function: ConvertBatchesFunction) -> None:
    """
    Test that sanity-checks loading of the model into the GPU and vectorizing an image.
    :param convert_batches_function: Function under test.
    :return: None
    """

    with NamedTemporaryFile(suffix=".hdf5", delete=True) as tmp:

        output = next(
            content_aware_timelapse.frames_to_vectors.conversion.frames_to_vectors(
                frames=viderator_test_common.create_black_frames_iterator(
                    image_resolution=ImageResolution(1000, 1000), count=1
                ),
                intermediate_info=IntermediateFileInfo(Path(tmp.name), "test signature"),
                batch_size=1,
                total_input_frames=1,
                convert_batches=convert_batches_function,
                gpus=discover_gpus(),
            )
        )

        assert output.any()
