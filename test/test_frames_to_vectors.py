"""
Test to make sure the frame -> vectors process works as expected.
"""

from pathlib import Path
from tempfile import NamedTemporaryFile
from test.test_viderator import viderator_test_common

from content_aware_timelapse import frames_to_vectors
from content_aware_timelapse.viderator.video_common import ImageResolution


def test_frames_to_vectors_pipeline() -> None:
    """
    Test that sanity-checks loading of the model into the GPU and vectorizing an image.
    :return: None
    """

    with NamedTemporaryFile(suffix=".hdf5", delete=True) as tmp:

        output = next(
            frames_to_vectors.frames_to_vectors(
                frames=viderator_test_common.create_black_frames_iterator(
                    image_resolution=ImageResolution(1000, 1000), count=1
                ),
                intermediate_path=Path(tmp.name),
                input_signature="test signature",
                batch_size=1,
                total_input_frames=1,
            )
        )

        assert output.shape == (197, 768)
        assert output.any() and output.all()
