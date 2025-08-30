"""
Common functionality needed to test viderator.
"""

from typing import cast

import numpy as np

from content_aware_timelapse.viderator.viderator_types import (
    ImageResolution,
    ImageSourceType,
    RGBInt8ImageType,
)


def create_black_frames_iterator(image_resolution: ImageResolution, count: int) -> ImageSourceType:
    """
    Creates a test iterator.
    :param image_resolution: Resolution of resulting frames.
    :param count: Number of frames to create.
    :return: Image source of the black frames.
    """

    for _ in range(count):
        yield cast(
            RGBInt8ImageType,
            np.zeros(shape=(image_resolution.height, image_resolution.width, 3), dtype=np.uint8),
        )
