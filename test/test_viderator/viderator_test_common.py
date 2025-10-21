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
            np.ones(shape=(image_resolution.height, image_resolution.width, 3), dtype=np.uint8),
        )


def create_random_frames_iterator(image_resolution: ImageResolution, count: int) -> ImageSourceType:
    """
    Creates a test iterator with random RGB frames.

    :param image_resolution: Resolution of resulting frames.
    :param count: Number of frames to create.
    :return: Image source of the random frames.
    """
    for _ in range(count):
        yield cast(
            RGBInt8ImageType,
            np.random.randint(
                low=0,
                high=256,
                size=(image_resolution.height, image_resolution.width, 3),
                dtype=np.uint8,
            ),
        )
