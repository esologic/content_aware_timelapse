"""
Common functionality and types used in still images and in video.
"""

from typing import Iterator, NamedTuple, NewType

import numpy as np
import numpy.typing as npt

# dimensions are (Width, Height, Colors). Type is np.uint8
RGBInt8ImageType = NewType("RGBInt8ImageType", npt.NDArray[np.uint8])
ImageSourceType = Iterator[RGBInt8ImageType]


class ImageResolution(NamedTuple):
    """
    Standard NT for image dimensions. Creators are responsible for making sure the order is
    correct.
    """

    width: int
    height: int


def image_resolution(image: RGBInt8ImageType) -> ImageResolution:
    """
    Get an image's resolution.
    :param image: To size.
    :return: Image resolution as an NT.
    """

    return ImageResolution(height=image.shape[0], width=image.shape[1])
