"""
Common types used in image/video
"""

from typing import Iterator, NamedTuple, NewType

import numpy as np
from numpy import typing as npt
from PIL.Image import Image as PILImage  # pylint: disable=unused-import

RGBInt8ImageType = NewType("RGBInt8ImageType", npt.NDArray[np.uint8])
"""
Dimensions are (Width, Height, Colors). Type is np.uint8
"""

ImageSourceType = Iterator[RGBInt8ImageType]


class ImageResolution(NamedTuple):
    """
    Standard NT for image dimensions. Creators are responsible for making sure the order is
    correct.
    """

    width: int
    height: int


class XYPoint(NamedTuple):
    """
    Standard NT for pixel locations. (0, 0) is in the top left corner of the image.
    """

    x: int
    y: int
