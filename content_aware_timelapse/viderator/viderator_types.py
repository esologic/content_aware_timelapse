"""
Common types used in image/video
"""

from typing import Iterator, NamedTuple, NewType, Optional

import click
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


class AspectRatio(NamedTuple):
    """
    Standard NT for an aspect ratio.
    """

    width: float
    height: float


class RectangleRegion(NamedTuple):
    """
    Standard NT for defining a rectangular area of an image.
    """

    top: int
    left: int
    bottom: int
    right: int


class AspectRatioParamType(click.ParamType):
    """
    Parameter for passing in an aspect ratio.
    """

    name: str = "aspect-ratio"

    def convert(
        self,
        value: str,
        param: Optional[click.Parameter],
        ctx: Optional[click.Context],
    ) -> AspectRatio:
        """
        Converts input string to namedtuple.
        :param value: To convert.
        :param param: Only consumed in error state.
        :param ctx: Only consumed in error state.
        :return: Parsed type.
        """

        try:
            width_str, height_str = str(value).split(":")
            width = float(width_str)
            height = float(height_str)
        except ValueError:
            self.fail(
                f"{value!r} is not a valid aspect ratio. Use the format WIDTH:HEIGHT",
                param,
                ctx,
            )

        if width <= 0 or height <= 0:
            self.fail(
                f"Aspect ratio values must be positive (got {width}:{height})",
                param,
                ctx,
            )

        return AspectRatio(width=width, height=height)
