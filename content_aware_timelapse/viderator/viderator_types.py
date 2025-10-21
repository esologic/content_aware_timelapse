"""
Common types used in image/video
"""

from typing import Iterator, List, NamedTuple, NewType, Optional

import click
import numpy as np
from numpy import typing as npt
from PIL.Image import Image as PILImage  # pylint: disable=unused-import

# Other modules can/should import the standard PILImage type from here.

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


class UniqueIntMatrix2DParamType(click.ParamType):
    """
    Parameter type for a 2D matrix of unique integers.
    Accepts strings in the format '1,2;3,4' where rows are separated by ';' and
    columns by ','.
    """

    name = "unique-int-matrix-2d"

    def convert(
        self,
        value: str,
        param: Optional[click.Parameter],
        ctx: Optional[click.Context],
    ) -> List[List[int]]:
        """
        Convert input string to a 2D list of unique integers.
        """
        try:
            rows = value.split(";")
            matrix: List[List[int]] = []
            seen: set[int] = set()

            for row in rows:
                if not row.strip():
                    continue  # skip empty rows
                cols = []
                for x in row.split(","):
                    x_int = int(x.strip())
                    if x_int in seen:
                        raise ValueError(f"Duplicate value detected: {x_int}")
                    seen.add(x_int)
                    cols.append(x_int)
                matrix.append(cols)

            if not matrix:
                raise ValueError("Matrix cannot be empty")

            # Ensure all rows have the same number of columns
            n_cols = len(matrix[0])
            if any(len(r) != n_cols for r in matrix):
                raise ValueError("All rows must have the same number of columns")

            return matrix
        except Exception as e:  # pylint: disable=broad-except
            self.fail(f"{value!r} is not a valid 2D unique integer matrix: {e}", param, ctx)
