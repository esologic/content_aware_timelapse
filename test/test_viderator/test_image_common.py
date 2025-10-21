"""
Test of common image math and manipulation functions.
"""

from typing import cast

import numpy as np
import pytest

from content_aware_timelapse.viderator import image_common
from content_aware_timelapse.viderator.viderator_types import (
    AspectRatio,
    ImageResolution,
    RectangleRegion,
    RGBInt8ImageType,
)


@pytest.mark.parametrize(
    "source, ratio, expected",
    [
        # Fits perfectly (16:9 inside 1920x1080)
        (
            ImageResolution(1920, 1080),
            AspectRatio(16, 9),
            ImageResolution(1920, 1080),
        ),
        # Cropped width (4:3 inside 1920x1080)
        (
            ImageResolution(1920, 1080),
            AspectRatio(4, 3),
            ImageResolution(1440, 1080),
        ),
        # Cropped height (16:9 inside 1200x1600)
        (
            ImageResolution(1200, 1600),
            AspectRatio(16, 9),
            ImageResolution(1200, 675),
        ),
        # Square ratio inside widescreen
        (
            ImageResolution(1920, 1080),
            AspectRatio(1, 1),
            ImageResolution(1080, 1080),
        ),
        # Square ratio inside tall portrait
        (
            ImageResolution(800, 1200),
            AspectRatio(1, 1),
            ImageResolution(800, 800),
        ),
        # Portrait 9:16 inside 1920x1080 (width limited)
        (
            ImageResolution(1920, 1080),
            AspectRatio(9, 16),
            ImageResolution(607, 1080),
        ),
        # Portrait 9:16 inside 1200x1600 (height limited)
        (
            ImageResolution(1200, 1600),
            AspectRatio(9, 16),
            ImageResolution(900, 1600),
        ),
    ],
)
def test_largest_fitting_region(
    source: ImageResolution, ratio: AspectRatio, expected: ImageResolution
) -> None:
    """
    Smoke test of the function.
    :param source: Input.
    :param ratio: Input.
    :param expected: Expected output.
    :return: None
    """
    assert image_common.largest_fitting_region(source, ratio) == expected


def test_reshape_from_regions_linear_layout() -> None:
    """
    Verify that a 2×2 input reshapes correctly into a 1×4 output layout.
    :return: None
    """

    region_size = 10
    height, width = region_size, region_size

    # Create 2×2 synthetic image
    input_image = cast(RGBInt8ImageType, np.zeros((height * 2, width * 2, 3), dtype=np.uint8))
    input_image[0:height, 0:width] = [255, 0, 0]  # red
    input_image[0:height, width : 2 * width] = [0, 255, 0]  # green
    input_image[height : 2 * height, 0:width] = [0, 0, 255]  # blue
    input_image[height : 2 * height, width : 2 * width] = [255, 255, 0]  # yellow

    # Define regions corresponding to the four quadrants
    regions = (
        RectangleRegion(0, 0, height, width),  # red
        RectangleRegion(0, width, height, 2 * width),  # green
        RectangleRegion(height, 0, 2 * height, width),  # blue
        RectangleRegion(height, width, 2 * height, 2 * width),  # yellow
    )

    # 1×4 layout — top row, left to right
    layout_matrix = [[0, 1, 2, 3]]

    reshaped = image_common.reshape_from_regions(
        image=input_image,
        prioritized_poi_regions=regions,
        layout_matrix=layout_matrix,
    )

    # Output should have 1 row of height `h` and 4 columns of width `w`
    assert reshaped.shape == (height, width * 4, 3)

    # Verify quadrants appear in correct order across the row
    assert np.all(reshaped[0:height, 0:width] == [255, 0, 0])  # red
    assert np.all(reshaped[0:height, width : 2 * width] == [0, 255, 0])  # green
    assert np.all(reshaped[0:height, 2 * width : 3 * width] == [0, 0, 255])  # blue
    assert np.all(reshaped[0:height, 3 * width : 4 * width] == [255, 255, 0])  # yellow


def test_reshape_from_regions_raises_on_mismatched_regions() -> None:
    """
    Ensure the function raises ValueError if region count doesn't match layout matrix size.
    :return: None
    """

    with pytest.raises(ValueError, match="Expected 4 regions"):
        image_common.reshape_from_regions(
            image=cast(RGBInt8ImageType, np.zeros((10, 10, 3), dtype=np.uint8)),
            prioritized_poi_regions=(RectangleRegion(0, 0, 5, 5),),
            layout_matrix=[[0, 1], [2, 3]],
        )
