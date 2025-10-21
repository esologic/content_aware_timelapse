"""
Test of common image math and manipulation functions.
"""

from pathlib import Path
from test import assets

import pytest

from content_aware_timelapse.viderator import image_common
from content_aware_timelapse.viderator.viderator_types import (
    AspectRatio,
    ImageResolution,
    RectangleRegion,
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


def test_reshape_from_regions() -> None:
    """

    :return:
    """

    input_image = image_common.load_rgb_image(path=assets.EASTERN_BOX_TURTLE_PATH)

    reshaped = image_common.reshape_from_regions(
        image=input_image,
        prioritized_poi_regions=(
            RectangleRegion(0, 0, 500, 500),
            RectangleRegion(0, 0, 500, 500),
            RectangleRegion(0, 0, 500, 500),
            RectangleRegion(500, 500, 1000, 1000),
        ),
        layout_matrix=[[0, 1], [2, 3]],
    )

    image_common.save_rgb_image(
        path=Path("./turtle.png"),
        image=reshaped,
    )
