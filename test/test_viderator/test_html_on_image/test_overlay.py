"""
Test to make sure HTML overlay writing works.
"""

from pathlib import Path
from test import assets as test_assets
from typing import Tuple, cast

import numpy as np
import pytest

import assets as repo_assets
from content_aware_timelapse.viderator import image_common
from content_aware_timelapse.viderator.html_on_image import overlay_on_image
from content_aware_timelapse.viderator.image_common import RGBInt8ImageType


def color_near(image: RGBInt8ImageType, target: Tuple[int, int, int], tol: int = 10) -> bool:
    """
    Check if any pixel in image is within tol of the target RGB color.
    :param image: To scan.
    :param target: Target RGB.
    :param tol: Tolerance.
    :return: If it's in the image or not.
    """
    return bool(np.any(np.all(np.abs(image - target) <= tol, axis=-1)))


@pytest.mark.integration
@pytest.mark.parametrize(
    "image_path", [repo_assets.EASTERN_BOX_TURTLE_PATH, test_assets.STREAM_SOFTWARE]
)
def test_simple_thumbnail_image_full_fast(artifact_root: Path, image_path: Path) -> None:
    """
    Checks overlay colors (text, shadow, gradient) while preserving some original colors.
    Faster version using boolean masks instead of np.unique.
    :param artifact_root: Artifact root directory to visually look at output.
    :param image_path: Image to test.
    """

    image = image_common.load_rgb_image(path=image_path)

    thumbnail = overlay_on_image.create_simple_thumbnail(
        image=image,
        upper_subtitle="You wouldn't believe it...",
        main_title="It's a Turtle",
        lower_title="Shell and eyes and all.",
        gradient_start="rgba(255, 172, 28, 0.8)",  # orange gradient start
        gradient_stop="rgba(255, 172, 28, 0)",  # orange gradient end
        text_color="rgba(255, 255, 255, 1)",  # white text
        shadow_color="rgba(0, 0, 0, 0.7)",  # dark shadow
    )

    image_common.save_rgb_image(path=artifact_root / "thumbnail.png", image=thumbnail)

    arr = cast(RGBInt8ImageType, np.array(thumbnail))

    # Original colors present (sample a subset for speed)
    sample_idx = np.random.choice(arr.shape[0] * arr.shape[1], size=1000, replace=False)
    sampled_pixels = arr.reshape(-1, 3)[sample_idx]
    assert any(
        tuple(p) in sampled_pixels
        for p in [tuple(c) for c in image.reshape(-1, 3)[:1000]]  # pylint: disable=no-member
    )

    # Check overlay colors exist
    expected_colors = [
        (255, 255, 255),  # white text
        (0, 0, 0),  # black text
        (255, 172, 28),  # orange gradient
    ]

    for color in expected_colors:
        assert color_near(arr, color, tol=10)
