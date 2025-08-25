"""
Unit Test of the VIT-specific library code.
"""

from typing import Tuple

import numpy as np
import pytest
from PIL import Image
from pytest import approx

from content_aware_timelapse.frames_to_vectors.vector_computation import compute_vectors_vit
from content_aware_timelapse.viderator.viderator_types import ImageResolution


@pytest.mark.parametrize("side_length,fill_color", [(224, (123, 116, 103))])
@pytest.mark.parametrize(
    "input_resolution",
    [
        ImageResolution(1000, 1000),  # square
        ImageResolution(2000, 1000),  # wide
        ImageResolution(1000, 2000),  # tall
        ImageResolution(3000, 1000),  # extra wide
        ImageResolution(1920, 1080),  # Widescreen
    ],
)
def test_create_padded_square_resizer(
    side_length: int,
    fill_color: Tuple[int, int, int],
    input_resolution: ImageResolution,
) -> None:
    resizer = compute_vectors_vit.create_padded_square_resizer(
        side_length=side_length, fill_color=fill_color
    )

    # Create a pure white RGB image.
    image = Image.fromarray(
        np.full(
            shape=(input_resolution.height, input_resolution.width, 3),
            fill_value=255,
            dtype=np.uint8,
        ),
        mode="RGB",
    )

    out = resizer(image)

    # Output must be exactly L x L
    assert out.size == (side_length, side_length)

    # Count colors; should be 1 (square) or 2 (content + pad)
    colors = out.getcolors(maxcolors=side_length * side_length)
    assert colors is not None
    color_dict = {color: count for count, color in colors}

    white_count = color_dict.get((255, 255, 255), 0)
    pad_count = color_dict.get(fill_color, 0)
    total = side_length * side_length

    # Sanity: counts add up
    assert white_count + pad_count == total

    # Expected fraction of content area equals min/max aspect ratio
    short = min(input_resolution.width, input_resolution.height)
    long = max(input_resolution.width, input_resolution.height)
    expected_ratio = short / long

    if short == long:
        # Square input, no padding
        assert pad_count == 0
        assert len(color_dict) == 1
        # White fraction should be 1.0
        assert white_count / total == approx(1.0, abs=1 / side_length)
    else:
        # Non-square, there must be padding
        assert pad_count > 0
        # Observed content fraction
        observed_ratio = white_count / total
        # Allow ±1 pixel on the shorter side → ±(1/L) in ratio
        assert observed_ratio == approx(expected_ratio, abs=1 / side_length)
