"""
Common functionality and types used in still images and in video.
"""

import logging
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from PIL import Image

from content_aware_timelapse.viderator.viderator_types import ImageResolution, RGBInt8ImageType

LOGGER = logging.getLogger(__name__)


def image_resolution(image: RGBInt8ImageType) -> ImageResolution:
    """
    Get an image's resolution.
    :param image: To size.
    :return: Image resolution as an NT.
    """

    return ImageResolution(height=image.shape[0], width=image.shape[1])


def load_rgb_image(path: Path) -> RGBInt8ImageType:
    """
    Loads an image from a path and returns it as an RGB uint8 numpy array.

    :param path: Path to the image file.
    :return: RGB image as a (H, W, 3) uint8 ndarray.
    """
    with Image.open(path) as img:
        img = img.convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
    return RGBInt8ImageType(arr)


def resize_image(
    image: RGBInt8ImageType, resolution: ImageResolution, delete: bool = False
) -> RGBInt8ImageType:
    """
    Resizes an image to the input resolution.
    Uses, `cv2.INTER_CUBIC`, which is visually good-looking but somewhat slow.
    May want to be able to pass this in.
    :param image: To scale.
    :param resolution: Output resolution.
    :param delete: If true, `del` will be used on `image` to force it's memory to be released.
    :return: Scaled image.
    """

    output = cast(
        RGBInt8ImageType,
        cv2.resize(image, (resolution.height, resolution.width), interpolation=cv2.INTER_CUBIC),
    )

    if delete:
        # The image has now been 'consumed', and can't be used again.
        # We delete this frame here to avoid memory leaks.
        # Not really sure if this is needed, but it shouldn't cause harm.
        del image

    # The scaled image.
    return output


def resize_image_max_side(
    image: RGBInt8ImageType, max_side_length: int, delete: bool = False
) -> RGBInt8ImageType:
    """
    Resizes an image such that its largest side is `max_side_length`, preserving aspect ratio.
    Uses `cv2.INTER_LINEAR` for speed.

    :param image: Input image to scale.
    :param max_side_length: Maximum length of the largest side after scaling.
    :param delete: If true, deletes the input image to free memory.
    :return: Scaled image.
    """
    resolution = image_resolution(image)

    # Determine scaling factor
    scale = max_side_length / max(resolution.height, resolution.width)

    # If image is already smaller than max_side_length, don't scale up
    if scale >= 1.0:
        output = image.copy()
    else:
        new_width = int(resolution.width * scale)
        new_height = int(resolution.height * scale)

        output = cast(
            RGBInt8ImageType,
            cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR),
        )

    if delete:
        del image

    return output
