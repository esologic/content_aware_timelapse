"""
Common functionality and types used in still images and in video.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from content_aware_timelapse.viderator.viderator_types import ImageResolution, RGBInt8ImageType


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
