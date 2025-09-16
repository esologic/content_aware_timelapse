"""
Common functionality and types used in still images and in video.
"""

import logging
from pathlib import Path
from typing import List, Tuple, cast

import cv2
import numpy as np
from PIL import Image, ImageDraw

from content_aware_timelapse.viderator.viderator_types import (
    AspectRatio,
    ImageResolution,
    RectangleRegion,
    RGBInt8ImageType,
    XYPoint,
)

LOGGER = logging.getLogger(__name__)


def image_resolution(image: RGBInt8ImageType) -> ImageResolution:
    """
    Get an image's resolution.
    :param image: To size.
    :return: Image resolution as an NT.
    """

    return ImageResolution(height=image.shape[0], width=image.shape[1])


def largest_fitting_region(
    source_resolution: ImageResolution, aspect_ratio: AspectRatio
) -> ImageResolution:
    """
    Determine the largest region of `source` that fits the input aspect ratio.
    :param source_resolution: Bounds.
    :param aspect_ratio: Desired aspect ratio.
    :return: Largest resolution.
    """

    # Try width-limited fit
    scale_w = source_resolution.width / aspect_ratio.width
    height_if_full_width = aspect_ratio.height * scale_w

    if height_if_full_width <= source_resolution.height:
        return ImageResolution(source_resolution.width, int(height_if_full_width))

    # Otherwise height-limited fit
    scale_h = source_resolution.height / aspect_ratio.height
    width_if_full_height = aspect_ratio.width * scale_h

    return ImageResolution(int(width_if_full_height), source_resolution.height)


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

        try:
            output = cast(
                RGBInt8ImageType,
                cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR),
            )
        except cv2.error as e:
            raise ValueError(
                f"Couldn't resize image. Shape: {image.shape}, "
                f"New Size {(new_width, new_height)}, "
                f"Max Side Length: {max_side_length}, "
                f"Scale: {scale}"
            ) from e

    if delete:
        del image

    return output


def crop_image(
    image: RGBInt8ImageType, region: RectangleRegion, delete: bool = False
) -> RGBInt8ImageType:
    """
    Crops an image to the given SquareRegion.

    :param image: Input image (H, W, 3) with dtype uint8.
    :param region: SquareRegion specifying (top, left, bottom, right).
    :param delete: If true, `del` will be used on `image` to free memory.
    :return: Cropped image as RGBInt8ImageType.
    """
    cropped = cast(
        RGBInt8ImageType,
        image[region.top : region.bottom, region.left : region.right],
    )

    if delete:
        del image

    return cropped


def draw_points_on_image(
    points: List[XYPoint],
    image: RGBInt8ImageType,
    radius: int = 5,
    color: Tuple[int, int, int] = (255, 0, 0),
) -> RGBInt8ImageType:
    """
    Draws a list of points onto the input image for visualization.
    :param points: To draw.
    :param image: Image to draw on.
    :param radius: Radius of point in pixels.
    :param color: Color of point (R, G, B)
    :return: Modified image.
    """

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    for point in points:

        x, y = point

        bbox = [x - radius, y - radius, x + radius, y + radius]
        draw.ellipse(bbox, fill=color, outline=color)

    # Always return NumPy array
    return RGBInt8ImageType(np.array(pil_image))


def draw_regions_on_image(
    regions: List[RectangleRegion],
    image: RGBInt8ImageType,
    color: Tuple[int, int, int] = (0, 255, 0),
    width: int = 3,
) -> RGBInt8ImageType:
    """
    Draws rectangular regions onto the input image for visualization.

    :param regions: List of SquareRegion objects to draw.
    :param image: Image to draw on (NumPy array).
    :param color: Color of rectangle (R, G, B)
    :param width: Line width of the rectangle.
    :return: Modified image (NumPy array).
    """
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    for region in regions:
        bbox = [region.left, region.top, region.right, region.bottom]
        draw.rectangle(bbox, outline=color, width=width)

    return RGBInt8ImageType(np.array(pil_image))
