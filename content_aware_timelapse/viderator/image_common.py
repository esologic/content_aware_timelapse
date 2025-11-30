"""
Common functionality and types used in still images and in video.
"""

import logging
import math
from numbers import Integral
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


def resolution_aspect_ratio(resolution: ImageResolution) -> AspectRatio:
    """
    Compute the aspect ratio of an image's resolution

    :param resolution:
    :return: AspectRatio NT with width and height as floats.
    """

    if isinstance(resolution.width, Integral) and isinstance(resolution.height, Integral):
        g = math.gcd(resolution.width, resolution.height)
        return AspectRatio(width=float(resolution.width // g), height=float(resolution.height // g))
    else:
        return AspectRatio(width=float(resolution.width), height=float(resolution.height))


def region_to_resolution(region: RectangleRegion) -> ImageResolution:
    """
    Converts a RectangleRegion into an ImageResolution.

    :param region: RectangleRegion defining the area.
    :return: ImageResolution with width and height of the region.
    """
    width = region.right - region.left
    height = region.bottom - region.top

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid RectangleRegion {region}: width and height must be positive.")

    return ImageResolution(width=width, height=height)


def largest_fitting_region(
    source_resolution: ImageResolution,
    aspect_ratio: AspectRatio,
    even_dimensions: bool = False,
) -> ImageResolution:
    """
    Determine the largest region of `source` that fits the input aspect ratio.
    :param source_resolution: Bounds.
    :param aspect_ratio: Desired aspect ratio.
    :param even_dimensions: If True, the output resolution will be rounded down such that the
    dimensions are both even numbers. Useful for encoding.
    :return: Largest resolution.
    """

    # Try width-limited fit
    scale_w = source_resolution.width / aspect_ratio.width
    height_if_full_width = aspect_ratio.height * scale_w

    if height_if_full_width <= source_resolution.height:
        width = source_resolution.width
        height = int(height_if_full_width)
    else:
        # Otherwise height-limited fit
        scale_h = source_resolution.height / aspect_ratio.height
        width = int(aspect_ratio.width * scale_h)
        height = source_resolution.height

    if even_dimensions:
        width -= width % 2
        height -= height % 2

    return ImageResolution(width, height)


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


def save_rgb_image(path: Path, image: RGBInt8ImageType) -> None:
    """
    Saves an RGB uint8 numpy array to a file at the given path.

    :param path: Destination file path.
    :param image: RGB image as a (H, W, 3) uint8 ndarray.
    """
    img = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    img.save(path)


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

        bbox = [point.x - radius, point.y - radius, point.x + radius, point.y + radius]
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


def composite_region(
    destination_image: RGBInt8ImageType,
    destination_region: RectangleRegion,
    image_to_add: RGBInt8ImageType,
) -> RGBInt8ImageType:
    """
    Pastes an image into the destination image at the coordinates defined
    by `destination_region`.

    :param destination_image: The pre-allocated composited image.
    :param destination_region: RectangleRegion defining the top-left corner for placement.
    :param image_to_add: The region to paste into the destination image.
    :return: Updated destination image.
    """

    res = region_to_resolution(destination_region)

    # Ensure the source image matches the destination region size
    if image_to_add.shape[0] != res.height or image_to_add.shape[1] != res.width:
        raise ValueError(
            f"Source image shape {image_to_add.shape[:2]} "
            f"does not match expected region size {(res.width, res.height)}"
        )

    # Paste the source image into the destination image
    destination_image[
        destination_region.top : destination_region.bottom,
        destination_region.left : destination_region.right,
    ] = image_to_add

    return destination_image


def reshape_from_regions(  # pylint: disable=too-many-locals
    image: RGBInt8ImageType,
    prioritized_poi_regions: Tuple[RectangleRegion, ...],
    layout_matrix: List[List[int]],
) -> RGBInt8ImageType:
    """
    Takes region of `image` and composites them in the layout defined by `layout_matrix`, forming
    a new image.

    :param image: The source image to crop from.
    :param prioritized_poi_regions: Flat list of rectangular regions to crop.
    :param layout_matrix: 2D grid of integers representing priorities.
    :return: Composited output image.
    """
    num_rows = len(layout_matrix)
    num_cols = len(layout_matrix[0])

    if len(prioritized_poi_regions) != num_rows * num_cols:
        raise ValueError(
            f"Expected {num_rows * num_cols} regions for a {num_rows}x{num_cols} matrix, "
            f"but got {len(prioritized_poi_regions)}."
        )

    # Determine size of each region
    region_resolution = region_to_resolution(prioritized_poi_regions[0])
    output_width = num_cols * region_resolution.width
    output_height = num_rows * region_resolution.height

    # Create empty canvas
    composited: RGBInt8ImageType = cast(
        RGBInt8ImageType, np.zeros((output_height, output_width, 3), dtype=np.uint8)
    )

    # Reshape flat regions into rows
    row_regions = [
        prioritized_poi_regions[r * num_cols : (r + 1) * num_cols] for r in range(num_rows)
    ]

    # Pair each region with its priority and record its grid position
    regions_with_priority: list[tuple[RectangleRegion, int, int, int]] = []
    for row_idx, (row_region_list, row_priority_list) in enumerate(zip(row_regions, layout_matrix)):
        for col_idx, (region, priority) in enumerate(zip(row_region_list, row_priority_list)):
            regions_with_priority.append((region, priority, row_idx, col_idx))

    # Sort descending by priority
    regions_with_priority.sort(key=lambda rp: rp[1], reverse=True)

    # Composite regions in priority order
    for region, _priority, row_idx, col_idx in regions_with_priority:
        cropped = crop_image(image=image, region=region)

        # Compute destination coordinates in the output canvas
        destination_region = RectangleRegion(
            top=row_idx * region_resolution.height,
            left=col_idx * region_resolution.width,
            bottom=(row_idx + 1) * region_resolution.height,
            right=(col_idx + 1) * region_resolution.width,
        )

        composite_region(
            destination_image=composited,
            destination_region=destination_region,
            image_to_add=cropped,
        )

    return composited
