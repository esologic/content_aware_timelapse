""" """

import tempfile
from pathlib import Path

import numpy as np
from html2image import Html2Image
from PIL import Image

from content_aware_timelapse.viderator import image_common
from content_aware_timelapse.viderator.html_on_image import html_template
from content_aware_timelapse.viderator.html_on_image.html_template import FilledTemplate
from content_aware_timelapse.viderator.viderator_types import RGBInt8ImageType


def template_over_image(
    filled_template: FilledTemplate, image: RGBInt8ImageType
) -> RGBInt8ImageType:
    """

    :param filled_template:
    :param image:
    :return:
    """

    input_resolution = image_common.image_resolution(image)

    with tempfile.TemporaryDirectory() as temp_dir:

        renderer = Html2Image(
            size=(input_resolution.width, input_resolution.height),
            custom_flags=[
                "--hide-scrollbars",
                "--force-device-scale-factor=1.0",
                "--default-background-color=00000000",
            ],
            output_path=temp_dir,
        )

        renderer.screenshot(html_str=filled_template, save_as="overlay.png")
        overlay = Image.open(Path(temp_dir) / "overlay.png").convert("RGBA")

    input_base = Image.fromarray(image).convert("RGBA")
    input_base.alpha_composite(overlay, (0, 0))

    # Convert back to RGB and then to numpy array
    final_array = np.array(input_base.convert("RGB"), dtype=np.uint8)

    return RGBInt8ImageType(final_array)


def create_simple_thumbnail(image: RGBInt8ImageType) -> RGBInt8ImageType:
    """

    :param image:
    :return:
    """

    input_resolution = image_common.image_resolution(image)

    thumbnail_template = html_template.simple_thumbnail_overlay(
        width=input_resolution.width,
        height=input_resolution.height,
        gradient_start="rgba(255, 172, 28, 0.8)",
        gradient_stop="rgba(255, 172, 28, 0)",
    )

    return template_over_image(
        filled_template=thumbnail_template,
        image=image,
    )
