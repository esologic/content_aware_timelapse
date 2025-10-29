"""
Combines the HTML templating process with image manipulation, enabling drawing webpages on top of
images.

Note: Needs google chrome!!

wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install ./google-chrome-stable_current_amd64.deb

Eventually we should use a docker image or something to do this.
"""

import tempfile
from pathlib import Path

import numpy as np
from html2image import Html2Image
from PIL import Image

from content_aware_timelapse.viderator import image_common
from content_aware_timelapse.viderator.html_on_image import read_html_template
from content_aware_timelapse.viderator.html_on_image.read_html_template import FilledTemplate
from content_aware_timelapse.viderator.viderator_types import RGBInt8ImageType


def template_over_image(
    filled_template: FilledTemplate, image: RGBInt8ImageType
) -> RGBInt8ImageType:
    """
    Take a filled template document (a string that is an HTML document), render it as an image,
    and composite that image on top of the input `image`.
    :param filled_template: HTML text.
    :param image: To draw on.
    :return: Document rendered and added on top of `image`.
    """

    input_resolution = image_common.image_resolution(image)

    with tempfile.TemporaryDirectory() as temp_dir:

        Path(temp_dir).mkdir(parents=True, exist_ok=True)

        renderer = Html2Image(
            size=(input_resolution.width, input_resolution.height),
            custom_flags=[
                "--hide-scrollbars",
                "--force-device-scale-factor=1.0",
                "--default-background-color=00000000",
            ],
            output_path=temp_dir,
            browser_executable="/usr/bin/google-chrome",
        )

        renderer.screenshot(html_str=filled_template, save_as="overlay.png")
        overlay = Image.open(Path(temp_dir) / "overlay.png").convert("RGBA")

    input_base = Image.fromarray(image).convert("RGBA")
    input_base.alpha_composite(overlay, (0, 0))

    # Convert back to RGB, squashing alpha, and then to numpy array
    final_array = np.array(input_base.convert("RGB"), dtype=np.uint8)

    return RGBInt8ImageType(final_array)


def create_simple_thumbnail(  # pylint: disable=too-many-positional-arguments
    image: RGBInt8ImageType,
    upper_subtitle: str,
    main_title: str,
    lower_title: str,
    gradient_start: str,
    gradient_stop: str,
    text_color: str,
    shadow_color: str,
) -> RGBInt8ImageType:
    """
    Convenience function, combining the "simple thumbnail overlay" asset with a given image.
    :param image: Rendered HTML document is composited over this image.
    :param upper_subtitle: Passed to template.
    :param main_title: Passed to template.
    :param lower_title: Passed to template.
    :param gradient_start: Passed to template.
    :param gradient_stop: Passed to template.
    :param text_color: Passed to template.
    :param shadow_color: Passed to template.
    :return: The input image with the rendered "Simple Thumbnail Overlay" drawn on top.
    """

    input_resolution = image_common.image_resolution(image)

    filled_template = read_html_template.simple_thumbnail_overlay(
        width=input_resolution.width,
        height=input_resolution.height,
        upper_subtitle=upper_subtitle,
        main_title=main_title,
        lower_title=lower_title,
        gradient_start=gradient_start,
        gradient_stop=gradient_stop,
        text_color=text_color,
        shadow_color=shadow_color,
    )

    return template_over_image(
        filled_template=filled_template,
        image=image,
    )
