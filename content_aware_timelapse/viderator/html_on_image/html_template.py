"""Prompt rendering helpers."""

# pylint: disable=too-many-positional-arguments

from pathlib import Path
from typing import NewType, Union, cast

from jinja2 import Template

from content_aware_timelapse.viderator.html_on_image import assets

FilledTemplate = NewType("FilledTemplate", str)


def render_html_template(template_path: Path, **context: Union[str, int, float]) -> FilledTemplate:
    """
    Load and render a Jinja2 prompt template.

    :param template_path: Path to the template file.
    :param context: Variables to substitute into the template.
    :return: Rendered template string.
    """

    with open(template_path, encoding="utf-8") as fp:
        template = Template(fp.read())

    return cast(FilledTemplate, template.render(context))


def simple_thumbnail_overlay(
    width: int, height: int, gradient_start: str, gradient_stop: str
) -> FilledTemplate:
    """Render the 'simple thumbnail overlay' template."""
    return render_html_template(
        assets.SIMPLE_THUMBNAIL_OVERLAY_TEMPLATE,
        width=width,
        height=height,
        gradient_start=gradient_start,
        gradient_stop=gradient_stop,
    )
