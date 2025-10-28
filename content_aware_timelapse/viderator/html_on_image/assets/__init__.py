"""Makes looking up asset locations easier in other code"""

from pathlib import Path

_ASSETS_DIRECTORY = Path(__file__).parent.resolve()

SIMPLE_THUMBNAIL_OVERLAY_TEMPLATE = (
    _ASSETS_DIRECTORY / "simple_thumbnail_overlay_template.html.jinja2"
)
