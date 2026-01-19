"""Uses pathlib to make referencing test assets by path easier."""

from pathlib import Path

_ASSETS_DIRECTORY = Path(__file__).parent.resolve()

EASTERN_BOX_TURTLE_PATH = _ASSETS_DIRECTORY / "eastern_box_turtle.jpg"
