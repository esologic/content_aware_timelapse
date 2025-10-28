from pathlib import Path
from test import assets

from content_aware_timelapse.viderator import image_common
from content_aware_timelapse.viderator.html_on_image import overlay


def test_simple_thumbnail_image(artifact_root: Path) -> None:

    turtle = image_common.load_rgb_image(path=assets.EASTERN_BOX_TURTLE_PATH)

    thumbnail = overlay.create_simple_thumbnail(image=turtle)

    image_common.save_rgb_image(path=artifact_root / "thumbnail_turtle.png", image=thumbnail)
