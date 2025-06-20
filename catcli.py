"""Main module."""

import logging
from pathlib import Path
from typing import List, Optional

import click

from content_aware_timelapse.cat_pipeline import create_timelapse

LOGGER_FORMAT = "[%(asctime)s - %(process)s - %(name)20s - %(levelname)s] %(message)s"
LOGGER_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOGGER_FORMAT,
    datefmt=LOGGER_DATE_FORMAT,
)

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option(
    "--input",
    "-i",
    "input_files",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path),
    help="Input file(s). Can be given multiple times.",
    required=True,
    multiple=True,
)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True, path_type=Path),
    help="Output will be written to this file.",
    required=True,
)
@click.option(
    "--duration",
    "-d",
    type=click.FloatRange(min=1),
    help="Frames are sent to GPU for processing in batches of this size.",
    required=True,
    default=30.0,
    show_default=True,
)
@click.option(
    "--output-fps",
    "-f",
    type=click.FloatRange(min=1),
    help="Frames are sent to GPU for processing in batches of this size.",
    required=True,
    default=60.0,
    show_default=True,
)
@click.option(
    "--batch-size",
    "-b",
    type=click.IntRange(min=1),
    help="Frames are sent to GPU for processing in batches of this size.",
    required=True,
    default=600,
    show_default=True,
)
@click.option(
    "--vectors-path",
    "-v",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path),
    help="Intermediate vectors will be written to this path. Can be used to re-run.",
    required=False,
)
@click.option(
    "--viz-path",
    "-z",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path),
    help=(
        "A visualisation describing the timelapse creation "
        "process will be written to this path if given"
    ),
    required=False,
)
def main(  # pylint: disable=too-many-locals
    input_files: List[Path],
    output_path: Path,
    duration: float,
    output_fps: float,
    batch_size: int,
    vectors_path: Optional[Path],
    viz_path: Optional[Path],
) -> None:
    """
    Create a timelapse based on the most interesting parts of a video rather than blindly
    down-selecting frames.

    \f

    :param input_files: See click docs.
    :param output_path: See click docs.
    :param duration: See click docs.
    :param output_fps: See click docs.
    :param batch_size: See click docs.
    :param vectors_path: See click docs.
    :param viz_path: See click docs.
    :return: None
    """

    create_timelapse(
        input_files=input_files,
        output_path=output_path,
        duration=duration,
        output_fps=output_fps,
        batch_size=batch_size,
        vectors_path=vectors_path,
        plot_path=viz_path,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
