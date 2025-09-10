"""Main module."""

import itertools
import logging
from enum import Enum
from pathlib import Path
from typing import List, Optional

import click
from tqdm import tqdm

from content_aware_timelapse import cat_pipeline
from content_aware_timelapse.cli_common import create_enum_option
from content_aware_timelapse.frames_to_vectors.vector_computation.compute_vectors_clip import (
    CONVERT_CLIP,
)
from content_aware_timelapse.frames_to_vectors.vector_computation.compute_vectors_vit import (
    CONVERT_POIS_VIT_ATTENTION,
    CONVERT_SCORE_VIT_ATTENTION,
    CONVERT_SCORE_VIT_CLS,
)
from content_aware_timelapse.viderator import video_common

LOGGER_FORMAT = "[%(asctime)s - %(process)s - %(name)20s - %(levelname)s] %(message)s"
LOGGER_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOGGER_FORMAT,
    datefmt=LOGGER_DATE_FORMAT,
)

LOGGER = logging.getLogger(__name__)


class VectorBackendScores(str, Enum):
    """
    For the CLI, string representations of the different vectorization backends.
    """

    vit_cls = "vit-cls"
    vit_attention = "vit-attention"
    clip = "clip"


class VectorBackendPOIs(str, Enum):
    """
    For the CLI, string representations of the different Point of Interest backends.
    """

    vit_attention = "vit-attention"


input_files_arg = click.option(
    "--input",
    "-i",
    "input_files",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path),
    help="Input file(s). Can be given multiple times.",
    required=True,
    multiple=True,
)

output_path_arg = click.option(
    "--output-path",
    "-o",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True, path_type=Path),
    help="Output will be written to this file.",
    required=True,
)

duration_arg = click.option(
    "--duration",
    "-d",
    type=click.FloatRange(min=1),
    help="Frames are sent to GPU for processing in batches of this size.",
    required=True,
    default=30.0,
    show_default=True,
)

output_fps_arg = click.option(
    "--output-fps",
    "-f",
    type=click.FloatRange(min=1),
    help="Frames are sent to GPU for processing in batches of this size.",
    required=True,
    default=60.0,
    show_default=True,
)

# Content-aware parameters

batch_size_arg = click.option(
    "--batch-size",
    "-b",
    type=click.IntRange(min=1),
    help="Frames are sent to GPU for processing in batches of this size.",
    required=True,
    default=600,
    show_default=True,
)

buffer_size_arg = click.option(
    "--buffer-size",
    "-b",
    type=click.IntRange(min=0),
    help=(
        "The number of frames to load into an in-memory buffer. "
        "This makes sure the GPUs have fast access to more frames rather than have the GPU "
        "waiting on disk/network IO."
    ),
    required=False,
    default=0,
    show_default=True,
)


viz_path_arg = click.option(
    "--viz-path",
    "-z",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path),
    help=(
        "A visualisation describing the timelapse creation "
        "process will be written to this path if given"
    ),
    required=False,
)


_CONVERSION_SCORING_FUNCTIONS_LOOKUP = {
    VectorBackendScores.vit_cls: CONVERT_SCORE_VIT_CLS,
    VectorBackendScores.vit_attention: CONVERT_SCORE_VIT_ATTENTION,
    VectorBackendScores.clip: CONVERT_CLIP,
}


@click.group()
def cli() -> None:
    """
    Tools to create timelapses.

    \f

    :return: None
    """


@cli.command(short_help="Looks at the content of the frames to choose the most interesting ones.")
@input_files_arg
@output_path_arg
@duration_arg
@output_fps_arg
@batch_size_arg
@buffer_size_arg
@create_enum_option(
    arg_flag="--backend",
    help_message="Sets which vectorization backend is used.",
    default=VectorBackendScores.vit_cls,
    input_enum=VectorBackendScores,
)
@click.option(
    "--vectors-path",
    "-v",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path),
    help="Intermediate vectors will be written to this path. Can be used to re-run.",
    required=False,
)
@viz_path_arg
def content(  # pylint: disable=too-many-locals,too-many-positional-arguments
    input_files: List[Path],
    output_path: Path,
    duration: float,
    output_fps: float,
    batch_size: int,
    buffer_size: Optional[int],
    backend: VectorBackendScores,
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
    :param buffer_size: See click docs.
    :param backend: See click docs.
    :param vectors_path: See click docs.
    :param viz_path: See click docs.
    :return: None
    """

    cat_pipeline.create_uncropped_timelapse(
        input_files=input_files,
        output_path=output_path,
        duration=duration,
        output_fps=output_fps,
        batch_size=batch_size,
        buffer_size=buffer_size,
        conversion_scoring_functions=_CONVERSION_SCORING_FUNCTIONS_LOOKUP[backend],
        vectors_path=vectors_path,
        plot_path=viz_path,
    )


_CONVERSION_POIS_FUNCTIONS_LOOKUP = {VectorBackendPOIs.vit_attention: CONVERT_POIS_VIT_ATTENTION}


@cli.command(short_help="Adds content aware cropping at the cost of performance.")
@input_files_arg
@output_path_arg
@duration_arg
@output_fps_arg
@batch_size_arg
@buffer_size_arg
@create_enum_option(
    arg_flag="--backend-pois",
    help_message="Sets which vectorization backend is used.",
    default=VectorBackendPOIs.vit_attention,
    input_enum=VectorBackendPOIs,
)
@create_enum_option(
    arg_flag="--backend-scores",
    help_message="Sets which vectorization backend is used.",
    default=VectorBackendScores.vit_cls,
    input_enum=VectorBackendScores,
)
@click.option(
    "--vectors-path-pois",
    "-vp",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path),
    help="Intermediate vectors will be written to this path. Can be used to re-run.",
    required=False,
)
@click.option(
    "--vectors-path-scores",
    "-vs",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path),
    help="Intermediate vectors will be written to this path. Can be used to re-run.",
    required=False,
)
@viz_path_arg
def content_cropped(  # pylint: disable=too-many-locals,too-many-positional-arguments,too-many-arguments
    input_files: List[Path],
    output_path: Path,
    duration: float,
    output_fps: float,
    batch_size: int,
    buffer_size: Optional[int],
    backend_pois: VectorBackendPOIs,
    backend_scores: VectorBackendScores,
    vectors_path_pois: Optional[Path],
    vectors_path_scores: Optional[Path],
    viz_path: Optional[Path],
) -> None:
    """
    Create a timelapse based on the most interesting parts of a video rather than blindly
    down-selecting frames. Adds cropping to a specific aspect ratio based on the contents of the
    video.

    \f

    :param input_files: See click docs.
    :param output_path: See click docs.
    :param duration: See click docs.
    :param output_fps: See click docs.
    :param batch_size: See click docs.
    :param buffer_size: See click docs.
    :param backend_pois: See click docs.
    :param backend_scores: See click docs.
    :param vectors_path_pois: See click docs.
    :param vectors_path_scores: See click docs.
    :param viz_path: See click docs.
    :return: None
    """

    cat_pipeline.create_cropped_timelapse(
        input_files=input_files,
        output_path=output_path,
        duration=duration,
        output_fps=output_fps,
        batch_size=batch_size,
        buffer_size=buffer_size,
        conversion_pois_functions=_CONVERSION_POIS_FUNCTIONS_LOOKUP[backend_pois],
        conversion_scoring_functions=_CONVERSION_SCORING_FUNCTIONS_LOOKUP[backend_scores],
        pois_vectors_path=vectors_path_pois,
        scores_vectors_path=vectors_path_scores,
        plot_path=viz_path,
    )


@cli.command(
    short_help=(
        "Evenly down-selects the input, taking every N frames until the "
        "desired output length is reached."
    )
)
@input_files_arg
@output_path_arg
@duration_arg
@output_fps_arg
def classic(  # pylint: disable=too-many-locals
    input_files: List[Path],
    output_path: Path,
    duration: float,
    output_fps: float,
) -> None:
    """
    Evenly down-selects the input, taking every N frames until the desired output length is reached.
    This is the classic timelapse method.

    \f

    :param input_files: See click docs.
    :param output_path: See click docs.
    :param duration: See click docs.
    :param output_fps: See click docs.
    :return: None
    """

    source_frames = cat_pipeline.load_input_videos(input_files=input_files)

    take_every = int(
        source_frames.total_frame_count
        / cat_pipeline.calculate_output_frames(duration=duration, output_fps=output_fps)
    )

    video_common.write_source_to_disk_consume(
        source=itertools.islice(
            tqdm(
                source_frames.frames,
                total=source_frames.total_frame_count,
                unit="Frames",
                ncols=100,
                desc="Scoring Images",
            ),
            None,
            None,
            take_every,
        ),
        video_path=output_path,
        video_fps=output_fps,
        high_quality=True,
    )


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
