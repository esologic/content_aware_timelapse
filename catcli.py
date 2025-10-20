"""Main module."""

import itertools
import logging
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import click

from content_aware_timelapse import cat_pipelines
from content_aware_timelapse.cli_common import create_enum_option
from content_aware_timelapse.frames_to_vectors.vector_computation.compute_vectors_clip import (
    CONVERT_CLIP,
)
from content_aware_timelapse.frames_to_vectors.vector_computation.compute_vectors_vit import (
    CONVERT_POIS_VIT_ATTENTION,
    CONVERT_SCORE_VIT_ATTENTION,
    CONVERT_SCORE_VIT_CLS,
)
from content_aware_timelapse.gpu_discovery import GPUDescription, discover_gpus
from content_aware_timelapse.viderator import video_common
from content_aware_timelapse.viderator.viderator_types import AspectRatio, AspectRatioParamType

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
    help="Desired duration of the output video in seconds.",
    required=True,
    default=30.0,
    show_default=True,
)

output_fps_arg = click.option(
    "--output-fps",
    "-f",
    type=click.FloatRange(min=1),
    help="Desired frames/second of the output video.",
    required=True,
    default=60.0,
    show_default=True,
)

# Content-aware parameters

buffer_size_arg = click.option(
    "--buffer-size",
    "-bu",
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

deselect_arg = click.option(
    "--deselect",
    "-de",
    type=click.IntRange(min=0),
    help="Frames surrounding high scores will be dropped by a radius that starts with this value.",
    required=False,
    default=1000,
    show_default=True,
)

audio_paths_arg = click.option(
    "--audio",
    "-a",
    type=click.Path(file_okay=True, exists=True, dir_okay=False, writable=True, path_type=Path),
    help="If given, these audio(s) will be added to the resulting video.",
    required=False,
    multiple=True,
)

gpus_arg = click.option(
    "--gpu",
    "-g",
    type=click.Choice(choices=discover_gpus()),
    help="The GPU(s) to use for computation. Can be given multiple times.",
    required=False,
    multiple=True,
)


_CONVERSION_SCORING_FUNCTIONS_LOOKUP = {
    VectorBackendScores.vit_cls: CONVERT_SCORE_VIT_CLS,
    VectorBackendScores.vit_attention: CONVERT_SCORE_VIT_ATTENTION,
    VectorBackendScores.clip: CONVERT_CLIP,
}


@click.group()
def cli() -> None:
    """
    Uses the contents of the frames in source files to create timelapses.

    \f

    :return: None
    """


@cli.command(
    short_help=(
        "Numerically scores the input frames based on their contents, "
        "then selects the best frames."
    )
)
@input_files_arg
@output_path_arg
@duration_arg
@output_fps_arg
@click.option(
    "--batch-size",
    "-ba",
    type=click.IntRange(min=1),
    help="Frames are sent to GPU for processing in batches of this size.",
    required=True,
    default=600,
    show_default=True,
)
@buffer_size_arg
@create_enum_option(
    arg_flag="--backend",
    help_message="Sets which vectorization backend is used to score the frames.",
    default=VectorBackendScores.vit_cls,
    input_enum=VectorBackendScores,
)
@deselect_arg
@audio_paths_arg
@click.option(
    "--vectors-path",
    "-v",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path),
    help="Intermediate vectors will be written to this path. Can be used to re-run.",
    required=False,
)
@viz_path_arg
@gpus_arg
@click.option(
    "--best-frame-path",
    "-bf",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path),
    help=(
        "If given, the highest scoring frame in the input will be written to this path. "
        "Good for thumbnails."
    ),
    required=False,
)
def content(  # pylint: disable=too-many-locals,too-many-positional-arguments,too-many-arguments
    input_files: List[Path],
    output_path: Path,
    duration: float,
    output_fps: float,
    batch_size: int,
    buffer_size: Optional[int],
    backend: VectorBackendScores,
    deselect: int,
    audio: List[Path],
    vectors_path: Optional[Path],
    viz_path: Optional[Path],
    best_frame_path: Optional[Path],
    gpu: Tuple[GPUDescription, ...],
) -> None:
    """
    Numerically scores the input frames based on their contents, then selects the best frames.

    \f

    :param input_files: See click docs.
    :param output_path: See click docs.
    :param duration: See click docs.
    :param output_fps: See click docs.
    :param batch_size: See click docs.
    :param buffer_size: See click docs.
    :param backend: See click docs.
    :param deselect: See click docs.
    :param audio: See click docs.
    :param vectors_path: See click docs.
    :param viz_path: See click docs.
    :param gpu: See click docs.
    :param best_frame_path: See click docs.
    :return: None
    """

    cat_pipelines.create_timelapse_score(
        input_files=input_files,
        output_path=output_path,
        duration=duration,
        output_fps=output_fps,
        batch_size=batch_size,
        buffer_size=buffer_size,
        conversion_scoring_functions=_CONVERSION_SCORING_FUNCTIONS_LOOKUP[backend],
        deselection_radius_frames=deselect,
        audio_paths=audio,
        vectors_path=vectors_path,
        plot_path=viz_path,
        gpus=gpu,
        best_frame_path=best_frame_path,
    )


_CONVERSION_POIS_FUNCTIONS_LOOKUP = {VectorBackendPOIs.vit_attention: CONVERT_POIS_VIT_ATTENTION}

ASPECT_RATIO: AspectRatioParamType = AspectRatioParamType()


@cli.command(
    short_help=(
        "Crops the input to the most interesting region, "
        "then selects the best frames of cropped region."
    )
)
@input_files_arg
@output_path_arg
@duration_arg
@output_fps_arg
@click.option(
    "--batch-size-pois",
    "-bp",
    type=click.IntRange(min=1),
    help=(
        "Scaled frames for Points of Interest calculation are sent to GPU for"
        " processing in batches of this size."
    ),
    required=True,
    default=600,
    show_default=True,
)
@click.option(
    "--batch-size-scores",
    "-bs",
    type=click.IntRange(min=1),
    help="Scaled frames for scoring are sent to GPU for processing in batches of this size.",
    required=True,
    default=600,
    show_default=True,
)
@buffer_size_arg
@create_enum_option(
    arg_flag="--backend-pois",
    help_message="Sets which Points of Interest discovery backend is used.",
    default=VectorBackendPOIs.vit_attention,
    input_enum=VectorBackendPOIs,
)
@create_enum_option(
    arg_flag="--backend-scores",
    help_message="Sets which scoring backend is used.",
    default=VectorBackendScores.vit_cls,
    input_enum=VectorBackendScores,
)
@click.option(
    "--aspect-ratio",
    "-r",
    type=ASPECT_RATIO,
    required=True,
    help="Aspect ratio in the format WIDTH:HEIGHT (e.g., 16:9, 4:3, 1.85:1).",
)
@deselect_arg
@audio_paths_arg
@click.option(
    "--save-cropped",
    "-sc",
    type=click.BOOL,
    help=(
        "The winning cropped region of the video is written to disk as a part of the process. "
        "Set this to save this video next to the output, otherwise a tempfile is used."
    ),
    required=False,
    default=True,
    show_default=True,
)
@click.option(
    "--vectors-path-pois",
    "-vp",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path),
    help="Intermediate POI vectors will be written to this path. Can be used to re-run.",
    required=False,
)
@click.option(
    "--vectors-path-scores",
    "-vs",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path),
    help="Intermediate scoring vectors will be written to this path. Can be used to re-run.",
    required=False,
)
@viz_path_arg
@gpus_arg
def content_cropped(  # pylint: disable=too-many-locals,too-many-positional-arguments,too-many-arguments
    input_files: List[Path],
    output_path: Path,
    duration: float,
    output_fps: float,
    batch_size_pois: int,
    batch_size_scores: int,
    buffer_size: Optional[int],
    backend_pois: VectorBackendPOIs,
    backend_scores: VectorBackendScores,
    aspect_ratio: AspectRatio,
    deselect: int,
    save_cropped: bool,
    audio: List[Path],
    vectors_path_pois: Optional[Path],
    vectors_path_scores: Optional[Path],
    viz_path: Optional[Path],
    gpu: Tuple[GPUDescription, ...],
) -> None:
    """
    Crops the input to the most interesting region, then selects the best frames of cropped region.

    \f

    :param input_files: See click docs.
    :param output_path: See click docs.
    :param duration: See click docs.
    :param output_fps: See click docs.
    :param batch_size_pois: See click docs.
    :param batch_size_scores: See click docs.
    :param buffer_size: See click docs.
    :param backend_pois: See click docs.
    :param backend_scores: See click docs.
    :param aspect_ratio: See click docs.
    :param deselect: See click docs.
    :param save_cropped: See click docs.
    :param audio: See click docs.
    :param vectors_path_pois: See click docs.
    :param vectors_path_scores: See click docs.
    :param viz_path: See click docs.
    :param gpu: See click docs.
    :return: None
    """

    cat_pipelines.create_timelapse_crop_score(
        input_files=input_files,
        output_path=output_path,
        duration=duration,
        output_fps=output_fps,
        batch_size_pois=batch_size_pois,
        batch_size_scores=batch_size_scores,
        scaled_frames_buffer_size=buffer_size,
        conversion_pois_functions=_CONVERSION_POIS_FUNCTIONS_LOOKUP[backend_pois],
        conversion_scoring_functions=_CONVERSION_SCORING_FUNCTIONS_LOOKUP[backend_scores],
        aspect_ratio=aspect_ratio,
        scoring_deselection_radius_frames=deselect,
        save_cropped_intermediate=save_cropped,
        audio_paths=audio,
        pois_vectors_path=vectors_path_pois,
        scores_vectors_path=vectors_path_scores,
        plot_path=viz_path,
        gpus=gpu,
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
@audio_paths_arg
def classic(  # pylint: disable=too-many-locals
    input_files: List[Path],
    output_path: Path,
    duration: float,
    output_fps: float,
    audio: List[Path],
) -> None:
    """
    Evenly down-selects the input, taking every N frames until the desired output length is reached.
    This is the classic timelapse method.

    \f

    :param input_files: See click docs.
    :param output_path: See click docs.
    :param duration: See click docs.
    :param output_fps: See click docs.
    :param audio: See click docs.
    :return: None
    """

    source_frames = cat_pipelines.load_input_videos(
        input_files=input_files, tqdm_desc="Reading Input Frames"
    )

    take_every = int(
        source_frames.total_frame_count
        / cat_pipelines.calculate_output_frames(duration=duration, output_fps=output_fps)
    )

    video_common.write_source_to_disk_consume(
        source=itertools.islice(
            source_frames.frames,
            None,
            None,
            take_every,
        ),
        video_path=output_path,
        video_fps=output_fps,
        high_quality=True,
        audio_paths=audio,
    )


if __name__ == "__main__":
    cli()  # pylint: disable=unused-argument
