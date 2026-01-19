"""Main module."""

import itertools
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import click
from click_option_group import RequiredMutuallyExclusiveOptionGroup, optgroup

from content_aware_timelapse import cat_pipelines, catcli_ui
from content_aware_timelapse.frames_to_vectors.conversion_types import (
    ConversionPOIsFunctions,
    ConversionScoringFunctions,
)
from content_aware_timelapse.gpu_discovery import GPUDescription, discover_gpus
from content_aware_timelapse.viderator import video_common
from content_aware_timelapse.viderator.viderator_types import (
    AspectRatio,
    AspectRatioParamType,
    ImageResolution,
    ImageResolutionParamType,
    UniqueIntMatrix2DParamType,
)

LOGGER_FORMAT = "[%(asctime)s - %(process)s - %(name)20s - %(levelname)s] %(message)s"
LOGGER_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOGGER_FORMAT,
    datefmt=LOGGER_DATE_FORMAT,
)

LOGGER = logging.getLogger(__name__)


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
@catcli_ui.batch_size_scores_arg
@catcli_ui.frame_buffer_size_arg
@catcli_ui.backend_scores_arg
@catcli_ui.deselect_arg
@catcli_ui.vectors_path_scores_arg
@catcli_ui.viz_path_arg
@catcli_ui.gpus_arg
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
@catcli_ui.video_inputs_outputs_args()
def content(  # pylint: disable=too-many-locals,too-many-positional-arguments,too-many-arguments
    input_files: List[Path],
    output_path: Path,
    duration: float,
    output_fps: float,
    output_resolution: Optional[ImageResolution],
    resize_inputs: bool,
    batch_size_scores: int,
    frame_buffer_size: Optional[int],
    backend_scores: ConversionScoringFunctions,
    deselect: int,
    audio: List[Path],
    vectors_path_scores: Optional[Path],
    viz_path: Optional[Path],
    best_frame_path: Optional[Path],
    gpu: Optional[Tuple[GPUDescription, ...]],
) -> None:
    """
    Numerically scores the input frames based on their contents, then selects the best frames.

    \f

    :param input_files: See click docs.
    :param output_path: See click docs.
    :param duration: See click docs.
    :param output_fps: See click docs.
    :param output_resolution: See click docs.
    :param resize_inputs: See click docs.
    :param batch_size_scores: See click docs.
    :param frame_buffer_size: See click docs.
    :param backend_scores: See click docs.
    :param deselect: See click docs.
    :param audio: See click docs.
    :param vectors_path_scores: See click docs.
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
        output_resolution=output_resolution,
        resize_inputs=resize_inputs,
        batch_size=batch_size_scores,
        buffer_size=frame_buffer_size,
        conversion_scoring_functions=backend_scores,
        deselection_radius_frames=deselect,
        audio_paths=audio,
        vectors_path=vectors_path_scores,
        plot_path=viz_path,
        gpus=gpu if gpu else discover_gpus(),
        best_frame_path=best_frame_path,
    )


@cli.command(
    short_help=(
        "Crops the input to the most interesting region, "
        "then selects the best frames of cropped region."
    )
)
@catcli_ui.batch_size_pois_arg
@catcli_ui.batch_size_scores_arg
@catcli_ui.frame_buffer_size_arg
@catcli_ui.backend_scores_arg
@catcli_ui.backend_pois_arg
@catcli_ui.deselect_arg
@catcli_ui.vectors_path_pois_arg
@catcli_ui.vectors_path_scores_arg
@catcli_ui.viz_path_arg
@catcli_ui.gpus_arg
@optgroup.group(
    "Crop Config",
    help="Configures how the video will be cropped.",
    cls=RequiredMutuallyExclusiveOptionGroup,
)
@optgroup.option(
    "--aspect-ratio",
    "-ar",
    type=AspectRatioParamType(),
    help="Crop by aspect ratio in the format WIDTH:HEIGHT (e.g., 16:9, 4:3, 1.85:1).",
    default=None,
)
@optgroup.option(
    "--crop-resolution",
    "-cr",
    type=ImageResolutionParamType(),
    help="Crops to the most interesting video section of this size.",
    default=None,
)
@click.option(
    "--layout",
    "-lo",
    type=UniqueIntMatrix2DParamType(),
    required=True,
    help=(
        "If given, this will cut the input into multiple high scoring regions and composite the"
        " input into a single output video. For example, setting the aspect ratio to 1:1 and "
        "then setting this variable to 1;0;2 will created a vertical video, three squares tall "
        "with the best scoring region in the center."
    ),
    default="0",
    show_default=True,
)
@catcli_ui.video_inputs_outputs_args()
def content_cropped(  # pylint: disable=too-many-locals,too-many-positional-arguments,too-many-arguments
    input_files: List[Path],
    output_path: Path,
    duration: float,
    output_fps: float,
    output_resolution: Optional[ImageResolution],
    resize_inputs: bool,
    batch_size_pois: int,
    batch_size_scores: int,
    frame_buffer_size: Optional[int],
    backend_pois: ConversionPOIsFunctions,
    backend_scores: ConversionScoringFunctions,
    aspect_ratio: AspectRatio,
    crop_resolution: ImageResolution,
    deselect: int,
    audio: List[Path],
    vectors_path_pois: Optional[Path],
    vectors_path_scores: Optional[Path],
    viz_path: Optional[Path],
    gpu: Optional[Tuple[GPUDescription, ...]],
    layout: List[List[int]],
) -> None:
    """
    Crops the input to the most interesting region, then selects the best frames of cropped region.

    \f

    :param input_files: See click docs.
    :param output_path: See click docs.
    :param duration: See click docs.
    :param output_fps: See click docs.
    :param output_resolution: See click docs.
    :param resize_inputs: See click docs.
    :param batch_size_pois: See click docs.
    :param batch_size_scores: See click docs.
    :param frame_buffer_size: See click docs.
    :param backend_pois: See click docs.
    :param backend_scores: See click docs.
    :param aspect_ratio: See click docs.
    :param crop_resolution: See click docs.
    :param deselect: See click docs.
    :param audio: See click docs.
    :param vectors_path_pois: See click docs.
    :param vectors_path_scores: See click docs.
    :param viz_path: See click docs.
    :param gpu: See click docs.
    :param layout: See click docs.
    :return: None
    """

    cat_pipelines.create_timelapse_crop_score(
        input_files=input_files,
        output_path=output_path,
        duration=duration,
        output_fps=output_fps,
        output_resolution=output_resolution,
        resize_inputs=resize_inputs,
        batch_size_pois=batch_size_pois,
        batch_size_scores=batch_size_scores,
        scaled_frames_buffer_size=frame_buffer_size,
        conversion_pois_functions=backend_pois,
        conversion_scoring_functions=backend_scores,
        aspect_ratio=aspect_ratio,
        crop_resolution=crop_resolution,
        scoring_deselection_radius_frames=deselect,
        audio_paths=audio,
        pois_vectors_path=vectors_path_pois,
        scores_vectors_path=vectors_path_scores,
        plot_path=viz_path,
        gpus=gpu if gpu else discover_gpus(),
        layout_matrix=layout,
    )


@cli.command(
    short_help=(
        "Evenly down-selects the input, taking every N frames until the "
        "desired output length is reached."
    )
)
@catcli_ui.video_inputs_outputs_args()
def classic(  # pylint: disable=too-many-locals,too-many-positional-arguments
    input_files: List[Path],
    output_path: Path,
    duration: float,
    output_fps: float,
    output_resolution: Optional[ImageResolution],
    resize_inputs: bool,
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
    :param output_resolution: See click docs.
    :param resize_inputs: See click docs.
    :param audio: See click docs.
    :return: None
    """

    source_frames = cat_pipelines.load_input_videos(
        input_files=input_files, tqdm_desc="Reading Input Frames", resize_inputs=resize_inputs
    )

    take_every = int(
        source_frames.total_frame_count
        / cat_pipelines.calculate_output_frames(duration=duration, output_fps=output_fps)
    )

    video_common.write_source_to_disk_consume(
        source=cat_pipelines.optionally_resize(
            output_resolution=output_resolution,
            source=itertools.islice(
                source_frames.frames,
                None,
                None,
                take_every,
            ),
        ),
        video_path=output_path,
        video_fps=output_fps,
        high_quality=True,
        audio_paths=audio,
    )


if __name__ == "__main__":
    cli()  # pylint: disable=unused-argument
