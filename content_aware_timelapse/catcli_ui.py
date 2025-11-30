"""
CLI-specific functionality.
"""

import random
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Tuple, TypeVar

import click
from bonus_click.options import create_enum_option
from click import Context, Parameter
from click.decorators import FC
from click_option_group import MutuallyExclusiveOptionGroup, optgroup

from content_aware_timelapse.frames_to_vectors.conversion_types import (
    ConversionPOIsFunctions,
    ConversionScoringFunctions,
)
from content_aware_timelapse.frames_to_vectors.vector_computation.compute_vectors_clip import (
    CONVERT_CLIP,
)
from content_aware_timelapse.frames_to_vectors.vector_computation.compute_vectors_vit import (
    CONVERT_POIS_VIT_ATTENTION,
    CONVERT_SCORE_VIT_ATTENTION,
    CONVERT_SCORE_VIT_CLS,
)
from content_aware_timelapse.gpu_discovery import discover_gpus
from content_aware_timelapse.viderator.viderator_types import ImageResolutionParamType

# A generic type variable for Enum subclasses, essentially any enum subclass.
E = TypeVar("E", bound=Enum)
T = TypeVar("T")

ENV_VAR_PREFIX = "CAT"


def make_audio_option_group(name_prefix: str = "") -> Callable[[FC], FC]:
    """
    Defines the different methods to get set the audio that will be put under the resulting video.
    :param name_prefix: Prepended to the name the user will pass in for these options. Format is:
    `--{name_prefix}audio` / "--{name_prefix}audio-directory-random"
    :return: Function to become a click decorator that will produce the set of options.
    """

    def random_sort_audio_in_directory_callback(
        ctx: Context, _param: Parameter, value: Path
    ) -> None:
        """
        If the user provides a directory, collect all audio files within it,
        shuffle them randomly, and add the result to the "audio" option in the context.

        :param ctx: Click context
        :param _param: Click parameter (unused)
        :param value: Directory path provided via --random-audio-directory
        :return: None
        """

        if value is None:
            return None

        ctx.ensure_object(dict)

        # Gather supported audio files in the directory
        supported_exts = {".mp3", ".wav", ".flac", ".aiff", ".aac", ".ogg", ".m4a"}
        audio_files = [p for p in value.iterdir() if p.suffix.lower() in supported_exts]

        if not audio_files:
            raise click.BadParameter(f"No supported audio files found in directory: {value}")

        random.shuffle(audio_files)

        tuple_converted = tuple(audio_files)

        if ctx.params.get("audio", None) is None:
            ctx.params["audio"] = tuple_converted
        else:
            ctx.params["audio"] = ctx.params["audio"] + tuple_converted

        return None

    def add_paths_to_context(ctx: Context, _param: Parameter, value: Tuple[Path, ...]) -> None:
        """
        If the user provides a directory, collect all audio files within it,
        shuffle them randomly, and add the result to the "audio" option in the context.

        :param ctx: Click context
        :param _param: Click parameter (unused)
        :param value: Directory path provided via --random-audio-directory
        :return: None
        """

        ctx.ensure_object(dict)

        if ctx.params.get("audio", None) is None:
            ctx.params["audio"] = value
        else:
            ctx.params["audio"] = ctx.params["audio"] + value

    def decorator(command: FC) -> FC:
        """
        :param command: To wrap.
        :return: Decorated function.
        """

        group = optgroup.group(
            "Configures the audio that will be put under the resulting video",
            cls=MutuallyExclusiveOptionGroup,
        )

        for option in [
            optgroup.option(
                f"--{name_prefix}audio-directory-random",
                type=click.Path(
                    file_okay=False, exists=True, dir_okay=True, readable=True, path_type=Path
                ),
                help=(
                    "The audio files in this directory will be sorted in a "
                    "random order and added to resulting video."
                ),
                callback=random_sort_audio_in_directory_callback,
                envvar=f"{ENV_VAR_PREFIX}_AUDIO_DIRECTORY_RANDOM",
                expose_value=False,
                show_envvar=True,
            ),
            optgroup.option(
                f"--{name_prefix}audio",
                type=click.Path(
                    file_okay=True, exists=True, dir_okay=False, writable=True, path_type=Path
                ),
                help="If given, these audio(s) will be added to the resulting video.",
                multiple=True,
                envvar=f"{ENV_VAR_PREFIX}_AUDIO",
                callback=add_paths_to_context,
                expose_value=False,
                show_envvar=True,
            ),
        ]:
            option(command)

        group(command)

        return command

    return decorator


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


_CONVERSION_SCORING_FUNCTIONS_LOOKUP: Dict[VectorBackendScores, ConversionScoringFunctions] = {
    VectorBackendScores.vit_cls: CONVERT_SCORE_VIT_CLS,
    VectorBackendScores.vit_attention: CONVERT_SCORE_VIT_ATTENTION,
    VectorBackendScores.clip: CONVERT_CLIP,
}

_CONVERSION_POIS_FUNCTIONS_LOOKUP: Dict[VectorBackendPOIs, ConversionPOIsFunctions] = {
    VectorBackendPOIs.vit_attention: CONVERT_POIS_VIT_ATTENTION
}


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

output_resolution_arg = click.option(
    "--output-resolution",
    "-or",
    type=ImageResolutionParamType(),
    help="Desired resolution of the output video.",
    required=False,
    default=None,
    show_default=True,
    envvar=f"{ENV_VAR_PREFIX}_OUTPUT_RESOLUTION",
    show_envvar=True,
)

frame_buffer_size_arg = click.option(
    "--frame-buffer-size",
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
    envvar=f"{ENV_VAR_PREFIX}_FRAME_BUFFER_SIZE",
    show_envvar=True,
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
    envvar=f"{ENV_VAR_PREFIX}_DESELECT",
    show_envvar=True,
)

gpus_arg = click.option(
    "--gpu",
    "-g",
    type=click.Choice(choices=discover_gpus()),
    help="The GPU(s) to use for computation. Can be given multiple times.",
    required=False,
    multiple=True,
)

backend_pois_arg = create_enum_option(
    arg_flag="--backend-pois",
    help_message="Sets which Points of Interest discovery backend is used.",
    default=VectorBackendPOIs.vit_attention,
    input_enum=VectorBackendPOIs,
    lookup_fn=_CONVERSION_POIS_FUNCTIONS_LOOKUP.get,
    envvar=f"{ENV_VAR_PREFIX}_BACKEND_POIS",
)

backend_scores_arg = create_enum_option(
    arg_flag="--backend-scores",
    help_message="Sets which frame scoring backend is used.",
    default=VectorBackendScores.vit_cls,
    input_enum=VectorBackendScores,
    lookup_fn=_CONVERSION_SCORING_FUNCTIONS_LOOKUP.get,
    envvar=f"{ENV_VAR_PREFIX}_BACKEND_SCORES",
)

batch_size_pois_arg = click.option(
    "--batch-size-pois",
    "-bp",
    type=click.IntRange(min=1),
    help=(
        "Scaled frames for Points of Interest calculation are sent to GPU for "
        "processing in batches of this size."
    ),
    required=True,
    default=600,
    show_default=True,
    envvar=f"{ENV_VAR_PREFIX}_BATCH_POIS",
    show_envvar=True,
)

batch_size_scores_arg = click.option(
    "--batch-size-scores",
    "-bs",
    type=click.IntRange(min=1),
    help="Scaled frames for scoring are sent to GPU for processing in batches of this size.",
    required=True,
    default=600,
    show_default=True,
    envvar=f"{ENV_VAR_PREFIX}_BATCH_SCORES",
    show_envvar=True,
)

vectors_path_pois_arg = click.option(
    "--vectors-path-pois",
    "-vp",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path),
    help="Intermediate POI vectors will be written to this path. Can be used to re-run.",
    required=False,
)

vectors_path_scores_arg = click.option(
    "--vectors-path-scores",
    "-vs",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path),
    help="Intermediate scoring vectors will be written to this path. Can be used to re-run.",
    required=False,
)

resize_inputs_arg = click.option(
    "--resize-inputs",
    "-ri",
    type=click.BOOL,
    help=(
        "If inputs are of different resolutions, the larger videos are shrunk to the smallest "
        "input resolution for analysis."
    ),
    required=False,
    default=True,
    show_default=True,
    envvar=f"{ENV_VAR_PREFIX}_RESIZE_INPUTS",
    show_envvar=True,
)


def video_inputs_outputs_args() -> Callable[[FC], FC]:
    """
    Common set of args supported by all the video manipulation functions.
    :return: Function to become a click decorator that will produce the set of options.
    """

    def decorator(command: FC) -> FC:
        """
        :param command: To wrap.
        :return: Decorated function.
        """

        for option in [
            input_files_arg,
            output_path_arg,
            duration_arg,
            output_fps_arg,
            output_resolution_arg,
            make_audio_option_group(),
            resize_inputs_arg,
        ]:
            option(command)

        return command

    return decorator
