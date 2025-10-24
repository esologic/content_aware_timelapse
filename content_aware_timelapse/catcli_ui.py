"""
CLI-specific functionality.
"""

from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Optional, ParamSpec, TypeVar

import click
from click import Context, Parameter
from click.decorators import FC

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

# A generic type variable for Enum subclasses, essentially any enum subclass.
E = TypeVar("E", bound=Enum)


T = TypeVar("T")
P = ParamSpec("P")  # ParamSpec for the original function


def create_enum_option(
    arg_flag: str,
    help_message: str,
    default: E,
    input_enum: type[E],
    lookup_fn: Optional[Callable[[E], T]] = None,
) -> Callable[[FC], FC]:
    """
    Creates a Click option for an Enum type. Resulting input can be given as an index or as the
    string value from the enum.

    :param arg_flag: The argument flag for the Click option (e.g., "--cooler").
    :param help_message: Will be included in the --help message alongside the acceptable inputs
    to the Enum.
    :param default: The default value for the Click option, must be a member of `input_enum`.
    :param lookup_fn: If given, the resolved value will be passed to this function, then the click
    command will get whatever is returned as an argument.
    :param input_enum: The Enum class from which the option values are derived.
    :return: A Click option configured for the specified Enum.
    """

    try:
        input_enum(default)
    except ValueError as e:
        raise ValueError("Default value was not a member of the enum!") from e

    options_string = "\n".join(
        [f"   {idx}: {enum_member.value}" for idx, enum_member in enumerate(input_enum)]
    )

    help_string = (
        f"\b\n{help_message}\nOptions below. Either provide index or value:\n{options_string}"
    )

    def callback(_ctx: Context, _param: Parameter, value: str) -> "T | E":
        enum_options = list(input_enum)
        try:
            # Try interpreting as an index
            index = int(value)
            if 0 <= index < len(enum_options):
                return enum_options[index]
            else:
                raise click.BadParameter(
                    f"Index out of range. Valid range: 0-{len(enum_options)-1}."
                )
        except ValueError:
            # If not an index, validate as a string
            try:
                resolved: E = input_enum(value)
                return lookup_fn(resolved) if lookup_fn else resolved
            except ValueError as e:
                valid_choices = ", ".join([e.value for e in enum_options])
                raise click.BadParameter(
                    "Invalid choice. "
                    f"Valid names: {valid_choices}, or indices 0-{len(enum_options)-1}."
                ) from e

    return click.option(
        arg_flag,
        type=click.STRING,
        callback=callback,
        help=help_string,
        default=default.value,  # Ensure we use the string value for the default
        show_default=True,
    )


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

backend_pois_arg = create_enum_option(
    arg_flag="--backend-pois",
    help_message="Sets which Points of Interest discovery backend is used.",
    default=VectorBackendPOIs.vit_attention,
    input_enum=VectorBackendPOIs,
    lookup_fn=_CONVERSION_POIS_FUNCTIONS_LOOKUP.get,
)

backend_scores_arg = create_enum_option(
    arg_flag="--backend-scores",
    help_message="Sets which frame scoring backend is used.",
    default=VectorBackendScores.vit_cls,
    input_enum=VectorBackendScores,
    lookup_fn=_CONVERSION_SCORING_FUNCTIONS_LOOKUP.get,
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
)

batch_size_scores_arg = click.option(
    "--batch-size-scores",
    "-bs",
    type=click.IntRange(min=1),
    help="Scaled frames for scoring are sent to GPU for processing in batches of this size.",
    required=True,
    default=600,
    show_default=True,
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
