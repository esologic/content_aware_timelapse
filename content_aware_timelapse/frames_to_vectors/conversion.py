"""
Contains the frames to vectors pipeline, the actual conversion functions are standalone modules.
"""

import itertools
import logging
from pathlib import Path
from typing import Iterator, NamedTuple, Optional

import more_itertools
import numpy as np
from numpy import typing as npt

from content_aware_timelapse import vector_file
from content_aware_timelapse.frames_to_vectors.conversion_types import ConvertBatchesFunction
from content_aware_timelapse.viderator.viderator_types import ImageSourceType

LOGGER = logging.getLogger(__name__)


class IntermediateFileInfo(NamedTuple):
    """
    Fields needed to define an intermediate vector file.
    """

    path: Path
    signature: str


def frames_to_vectors(  # pylint:disable=too-many-positional-arguments
    frames: ImageSourceType,
    intermediate_info: Optional[IntermediateFileInfo],
    batch_size: int,
    total_input_frames: int,
    convert_batches: ConvertBatchesFunction,
) -> Iterator[npt.NDArray[np.float16]]:
    """
    Computes feature vectors from an input iterator or frames.
    Because this process is expensive, even with GPU, the intermediate vectors are written to disk
    to avoid re-doing the work.
    :param frames: Frames to process.
    :param intermediate_info: Describes the intermediate vector file, if not given it will not
    be used.
    intermediate vectors from the input file.
    :param batch_size: Number of frames to process at once. Should try to utilize all GPU memory.
    :param total_input_frames: Number of frames in `frames`.
    :param convert_batches: Function for going from a batch of images to vectors.
    :return: Vectors, one per input frame.
    """

    if intermediate_info is not None:

        intermediate = vector_file.read_vector_file(
            vector_file=intermediate_info.path, input_signature=intermediate_info.signature
        )

        LOGGER.info(f"Read in {intermediate.length} intermediate vectors from file.")

        fresh_tensors: Iterator[npt.NDArray[np.float16]] = iter([])

        if intermediate.length < total_input_frames:

            LOGGER.info(f"Need to compute {total_input_frames-intermediate.length} new vectors.")

            # Skip to the unprocessed section of the input.
            unprocessed_frames: ImageSourceType = itertools.islice(
                frames, intermediate.length, None
            )

            # Compute new vectors, writing the results to disk.
            fresh_tensors = convert_batches(
                frame_batches=more_itertools.chunked(unprocessed_frames, batch_size)
            )

            fresh_tensors = vector_file.write_vector_file_forward(
                vector_iterator=fresh_tensors,
                vector_file=intermediate_info.path,
                input_signature=intermediate_info.signature,
            )

        yield from itertools.chain.from_iterable(
            (
                intermediate.iterator,  # first output any vectors from disk.
                fresh_tensors,  # second, compute any vectors not found on disk.
            )
        )

    else:
        yield from convert_batches(frame_batches=more_itertools.chunked(frames, batch_size))
