"""
Because computing vectors can only happen with GPUs, which is expensive, we want to cache the
results of these operations on disk as hdf5 files, known within the application as "vector files".

This module relates to reading/writing these files.
"""

import hashlib
import io
import itertools
import json
import logging
import re
from pathlib import Path
from typing import Iterator, List, NamedTuple, Tuple

import h5py
import numpy as np
from numpy import typing as npt

LOGGER = logging.getLogger(__name__)

SIGNATURE_ATTRIBUTE_NAME = "signature"
VERSION_ATTRIBUTE_NAME = "vector_file_version"
CURRENT_VECTOR_FILE_VERSION = "1.0.0"


def create_videos_signature(video_paths: List[Path]) -> str:
    """
    Used to describe the contents of a vector file. Becomes an attribute in the HDF5 file.
    :param video_paths: Paths to the videos in the vector file.
    :return: A string, ready to be input to the HDF5 file.
    """

    def compute_partial_hash(video_path: Path) -> str:
        """
        Hashes the first 512KB of a video file.
        :param video_path: Path to the video.
        :return: Hash digest as a string.
        """

        sha256 = hashlib.sha256()

        with video_path.open("rb") as f:
            sha256.update(f.read(512_000))

        return sha256.hexdigest()

    # Convert dictionary to JSON string
    return json.dumps(
        {
            video.name: json.dumps(
                {
                    "file_size": video.stat().st_size,
                    "partial_sha256": compute_partial_hash(video),
                }
            )
            for video in sorted(video_paths, key=str)
        }
    )


class _LengthIterator(NamedTuple):
    """
    Intermediate type.
    Links the count of vectors on disk, in the file with an iterator that will emit those vectors.
    """

    length: int
    iterator: Iterator[npt.NDArray[np.float16]]


def _sort_groups_by_index(group_names: List[str]) -> List[str]:
    """
    Sort frame groups by the integer part of their names to preserve the correct order.
    Assumes frame names are in the format of "frame_<index>", where index is an integer.

    :param group_names: List of group names (strings) to sort.
    :return: List of group names sorted by their integer index.
    """
    # Extract the integer part from the frame names and sort by it
    return sorted(group_names, key=lambda x: int(re.search(r"(\d+)", x).group(1)))


def read_vector_file(vector_file: Path, input_signature: str) -> _LengthIterator:
    """
    Reads vectors from an HDF5 "vector file" as an iterator. When the iterator in the output is
    exhausted, it is auto-closed.

    If the input file is empty, the resulting count will be zero and the iterator will have
    no items in it.

    :param vector_file: Path to the HDF5 file.
    :param input_signature: Signature to validate against.
    :return: NT containing the number of vectors on disk and an iterator that produces them.
    """

    f = h5py.File(vector_file, "a")

    def iterate_from_file() -> Iterator[npt.NDArray[np.float16]]:
        """
        Yields each dataset in the frame groups.
        :return: An iterator of the datasets.
        """
        try:
            # Sort the group names using _sort_groups_by_index to preserve integer order
            for frame_name in _sort_groups_by_index(group_names=list(f.keys())):
                frame_group = f[frame_name]

                for dataset_name in frame_group:
                    dataset = frame_group[dataset_name]

                    # Ensure the dtype matches
                    yield dataset[()]
        finally:
            # Automatically close the file when iteration is done.
            f.close()

    # Validate the signature and check for frame groups
    if (
        SIGNATURE_ATTRIBUTE_NAME in f.attrs.keys()
        and f.attrs[SIGNATURE_ATTRIBUTE_NAME] == input_signature
        and len(f.keys())  # Ensure there are frame groups present
    ):
        completed_vectors = sum(
            len(f[frame_group].keys())
            for frame_group in f.keys()  # pylint: disable=consider-using-dict-items
        )

        return _LengthIterator(
            length=completed_vectors,
            iterator=iterate_from_file(),
        )
    else:
        # Close the file if the signature doesn't match or the file is empty
        f.close()
        return _LengthIterator(
            length=0,
            iterator=iter([]),
        )


def _setup_vector_file_metadata(f: h5py.File, input_signature: str) -> int:
    """
    Sets up the HDF5 file metadata for vector storage, ensuring compatibility with the input
    signature.

    :param f: HDF5 file object.
    :param input_signature: Signature string for validation or initialization.
    :return: The starting index for writing new frames.
    """
    # Determine starting index based on the existing frame groups
    if (
        SIGNATURE_ATTRIBUTE_NAME in f.attrs
        and f.attrs[SIGNATURE_ATTRIBUTE_NAME] == input_signature
        and len(f.keys()) > 0
    ):
        # Extract existing frame group names and find the highest index
        existing_indices = [
            int(frame_name.split("_")[1])
            for frame_name in f.keys()
            if frame_name.startswith("frame_")
        ]
        starting_index = max(existing_indices) + 1 if existing_indices else 0

    elif (
        SIGNATURE_ATTRIBUTE_NAME in f.attrs
        and f.attrs[SIGNATURE_ATTRIBUTE_NAME] != input_signature
        and len(f.keys()) > 0
    ):
        # If the signature doesn't match but the file is populated, raise an error
        raise ValueError("Can't write to vector file! Signature does not match.")

    else:
        # Initialize metadata for a new file
        f.attrs[SIGNATURE_ATTRIBUTE_NAME] = input_signature
        f.attrs[VERSION_ATTRIBUTE_NAME] = CURRENT_VECTOR_FILE_VERSION
        starting_index = 0

    return starting_index


def _create_hdf5_worker(
    index: str, vector: npt.NDArray[np.float16]
) -> Tuple[str, bytes, npt.NDArray[np.float16]]:
    """
    Create an in-memory HDF5 file of the input vector and return the BytesIO object containing that
    vector.
    :param index: The index of the frame of the input video that produced these vectors.
    :param vector: The vector to export.
    :return: A Tuple (The index forwarded from the input, the BytesIO containing the vector).
    """

    with io.BytesIO() as bytes_io:
        with h5py.File(bytes_io, "w") as f:
            f.create_dataset(
                name=f"vit_base_patch16_224_{index}",
                shape=vector.shape,
                dtype=vector.dtype,
                data=vector,
                compression="gzip",
                compression_opts=9,
                chunks=True,
                maxshape=vector.shape,
            )
            f.flush()

        output = index, bytes_io.getvalue(), vector

    return output


def write_vector_file_forward(
    vector_iterator: Iterator[npt.NDArray[np.float16]],
    vector_file: Path,
    input_signature: str,
) -> Iterator[npt.NDArray[np.float16]]:
    """
    Creates a new iterator that is a copy of the input iterator. Each time the new iterator is
    iterated upon, that file is written to disk in the input file.
    :param vector_iterator: Input vectors.
    :param vector_file: Path to the file to write.
    :param input_signature: Added to the newly created vector file as an attribute.
    :return: The input vectors in their original order. Now they've been written to disk.
    """

    with h5py.File(vector_file, "a") as f:

        starting_index = _setup_vector_file_metadata(f, input_signature)

        LOGGER.info(f"Starting to write vectors to: {vector_file}")

        indices: Iterator[str] = map(str, itertools.count(starting_index))

        results: Iterator[Tuple[str, npt.NDArray[np.float16]]] = zip(indices, vector_iterator)

        for index, vector in results:

            # New groups are created to be able to hold different datasets created for
            # each frame. For superstition reasons this also might prevent memory leaks.
            index_group = f.create_group(f"frame_{index}")

            index_group.create_dataset(
                name=f"vit_base_patch16_224_{index}",
                shape=vector.shape,
                dtype=vector.dtype,
                data=vector,
            )

            # Flushing is still recommended for safety
            f.flush()

            yield vector
