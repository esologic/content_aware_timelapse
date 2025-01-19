"""
Test writing/reading vector files.
"""

from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import numpy.typing as npt
import pytest

from content_aware_timelapse import vector_file


def random_vector_iterator(
    num_vectors: int, vector_shape: Tuple[int, ...], dtype: np.dtype  # type: ignore[type-arg]
) -> Iterator[npt.NDArray[np.floating]]:  # type: ignore[type-arg]
    """
    Generates an iterator of random vectors with the given shape.
    :param num_vectors: Number of vectors to generate.
    :param vector_shape: Shape of each vector (can be 2D or 3D).
    :param dtype: Data type of the vectors.
    :return: An iterator producing random vectors.
    """
    rng = np.random.default_rng(seed=42)  # Use a fixed seed for reproducibility
    return (rng.random(vector_shape, dtype=dtype) for _ in range(num_vectors))


@pytest.mark.parametrize(
    "num_vectors, vector_shape",
    [
        (10, (128,)),  # 2D vectors of shape (128,)
        (20, (64,)),  # 2D vectors of shape (64,)
        (5, (256,)),  # 2D vectors of shape (256,)
        (10, (64, 64)),  # 3D vectors of shape (64, 64)
        (5, (128, 128)),  # 3D vectors of shape (128, 128)
        (3, (32, 32, 32)),  # 3D vectors of shape (32, 32, 32)
        (200, (10,)),  # Really long vectors
        (3, (2000, 2000)),  # Really huge vectors.
    ],
)
@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64]  # Change type between float32 and float64
)
def test_write_and_read_vectors(
    tmpdir: str,
    num_vectors: int,
    vector_shape: Tuple[int, ...],
    dtype: np.dtype,  # type: ignore[type-arg]
) -> None:
    """
    Test writing and reading vectors using the HDF5 writer and reader functions with tmpdir.
    :param tmpdir: Pytest's temporary directory fixture.
    :param num_vectors: Number of vectors to generate.
    :param vector_shape: Shape of each vector (can be 2D or 3D).
    :param dtype: Data type of the vectors.
    :return: None
    """
    # Create a temporary HDF5 file in the tmpdir
    test_vector_file = Path(tmpdir).joinpath("test_vectors.h5")

    # Create the random vector iterator and convert it to a list for comparison
    random_vectors = list(random_vector_iterator(num_vectors, vector_shape, dtype))

    # Write the vectors to the file
    written_vectors: List[npt.NDArray[np.floating]] = list(  # type: ignore[type-arg]
        vector_file.write_vector_file_forward(
            vector_iterator=iter(random_vectors),
            vector_file=test_vector_file,
            input_signature="test_signature",
        )
    )

    result = vector_file.read_vector_file(
        vector_file=test_vector_file, input_signature="test_signature"
    )

    read_vectors = list(result.iterator)

    # Ensure the same number of vectors were written and read
    assert len(written_vectors) == len(read_vectors) == len(random_vectors)

    # Ensure all vectors match and identify whether the issue is in writing or reading
    for written, read, actual in zip(written_vectors, read_vectors, random_vectors):
        np.testing.assert_array_equal(written, actual)
        np.testing.assert_array_equal(read, actual)
