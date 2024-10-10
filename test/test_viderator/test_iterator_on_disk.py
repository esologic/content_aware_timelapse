"""
Test to make sure the cache functionality works as expected.
"""

from typing import List

import numpy as np
import pytest

from content_aware_timelapse.viderator.iterator_on_disk import tee_disk_cache


@pytest.mark.parametrize("copies", range(1, 4))
@pytest.mark.parametrize(
    "to_duplicate",
    [
        ["a", "screaming", "across", "the", "sky"],
        [0, 1, 2, 3],
    ],
)
def test_tee_disk_cache(to_duplicate: List[int | str], copies: int) -> None:
    """
    Test with a few different inputs, of type and length, make sure the resulting iterators are
    all the same.
    :param to_duplicate: Passed to function, this is the iterator to cache.
    :param copies: Passed to function, this is the number of copies to produce.
    :return: None
    """

    result = tee_disk_cache(iterator=iter(to_duplicate), copies=copies)
    primary = result[0]
    secondaries = result[1:]
    assert len(secondaries) == copies
    assert np.array_equal(to_duplicate, list(primary))
    for secondary in secondaries:
        values = list(secondary)
        assert np.array_equal(to_duplicate, values)
