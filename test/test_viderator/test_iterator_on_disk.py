"""
Test to make sure the cache functionality works as expected.
"""

import time
from itertools import tee
from test.assets import LONG_TEST_VIDEO_PATH, SAMPLE_TIMELAPSE_INPUT_PATH
from typing import List

import numpy as np
import pytest

from content_aware_timelapse import timeout_common
from content_aware_timelapse.viderator import video_common
from content_aware_timelapse.viderator.iterator_on_disk import disk_buffer, tee_disk_cache


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


def test_disk_buffer() -> None:
    """
    Test to make sure the input of the disk buffer function maintains the original iterator.
    :return: None
    """

    source, expected_output = tee(
        video_common.frames_in_video_opencv(
            video_path=SAMPLE_TIMELAPSE_INPUT_PATH,
        ).frames,
        2,
    )

    buffered_iterator_list = list(disk_buffer(source=source, buffer_size=100))
    expected_output_list = list(expected_output)

    assert len(buffered_iterator_list) == len(expected_output_list)

    for buffered_frame, input_frame in zip(buffered_iterator_list, expected_output_list):
        assert np.array_equal(buffered_frame, input_frame)


@pytest.mark.skip
def test_disk_buffer_speed() -> None:
    """
    Not really a test, just a way to see how much performance we get with the disk buffer approach.
    :return: None
    """

    source = video_common.frames_in_video_opencv(
        video_path=LONG_TEST_VIDEO_PATH,
    ).frames

    buffered_iterator = disk_buffer(source=source, buffer_size=10000)

    # Simulates chunks being pulled through the pipeline.
    time.sleep(300)

    _buffered_frames, buffered_time = timeout_common.measure_execution_time(list, buffered_iterator)
    _opencv_frames, opencv_time = timeout_common.measure_execution_time(
        list,
        video_common.frames_in_video_opencv(
            video_path=LONG_TEST_VIDEO_PATH,
        ).frames,
    )

    print(f"buffered time: {buffered_time}, opencv time: {opencv_time}")
