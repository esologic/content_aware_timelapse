"""
Test to make sure the cache functionality works as expected.
"""

import time
from itertools import tee
from test.assets import LONG_TEST_VIDEO_PATH, SAMPLE_TIMELAPSE_INPUT_PATH
from test.test_viderator import viderator_test_common
from typing import List, Optional

import numpy as np
import pytest

from content_aware_timelapse import timeout_common
from content_aware_timelapse.viderator import frames_in_video, video_common
from content_aware_timelapse.viderator.iterator_on_disk import (
    HDF5_COMPRESSED_SERIALIZER,
    HDF5_SERIALIZER,
    PICKLE_SERIALIZER,
    Serializer,
    disk_buffer,
    tee_disk_cache,
    video_file_tee,
)
from content_aware_timelapse.viderator.viderator_types import ImageResolution


def run_disk_cache_test(
    to_duplicate: List[int | str], copies: int, serializer: Optional[Serializer] = None
) -> None:
    """
    Helper to setup and run the test.
    :param to_duplicate: Param.
    :param copies:  Param.
    :param serializer: Param.
    :return: None
    """

    if serializer is None:
        result = tee_disk_cache(iterator=iter(to_duplicate), copies=copies)
    else:
        result = tee_disk_cache(iterator=iter(to_duplicate), copies=copies, serializer=serializer)

    primary = result[0]
    secondaries = result[1:]
    assert len(secondaries) == copies
    assert np.array_equal(to_duplicate, list(primary))
    for secondary in secondaries:
        values = list(secondary)
        assert np.array_equal(to_duplicate, values)


@pytest.mark.integration
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
    run_disk_cache_test(to_duplicate, copies)


@pytest.mark.integration
@pytest.mark.parametrize(
    "serializer", [HDF5_COMPRESSED_SERIALIZER, HDF5_SERIALIZER, PICKLE_SERIALIZER]
)
@pytest.mark.parametrize("copies", range(1, 4))
@pytest.mark.parametrize(
    "to_duplicate",
    [
        list(
            viderator_test_common.create_random_frames_iterator(
                image_resolution=ImageResolution(100, 200), count=50
            )
        ),
    ],
)
def test_tee_disk_cache_frames(
    serializer: Serializer, to_duplicate: List[int | str], copies: int
) -> None:
    """
    Specific UT for frames and serialization.
    param serializer: Serializer to use.
    :param to_duplicate: Passed to function, this is the iterator to cache.
    :param copies: Passed to function, this is the number of copies to produce.
    :return: None
    """

    run_disk_cache_test(to_duplicate, copies, serializer)


def test_disk_buffer() -> None:
    """
    Test to make sure the input of the disk buffer function maintains the original iterator.
    :return: None
    """

    source, expected_output = tee(
        frames_in_video.frames_in_video_opencv(
            video_path=SAMPLE_TIMELAPSE_INPUT_PATH,
        ).frames,
        2,
    )

    buffered_iterator_list = list(disk_buffer(source=source, buffer_size=100))
    expected_output_list = list(expected_output)

    assert len(buffered_iterator_list) == len(expected_output_list)

    for buffered_frame, input_frame in zip(buffered_iterator_list, expected_output_list):
        assert np.array_equal(buffered_frame, input_frame)


@pytest.mark.skip()
def test_disk_buffer_speed() -> None:
    """
    Not really a test, just a way to see how much performance we get with the disk buffer approach.
    :return: None
    """

    source = viderator_test_common.create_random_frames_iterator(
        image_resolution=ImageResolution(100, 200), count=50
    )

    buffered_iterator = disk_buffer(source=source, buffer_size=10000)

    # Simulates chunks being pulled through the pipeline.
    time.sleep(5)

    _buffered_frames, buffered_time = timeout_common.measure_execution_time(list, buffered_iterator)
    _opencv_frames, opencv_time = timeout_common.measure_execution_time(
        list,
        frames_in_video.frames_in_video_opencv(
            video_path=LONG_TEST_VIDEO_PATH,
        ).frames,
    )

    print(f"buffered time: {buffered_time}, opencv time: {opencv_time}")


@pytest.mark.integration
@pytest.mark.parametrize(
    "image_resolution,frame_count",
    [(ImageResolution(100, 100), 200), (ImageResolution(10, 10), 300_000)],
)
def test_video_file_tee(image_resolution: ImageResolution, frame_count: int) -> None:
    """
    Test to make sure going to disk with a video intermediate produces the same video.
    :param image_resolution: Resolution under test.
    :param frame_count: Frame count, varied to make sure big/small videos don't cause problems.
    :return: None
    """

    with video_common.video_safe_temp_path() as video_safe_temp_path:

        for copied_iterator in video_file_tee(
            source=viderator_test_common.create_random_frames_iterator(
                image_resolution=image_resolution, count=frame_count
            ),
            copies=2,
            video_fps=30,
            intermediate_video_path=video_safe_temp_path,
        ):
            assert sum(1 for _ in copied_iterator) == frame_count
