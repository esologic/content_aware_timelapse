"""
Main functionality, defines the pipeline.
"""

import itertools
import logging
from functools import partial
from pathlib import Path
from typing import Iterator, List, NamedTuple, Optional, Set, Tuple

import numpy as np
import pandas as pd
from numpy import typing as npt
from tqdm import tqdm

import content_aware_timelapse.viderator.frames_in_video
from content_aware_timelapse.frames_to_vectors import vector_scoring
from content_aware_timelapse.frames_to_vectors.conversion_types import (
    ConversionPOIsFunctions,
    ConversionScoringFunctions,
    IndexPointsOfInterest,
    IndexScores,
)
from content_aware_timelapse.vector_file import create_videos_signature
from content_aware_timelapse.viderator import (
    image_common,
    iterator_common,
    iterator_on_disk,
    video_common,
)
from content_aware_timelapse.viderator.video_common import VideoFrames
from content_aware_timelapse.viderator.viderator_types import (
    ImageResolution,
    ImageSourceType,
    RGBInt8ImageType,
    XYPoint,
)

LOGGER = logging.getLogger(__name__)


class _FramesCountResolution(NamedTuple):
    """
    Intermediate type for keeping track of the total number of frames in an iterator.
    """

    total_frame_count: int
    frames: ImageSourceType
    original_resolution: ImageResolution


def calculate_output_frames(duration: float, output_fps: float) -> int:
    """
    Canonical function to do this math.
    :param duration: Desired length of the video in seconds.
    :param output_fps: FPS of output.
    :return: Round number for output frames.
    """

    return int(duration * output_fps)


def load_input_videos(input_files: List[Path]) -> _FramesCountResolution:
    """
    Helper function to combine the input videos.
    :param input_files: List of input videos.
    :return: NT containing the total frame count and a joined iterator of all the input
    frames.
    """

    input_video_frames: List[VideoFrames] = list(
        map(content_aware_timelapse.viderator.frames_in_video.frames_in_video_opencv, input_files)
    )

    all_input_frames = itertools.chain.from_iterable(
        [video_frames.frames for video_frames in input_video_frames]
    )

    input_resolutions = [video_frames.original_resolution for video_frames in input_video_frames]

    if len(input_video_frames) > 1:
        raise ValueError("Input videos have different resolutions.")

    return _FramesCountResolution(
        total_frame_count=sum(
            (video_frames.total_frame_count for video_frames in input_video_frames)
        ),
        frames=all_input_frames,
        original_resolution=next(iter(input_resolutions)),
    )


def create_uncropped_timelapse(  # pylint: disable=too-many-locals,too-many-positional-arguments
    input_files: List[Path],
    output_path: Path,
    duration: float,
    output_fps: float,
    batch_size: int,
    buffer_size: int,
    conversion_scoring_functions: ConversionScoringFunctions,
    vectors_path: Optional[Path],
    plot_path: Optional[Path],
) -> None:
    """
    Main function to create the timelapse, defines the video processing pipeline.

    :param input_files: Videos from user to convert.
    :param output_path: Path to write the resulting output video to.
    :param duration: Desired duration for the output video in seconds.
    :param output_fps: Desired output video fps.
    :param batch_size: The number of frames from the input video to vectorize at once. The
    vectorization functions are responsible for doing the parallel loads onto the GPU.
    :param buffer_size: The number of frames to load into an in-memory buffer. This makes sure
    the GPUs have fast access to more frames rather than have the GPU waiting on disk/network IO.
    :param conversion_scoring_functions: A tuple of callables that contain the vectorization
    function and the function to score those vectors. Lets us swap the backend between different
    CV processes.
    :param vectors_path: Path to the vectors file. This file is an on-disk copy of each of the
    vectorization results for the input frames. This lets us recover from a run that has to be
    ended early, or lets us re-run selection without having to burn more compute.
    :param plot_path: If given, a visualization of the selection process is written to this file
    to aid in debugging.
    :return: None
    """

    input_signature = create_videos_signature(video_paths=input_files)

    frames_count_resolution = load_input_videos(input_files=input_files)

    LOGGER.info(f"Total frames to process: {frames_count_resolution.total_frame_count}.")

    frame_source = tqdm(
        frames_count_resolution.frames,
        total=frames_count_resolution.total_frame_count,
        unit="Frames",
        ncols=100,
        desc="Reading Images",
        maxinterval=1,
    )

    if conversion_scoring_functions.max_side_length is not None:
        frame_source = map(
            partial(
                image_common.resize_image_max_side,
                max_side_length=conversion_scoring_functions.max_side_length,
                delete=True,
            ),
            frame_source,
        )

    if buffer_size > 0:
        frame_source = iterator_common.preload_into_memory(
            source=frame_source, buffer_size=buffer_size, fill_buffer_before_yield=True
        )

    vectors: Iterator[npt.NDArray[np.float16]] = (
        content_aware_timelapse.frames_to_vectors.conversion.frames_to_vectors(
            frames=frame_source,
            intermediate_path=vectors_path,
            input_signature=input_signature,
            batch_size=batch_size,
            total_input_frames=frames_count_resolution.total_frame_count,
            convert_batches=conversion_scoring_functions.conversion,
        )
    )

    LOGGER.debug("Starting to sort output vectors by score.")

    score_indexes: List[IndexScores] = list(
        map(
            conversion_scoring_functions.scoring,
            tqdm(
                enumerate(vectors),
                total=frames_count_resolution.total_frame_count,
                unit="Frames",
                ncols=100,
                desc="Scoring Images",
                maxinterval=1,
            ),
        )
    )

    most_interesting_indices: Set[int] = set(
        vector_scoring.select_frames(
            score_weights=conversion_scoring_functions.weights,
            index_scores=score_indexes,
            num_output_frames=calculate_output_frames(duration=duration, output_fps=output_fps),
            plot_path=plot_path,
        )
    )

    final_frame_index: int = max(most_interesting_indices)

    # Slice the frames to include the frame at final_frame_index
    # Ensure the frame at final_frame_index is included, the plus one.
    sliced_frames: ImageSourceType = itertools.islice(
        load_input_videos(input_files=input_files).frames, None, final_frame_index + 1
    )

    most_interesting_frames: ImageSourceType = (
        index_frame[1]
        for index_frame in filter(
            lambda index_frame: index_frame[0] in most_interesting_indices,
            tqdm(
                enumerate(sliced_frames),
                total=final_frame_index + 1,
                unit="Frames",
                ncols=100,
                desc="Reading best frames for output",
            ),
        )
    )

    video_common.write_source_to_disk_consume(
        source=most_interesting_frames,
        video_path=output_path,
        video_fps=output_fps,
        high_quality=True,
    )


def video_multi_read(
    input_files: List[Path], resize_side_length: Optional[int], buffer_size: int, copies: int
) -> Tuple[Tuple[int, ImageResolution], Tuple[Iterator[RGBInt8ImageType], ...]]:
    """

    :param input_files:
    :param resize_side_length:
    :param buffer_size:
    :param copies:
    :return:
    """

    frames_count_resolution = load_input_videos(input_files=input_files)

    frame_source = frames_count_resolution.frames

    if resize_side_length is not None:
        frame_source = map(
            partial(
                image_common.resize_image_max_side,
                max_side_length=resize_side_length,
                delete=True,
            ),
            frame_source,
        )

    output_iterators = iterator_on_disk.tee_disk_cache(
        iterator=frame_source,
        copies=copies,
        serializer=iterator_on_disk.HDF5_SERIALIZER,
    )

    if buffer_size > 0:
        output_iterators = tuple(
            map(
                lambda iterator: iterator_common.preload_into_memory(
                    source=iterator, buffer_size=buffer_size, fill_buffer_before_yield=True
                ),
                output_iterators,
            )
        )

    return (
        frames_count_resolution.total_frame_count,
        frames_count_resolution.original_resolution,
    ), output_iterators


class PointFrameCount(NamedTuple):
    """
    Dynamic point with its frequency across frames.
    """

    point: XYPoint
    frame_count: int


def count_frames_filter_dynamic_points(
    points_of_interest: List[IndexPointsOfInterest], total_frame_count: int, threshold: float
) -> List[PointFrameCount]:
    """

    :param points_of_interest:
    :param total_frame_count:
    :param threshold:
    :return:
    """

    # Flatten to DataFrame
    df = pd.DataFrame.from_records(
        (
            (pt.x, pt.y, frame["frame_index"])
            for frame in points_of_interest
            for pt in frame["points_of_interest"]
        ),
        columns=["x", "y", "frame_index"],
    )

    # Count in how many frames each (x, y) appears
    frames_per_point = (
        df.groupby(["x", "y"])["frame_index"].nunique().reset_index(name="frame_count")
    )

    # Keep only dynamic points
    dynamic_points_df = frames_per_point[
        (frames_per_point["frame_count"] / total_frame_count) < threshold
    ]

    # Convert to list of DynamicPoint
    result: List[PointFrameCount] = [
        PointFrameCount(XYPoint(row.x, row.y), int(row.frame_count))
        for row in dynamic_points_df.itertuples(index=False)
    ]

    return result


def create_cropped_timelapse(  # pylint: disable=too-many-locals,too-many-positional-arguments,too-many-arguments,unused-argument,unused-variable
    input_files: List[Path],
    output_path: Path,
    duration: float,
    output_fps: float,
    batch_size: int,
    buffer_size: int,
    conversion_pois_functions: ConversionPOIsFunctions,
    conversion_scoring_functions: ConversionScoringFunctions,
    pois_vectors_path: Optional[Path],
    scores_vectors_path: Optional[Path],
    plot_path: Optional[Path],
) -> None:
    """

    :param input_files:
    :param output_path:
    :param duration:
    :param output_fps:
    :param batch_size:
    :param buffer_size:
    :param conversion_pois_functions:
    :param conversion_scoring_functions:
    :param pois_vectors_path:
    :param scores_vectors_path:
    :param plot_path:
    :return:
    """

    input_signature = create_videos_signature(video_paths=input_files)

    side_lengths = [
        conversion_scoring_functions.max_side_length,
        conversion_pois_functions.max_side_length,
    ]

    if len(set(side_lengths)) > 1:
        LOGGER.warning("Conversion functions have different intermediate side lengths.")

    # These will be resized, so we can't use this for output.
    (total_frame_count, original_source_resolution), (frames_for_pois, frames_for_scores) = (
        video_multi_read(
            input_files=input_files,
            resize_side_length=max(side_lengths),
            buffer_size=buffer_size,
            copies=1,
        )
    )

    vectors_for_pois: Iterator[npt.NDArray[np.float16]] = (
        content_aware_timelapse.frames_to_vectors.conversion.frames_to_vectors(
            frames=frames_for_pois,
            intermediate_path=pois_vectors_path,
            input_signature=input_signature,
            batch_size=batch_size,
            total_input_frames=total_frame_count,
            convert_batches=conversion_pois_functions.conversion,
        )
    )

    points_of_interest: List[IndexPointsOfInterest] = list(
        map(
            partial(
                conversion_pois_functions.compute_pois,
                original_source_resolution=original_source_resolution,
            ),
            tqdm(
                enumerate(vectors_for_pois),
                total=total_frame_count,
                unit="Frames",
                ncols=100,
                desc="Scoring Images",
                maxinterval=1,
            ),
        )
    )

    winning_points: List[PointFrameCount] = count_frames_filter_dynamic_points(
        points_of_interest=points_of_interest, total_frame_count=total_frame_count, threshold=0.7
    )

    print("stop")
