"""
Main functionality, defines the pipeline.
"""

import itertools
import logging
from pathlib import Path
from typing import Iterator, List, NamedTuple, Optional, Set

import numpy as np
from numpy import typing as npt
from tqdm import tqdm

import content_aware_timelapse.frames_to_vectors.conversion
import content_aware_timelapse.frames_to_vectors.vector_computation.compute_vectors_vit
from content_aware_timelapse.frames_to_vectors import vector_scoring
from content_aware_timelapse.frames_to_vectors.conversion_types import (
    ConversionScoringFunctions,
    IndexScores,
)
from content_aware_timelapse.vector_file import create_videos_signature
from content_aware_timelapse.viderator import iterator_common, video_common
from content_aware_timelapse.viderator.video_common import VideoFrames
from content_aware_timelapse.viderator.viderator_types import ImageSourceType

LOGGER = logging.getLogger(__name__)


class _FramesCount(NamedTuple):
    """
    Intermediate type for keeping track of the total number of frames in an iterator.
    """

    total_frame_count: int
    frames: ImageSourceType


def calculate_output_frames(duration: float, output_fps: float) -> int:
    """
    Canonical function to do this math.
    :param duration: Desired length of the video in seconds.
    :param output_fps: FPS of output.
    :return: Round number for output frames.
    """

    return int(duration * output_fps)


def load_input_videos(input_files: List[Path]) -> _FramesCount:
    """
    Helper function to combine the input videos.
    :param input_files: List of input videos.
    :return: NT containing the total frame count and a joined iterator of all the input
    frames.
    """

    input_video_frames: List[VideoFrames] = list(
        map(video_common.frames_in_video_opencv, input_files)
    )

    all_input_frames = itertools.chain.from_iterable(
        [video_frames.frames for video_frames in input_video_frames]
    )

    return _FramesCount(
        total_frame_count=sum(
            (video_frames.total_frame_count for video_frames in input_video_frames)
        ),
        frames=all_input_frames,
    )


def create_timelapse(  # pylint: disable=too-many-locals
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

    frames_count = load_input_videos(input_files=input_files)

    LOGGER.info(f"Total frames to process: {frames_count.total_frame_count}.")

    if buffer_size > 0:
        frame_source = iterator_common.preload_into_memory(
            source=frames_count.frames,
            buffer_size=buffer_size,
        )
    else:
        frame_source = frames_count.frames

    vectors: Iterator[npt.NDArray[np.float16]] = (
        content_aware_timelapse.frames_to_vectors.conversion.frames_to_vectors(
            frames=frame_source,
            intermediate_path=vectors_path,
            input_signature=input_signature,
            batch_size=batch_size,
            total_input_frames=frames_count.total_frame_count,
            convert_batches=conversion_scoring_functions.conversion,
        )
    )

    LOGGER.debug("Starting to sort output vectors by score.")

    score_indexes: List[IndexScores] = list(
        map(
            conversion_scoring_functions.scoring,
            tqdm(
                enumerate(vectors),
                total=frames_count.total_frame_count,
                unit="Frames",
                ncols=100,
                desc="Scoring Images",
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
