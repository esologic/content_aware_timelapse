"""
Main functionality, defines the pipeline.
"""

import itertools
import logging
from pathlib import Path
from typing import Iterator, List, NamedTuple, Optional, Set, Tuple

import numpy as np
from numpy import typing as npt
from tqdm import tqdm

from content_aware_timelapse import frames_to_vectors
from content_aware_timelapse.frames_to_vectors import create_videos_signature
from content_aware_timelapse.viderator import iterator_common, video_common
from content_aware_timelapse.viderator.image_common import ImageSourceType
from content_aware_timelapse.viderator.video_common import VideoFrames

LOGGER = logging.getLogger(__name__)


class _FramesCount(NamedTuple):
    """
    Intermediate type for keeping track of the total number of frames in an iterator.
    """

    total_frame_count: int
    frames: ImageSourceType


class _ScoreIndex(NamedTuple):
    """
    Intermediate type for linking the Euclidean distance between the next frame and the index
    of the frame.
    """

    score: float
    idx: int


def calculate_score(packed: Tuple[int, npt.NDArray[np.float16]]) -> _ScoreIndex:
    """
    Calculate a combined score from various metrics of the attention map.

    This version focuses on:
        * entropy
        * saliency
        * variance

    Without comparing directionality to a previous map.

    :param packed: Arguments packed as a tuple. First item is the index of the vector, second item
    is the vector.
    :return: Combined score.
    """

    index, attention_map = packed

    # Convert to float32 to prevent overflow during calculations
    attention_map = attention_map.astype(np.float32)

    # Normalize the attention map
    # normalized_map = attention_map / (np.sum(attention_map) + 1e-8)

    # Filter out zeros to avoid log(0) issues in entropy calculation
    # nonzero_map = normalized_map[normalized_map > 0]

    # Entropy-based score (only for non-zero elements)
    # _ = -np.sum(nonzero_map * np.log(nonzero_map + 1e-8))

    # Variance-based score (how spread out the attention is)
    variance_score = np.var(attention_map)
    saliency_score = np.max(attention_map)

    combined_score = 0.5 * variance_score + 0.5 * saliency_score

    return _ScoreIndex(score=float(combined_score), idx=index)


def create_timelapse(  # pylint: disable=too-many-locals
    input_files: List[Path],
    output_path: Path,
    duration: float,
    output_fps: float,
    batch_size: int,
    vectors_path: Optional[Path],
) -> None:
    """
    Create a timelapse based on the most interesting parts of a video rather than blindly
    down-selecting frames.

    \f

    :param input_files: See click docs.
    :param output_path: See click docs.
    :param duration: See click docs.
    :param output_fps: See click docs.
    :param batch_size: See click docs.
    :param vectors_path: See click docs.
    :return: None
    """

    input_signature = create_videos_signature(video_paths=input_files)

    def load_input_videos() -> _FramesCount:
        """
        Helper function to combine the input videos.
        :return: NT containing the total frame count and a joined iterator of all the input
        frames.
        """

        input_video_frames: List[VideoFrames] = list(map(video_common.frames_in_video, input_files))

        all_input_frames = itertools.chain.from_iterable(
            [video_frames.frames for video_frames in input_video_frames]
        )

        return _FramesCount(
            total_frame_count=sum(
                (video_frames.total_frame_count for video_frames in input_video_frames)
            ),
            frames=all_input_frames,
        )

    frames_count = load_input_videos()

    LOGGER.info(f"Total frames to process: {frames_count.total_frame_count}.")

    vectors: Iterator[npt.NDArray[np.float16]] = frames_to_vectors.frames_to_vectors(
        frames=iterator_common.preload_into_memory(
            source=frames_count.frames, buffer_size=batch_size * 3
        ),
        intermediate_path=vectors_path,
        input_signature=input_signature,
        batch_size=batch_size,
        total_input_frames=frames_count.total_frame_count,
    )

    LOGGER.debug("Starting to sort output vectors by score.")

    # Use imap_unordered for parallel processing and tqdm for progress bar
    score_indexes: Iterator[_ScoreIndex] = map(
        calculate_score,
        tqdm(
            enumerate(vectors),
            total=frames_count.total_frame_count,
            unit="Frames",
            ncols=100,
            desc="Scoring Images",
        ),
    )

    sorted_by_score: List[_ScoreIndex] = sorted(
        score_indexes, key=lambda distance_index: distance_index.score, reverse=True
    )

    most_interesting_indices: Set[int] = set(
        map(
            lambda distinct_index: distinct_index.idx, sorted_by_score[: int(duration * output_fps)]
        )
    )

    final_frame_index: int = max(most_interesting_indices)

    # Slice the frames to include the frame at final_frame_index
    # Ensure the frame at final_frame_index is included, the plus one.
    sliced_frames: ImageSourceType = itertools.islice(
        load_input_videos().frames, None, final_frame_index + 1
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
