"""Main module."""

import itertools
import logging
from pathlib import Path
from typing import Iterator, List, NamedTuple, Optional

import click
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from content_aware_timelapse import frames_to_vectors
from content_aware_timelapse.frames_to_vectors import create_videos_signature
from content_aware_timelapse.viderator import video_common
from content_aware_timelapse.viderator.video_common import ImageSourceType, VideoFrames

LOGGER_FORMAT = "[%(asctime)s - %(process)s - %(name)20s - %(levelname)s] %(message)s"
LOGGER_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOGGER_FORMAT,
    datefmt=LOGGER_DATE_FORMAT,
)

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


@click.command()
@click.option(
    "--input",
    "-i",
    "input_files",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path),
    help="Input file(s). Can be given multiple times.",
    required=True,
    multiple=True,
)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True, path_type=Path),
    help="Output will be written to this file.",
    required=True,
)
@click.option(
    "--duration",
    "-d",
    type=click.FloatRange(min=1),
    help="Frames are sent to GPU for processing in batches of this size.",
    required=True,
    default=30.0,
    show_default=True,
)
@click.option(
    "--output-fps",
    "-f",
    type=click.FloatRange(min=1),
    help="Frames are sent to GPU for processing in batches of this size.",
    required=True,
    default=60.0,
    show_default=True,
)
@click.option(
    "--batch-size",
    "-b",
    type=click.IntRange(min=1),
    help="Frames are sent to GPU for processing in batches of this size.",
    required=True,
    default=600,
    show_default=True,
)
@click.option(
    "--vectors-path",
    "-v",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path),
    help="Intermediate vectors will be written to this path. Can be used to re-run.",
    required=False,
)
def main(  # pylint: disable=too-many-locals
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

    processing_frames = video_common.display_frame_forward_opencv(source=frames_count.frames)
    # processing_frames = iterator_common.preload_into_memory(
    #     processing_frames, buffer_size=int(math.ceil(batch_size))
    # )

    def calculate_score(attention_map: npt.NDArray[np.float16]) -> float:
        """
        Calculate the Euclidean distance between two feature vectors.

        :param current_features: First feature vector (NumPy array).
        :param previous_features: Second feature vector (NumPy array).
        :return: Euclidean distance as a float.
        """
        return float(np.mean(attention_map))

    vectors = frames_to_vectors.frames_to_vectors(
        frames=processing_frames,
        intermediate_path=vectors_path,
        input_signature=input_signature,
        batch_size=batch_size,
        total_input_frames=frames_count.total_frame_count,
    )

    score_indexes: Iterator[_ScoreIndex] = (
        _ScoreIndex(score=calculate_score(attention_map), idx=index)
        for index, attention_map in tqdm(
            enumerate(vectors),
            total=frames_count.total_frame_count,
            unit="Frames",
            ncols=100,
            desc="Scoring Images",
        )
    )

    sorted_by_score: List[_ScoreIndex] = sorted(
        score_indexes, key=lambda distance_index: distance_index.score, reverse=True
    )

    sliced: List[_ScoreIndex] = sorted_by_score[: int(duration * output_fps)]

    indices_only = set(map(lambda distinct_index: distinct_index.idx, sliced))

    output_iterator: ImageSourceType = (
        index_frame[1]
        for index_frame in filter(
            lambda index_frame: index_frame[0] in indices_only,
            enumerate(load_input_videos().frames),
        )
    )

    progress_bar_output: ImageSourceType = (
        item
        for item in tqdm(
            output_iterator, total=len(sliced), unit="Frames", ncols=100, desc="Writing to Disk"
        )
    )

    video_common.write_source_to_disk_consume(
        source=progress_bar_output, video_path=output_path, video_fps=output_fps, high_quality=True
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
