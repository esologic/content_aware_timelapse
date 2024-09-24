"""Main module."""

import itertools
import logging
import math
from pathlib import Path
from typing import Iterator, List, NamedTuple

import click
import more_itertools
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torchvision import models
from tqdm import tqdm

from content_aware_timelapse.viderator import iterator_common, video_common
from content_aware_timelapse.viderator.video_common import (
    ImageSourceType,
    RGBInt8ImageType,
    VideoFrames,
)

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


class _DistanceIndex(NamedTuple):
    """
    Intermediate type for linking the Euclidean distance between the next frame and the index
    of the frame.
    """

    distance: float
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
    default=85,
    show_default=True,
)
def main(  # pylint: disable=too-many-locals
    input_files: List[Path], output_path: Path, duration: float, output_fps: float, batch_size: int
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
    :return: None
    """

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
    processing_frames = iterator_common.preload_into_memory(
        processing_frames, buffer_size=int(math.ceil(2.5 * batch_size))
    )

    # Load a pretrained model (e.g., ResNet18)
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.eval()  # Set to evaluation mode

    # Move the model to GPU if available and convert to FP16
    if torch.cuda.is_available():
        model = model.cuda()

    model = model.half()

    def images_to_feature_vectors(
        images: List[RGBInt8ImageType],
    ) -> Iterator[npt.NDArray[np.float16]]:
        """
        Convert a list of RGB images into feature vectors using a pretrained ResNet model.

        The images are preprocessed, converted to FP16, and then passed through the model to obtain
        feature vectors.

        :param images: List of images in RGB format (as NumPy arrays).
        :return: List of feature vectors as NumPy arrays.
        """
        # Preprocess images and stack them into a batch tensor
        tensor_image_batch = torch.stack(
            [torch.from_numpy(img).permute(2, 0, 1).to(torch.float16) / 255.0 for img in images],
        ).pin_memory()

        # Move the input tensor to GPU if available
        if torch.cuda.is_available():
            tensor_image_batch = tensor_image_batch.cuda(non_blocking=True)

        # Disable gradient calculation and pass the batch through the model
        with torch.no_grad():
            features: Tensor = model(tensor_image_batch)

        # Move the features back to CPU if they were on GPU
        features = features.cpu()

        # Split the batch back into individual feature vectors
        individual_features = features.unbind(0)  # List of tensors, each with shape (1000,)

        return map(lambda tensor: tensor.numpy(), individual_features)

    def calculate_distance(
        current_features: npt.NDArray[np.float16], previous_features: npt.NDArray[np.float16]
    ) -> float:
        """
        Calculate the Euclidean distance between two feature vectors.

        :param current_features: First feature vector (NumPy array).
        :param previous_features: Second feature vector (NumPy array).
        :return: Euclidean distance as a float.
        """
        return float(np.linalg.norm(current_features - previous_features))

    # Convert frames to feature vectors and chain them together
    frames_as_vectors = itertools.chain.from_iterable(
        map(images_to_feature_vectors, more_itertools.chunked(processing_frames, batch_size))
    )

    distance_indexes: Iterator[_DistanceIndex] = (
        _DistanceIndex(distance=calculate_distance(first_image, second_image), idx=index)
        for index, (first_image, second_image) in tqdm(
            enumerate(more_itertools.pairwise(frames_as_vectors)),
            total=frames_count.total_frame_count,
            unit="Frames",
            ncols=100,
            desc="Vectorizing Images",
        )
    )

    sorted_by_distance: List[_DistanceIndex] = sorted(
        distance_indexes, key=lambda distance_index: distance_index.distance, reverse=True
    )

    sliced: List[_DistanceIndex] = sorted_by_distance[: int(duration * output_fps)]

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
