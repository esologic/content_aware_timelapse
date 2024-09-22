"""Main module."""

import itertools
import logging
from pathlib import Path
from typing import List

import click
import more_itertools
import numpy as np
import torch
from torchvision import models

from content_aware_timelapse.viderator import iterator_common, video_common
from content_aware_timelapse.viderator.video_common import RGBInt8ImageType

LOGGER_FORMAT = "[%(asctime)s - %(process)s - %(name)20s - %(levelname)s] %(message)s"
LOGGER_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOGGER_FORMAT,
    datefmt=LOGGER_DATE_FORMAT,
)

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option(
    "--input",
    "-i",
    "input_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path),
    help="Input file.",
    required=True,
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
    show_default=True
)
@click.option(
    "--output-fps",
    "-f",
    type=click.FloatRange(min=1),
    help="Frames are sent to GPU for processing in batches of this size.",
    required=True,
    default=60.0,
    show_default=True
)
@click.option(
    "--batch-size",
    "-b",
    type=click.IntRange(min=1),
    help="Frames are sent to GPU for processing in batches of this size.",
    required=True,
    default=85,
    show_default=True
)
def main(input_file: Path, output_path: Path, duration: float, output_fps: float, batch_size: int) -> None:  # pylint: disable=unused-argument
    """


    This CLI reads a video file, extracts frames, calculates feature vectors using a pretrained ResNet18 model,
    and computes the Euclidean distance between feature vectors of consecutive frames.

    :param input_file: Path to the input video file.
    :param output: Path to the output file (not currently used).
    :return: None
    """

    # Load a pretrained model (e.g., ResNet18)
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.eval()  # Set to evaluation mode

    # Move the model to GPU if available and convert to FP16
    if torch.cuda.is_available():
        model = model.cuda()

    model = model.half()

    for param in model.parameters():
        param.requires_grad = False

    def images_to_feature_vectors(images: List[RGBInt8ImageType]) -> List[np.ndarray]:
        """
        Convert a list of RGB images into feature vectors using a pretrained ResNet model.

        The images are preprocessed, converted to FP16, and then passed through the model to obtain feature vectors.

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
            features = model(tensor_image_batch)

        # Move the features back to CPU if they were on GPU
        features = features.cpu()

        print(type(features))

        # Split the batch back into individual feature vectors
        individual_features = features.unbind(0)  # List of tensors, each with shape (1000,)

        return [f.numpy() for f in individual_features]

    def calculate_distance(current_features: np.ndarray, previous_features: np.ndarray) -> float:
        """
        Calculate the Euclidean distance between two feature vectors.

        :param current_features: First feature vector (NumPy array).
        :param previous_features: Second feature vector (NumPy array).
        :return: Euclidean distance as a float.
        """
        return np.linalg.norm(current_features - previous_features)

    # Get the video frames and process them
    video_source = video_common.frames_in_video(video_path=input_file)
    processing_frames = video_common.display_frame_forward_opencv(source=video_source.frames)
    processing_frames = iterator_common.items_per_second(processing_frames, queue_size=500)

    # Convert frames to feature vectors and chain them together
    frames_as_vectors = itertools.chain.from_iterable(
        map(images_to_feature_vectors, more_itertools.chunked(processing_frames, batch_size))
    )

    # Calculate and print distances between consecutive frames
    for first_image, second_image in more_itertools.chunked(frames_as_vectors, 2):
        print(calculate_distance(first_image, second_image))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
