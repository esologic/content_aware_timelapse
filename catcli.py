"""Main module."""

import logging
from pathlib import Path

import click
import more_itertools
import numpy as np
import torch
from torchvision import models

from content_aware_timelapse.viderator import video_common

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
    "--output",
    "-o",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True, path_type=Path),
    help="Output will be written to this file.",
    required=True,
)
def main(input_file: Path, _: Path) -> None:
    """
    This is the CLI!

    \f

    :return: None
    """

    # Load a pretrained model (e.g., ResNet18)
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.eval()  # Set to evaluation mode

    # Move the model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    def image_to_features(image):
        # Load and preprocess the image
        tensor_img = torch.from_numpy(image).permute(2, 0, 1)  # Change from HWC to CHW format
        tensor_img = tensor_img.to(torch.float32) / 255.0  # Normalize to [0, 1]
        tensor_img = tensor_img.unsqueeze(0)  # Add batch dimension

        # Move the input tensor to GPU if available
        if torch.cuda.is_available():
            tensor_img = tensor_img.cuda()

        with torch.no_grad():  # Disable gradient calculation
            features = model(tensor_img)  # Get features from the model

        return features.cpu().numpy()  # Move to CPU and return as NumPy array

    def calculate_distance(features1, features2):
        # Calculate Euclidean distance
        distance = np.linalg.norm(features1 - features2)
        return distance

    frames = video_common.frames_in_video(video_path=input_file)
    displayed_frames = video_common.display_frame_forward_opencv(source=frames.frames)

    for images in more_itertools.chunked(displayed_frames, 2):

        first_image, second_image = map(image_to_features, images)

        print(calculate_distance(first_image, second_image))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
