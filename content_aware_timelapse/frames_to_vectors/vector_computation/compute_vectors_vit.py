"""
Defines the forward features method of going from frames to vectors.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator, List

import more_itertools
import numpy as np
import timm
import torch
from numpy import typing as npt
from PIL import Image
from torchvision import transforms

from content_aware_timelapse.viderator.image_common import RGBInt8ImageType

LOGGER = logging.getLogger(__name__)


def compute_vectors_vit_cls(
    frame_batches: Iterator[List[RGBInt8ImageType]],
) -> Iterator[npt.NDArray[np.float16]]:
    """
    Computes new vectors from the input frames. Uses GPU acceleration if available.
    :param frame_batches: Iterator of lists of frames to compute vectors. Frames are processed
    in batches, but output will be one vector per frame.
    :return: Iterator of vectors, one per input frame.
    """

    LOGGER.debug(f"Detected {torch.cuda.device_count()} GPUs. Loading Model")

    def load_model_onto_gpu(gpu_index: int) -> torch.nn.Module:
        """
        Creates the model and loads it onto the target GPU, adding logging.
        :param gpu_index: Target of GPU.
        :return: the model for use.
        """
        LOGGER.debug(f"Loading model onto index: {gpu_index}...")
        try:
            model: torch.nn.Module = (
                timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
                .eval()
                .half()
                .cuda(device=gpu_index)
            )
            return model
        except RuntimeError:
            LOGGER.error("Ran into error loading model!")
            raise

    # Load models to each GPU
    models: List[torch.nn.Module] = list(map(load_model_onto_gpu, range(torch.cuda.device_count())))

    vit_transform = transforms.Compose(
        [
            # Resize the smaller edge to 224 (maintains aspect ratio)
            transforms.Resize(224),
            # Center crop (or pad) the image to 224x224
            transforms.CenterCrop(224),
            # Convert to tensor and scale [0, 255] -> [0, 1]
            transforms.ToTensor(),
            # ViT normalization
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def process_images_for_model(
        image_batch: List[RGBInt8ImageType], model: torch.nn.Module
    ) -> torch.Tensor:
        """
        Preprocesses a batch of images and extracts feature vectors using the specified model.

        :param image_batch: List of RGB images in NumPy format.
        :param model: The model to process the image batch.
        :return: Feature vector tensor after processing the batch on the model's device.
        """
        LOGGER.debug("Sent images for inference.")

        # Preprocess images and stack them into a batch tensor by converting each image to PIL
        tensor_image_batch = torch.stack(
            [vit_transform(Image.fromarray(img)).to(torch.float16) for img in image_batch]
        )

        # Pin memory for better performance during transfer to GPU
        pinned = tensor_image_batch.pin_memory()

        # Move the input tensor to the correct GPU (same as the model's device)
        device = next(model.parameters()).device
        tensor_image_batch = pinned.to(device, non_blocking=True)

        # Disable gradient calculation and pass the batch through the model
        with torch.no_grad():
            LOGGER.debug(f"Sending images to {device}...")

            # Forward pass for feature extraction
            output: torch.Tensor = model.forward_features(tensor_image_batch)
            return output

    def images_to_feature_vectors(
        image_batches: List[List[RGBInt8ImageType]], executor: ThreadPoolExecutor
    ) -> Iterator[npt.NDArray[np.float16]]:
        """
        Convert a list of RGB image batches into feature vectors using pretrained models on multiple
        GPUs.

        The images are preprocessed, converted to FP16, and passed through each model to obtain
        feature vectors. The final result is an iterator over the vectors.

        :param image_batches: List of image batches (each containing a list of RGB images).
        :param executor: A ThreadPoolExecutor to handle concurrent image processing.
        :return: Iterator of feature vectors, one per image.
        """
        # Submit image batches to thread pool
        futures = [
            executor.submit(process_images_for_model, image_batch, model)
            for image_batch, model in zip(image_batches, models)
        ]

        # Collect results as they are completed
        for index, future in enumerate(as_completed(futures)):
            # Extract the CLS token embedding for each image in the batch.
            # The CLS token is at index 0 of the second dimension.
            cls_embeddings_batch = future.result()[:, 0, :].cpu().numpy()

            # Yield each CLS embedding individually
            yield from cls_embeddings_batch
            LOGGER.debug(
                f"Got back image batch #{index} from GPU. "
                f"Yielded CLS embeddings for {cls_embeddings_batch.shape[0]} images."
            )

    # Create a single thread pool for all image batches
    with ThreadPoolExecutor(max_workers=len(models)) as e:
        for batch in more_itertools.chunked(frame_batches, len(models)):
            yield from images_to_feature_vectors(batch, e)
            # TODO: We can print reference counts here to look for memory leaks.
