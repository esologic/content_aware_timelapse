"""
Defines the forward features method of going from frames to vectors.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterator, List, Tuple, cast

import more_itertools
import numpy as np
import timm
import torch
import torchvision.transforms.functional as F
from numpy import typing as npt
from PIL import Image
from torch import Tensor
from torch.nn.modules.dropout import Dropout
from torchvision import transforms

from content_aware_timelapse.frames_to_vectors.conversion_types import ConversionScoringFunctions
from content_aware_timelapse.frames_to_vectors.vector_scoring import IndexScores
from content_aware_timelapse.viderator.viderator_types import PILImage, RGBInt8ImageType

LOGGER = logging.getLogger(__name__)


def create_padded_square_resizer(
    side_length: int = 224, fill_color: Tuple[int, int, int] = (123, 116, 103)
) -> Callable[[PILImage], PILImage]:
    """
    Create a function that when called resizes the input image to a square with a pad.
    :param side_length: Desired output side length.
    :param fill_color: Color of the pad.
    :return: Callable that does the conversion.
    """

    def output_callable(img: PILImage) -> PILImage:
        """
        Callable.
        :param img: To convert.
        :return: Converted.
        """

        img.thumbnail(
            (side_length, side_length),
            Image.BICUBIC,  # type: ignore[attr-defined] # pylint: disable=no-member
        )

        # Compute padding amounts
        delta_w = side_length - img.size[0]
        delta_h = side_length - img.size[1]
        padding = (
            delta_w // 2,
            delta_h // 2,
            delta_w - (delta_w // 2),
            delta_h - (delta_h // 2),
        )

        out = cast(PILImage, F.pad(img, padding, fill=fill_color))

        return out

    return output_callable


VIT_IMAGE_TRANSFORM = transforms.Compose(
    [
        create_padded_square_resizer(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _compute_vectors_vit_cls(
    frame_batches: Iterator[List[RGBInt8ImageType]],
) -> Iterator[npt.NDArray[np.float16]]:
    """
    Computes new vectors from the input frames. Uses GPU acceleration if available.
    :param frame_batches: Iterator of lists of frames to compute vectors. Frames are processed
    in batches, but output will be one vector per frame.
    :return: Iterator of vectors, one per input frame.
    """

    timm.layers.set_fused_attn(False)

    LOGGER.debug(f"Detected {torch.cuda.device_count()} GPUs. Loading Model")

    def load_model_onto_gpu(gpu_index: int) -> torch.nn.Module:
        """
        Creates the model, loads it onto the target GPU, and registers forward hooks
        to capture attention weights.
        :param gpu_index: Target of GPU.
        :return: the model for use.
        """
        LOGGER.debug(f"Loading model onto index: {gpu_index} and registering attention hooks...")

        try:
            model: torch.nn.Module = (
                timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
                .eval()
                .half()
                .cuda(device=gpu_index)
            )

            model.attention_weights_container = []  # type: ignore[assignment]

            def hook_fn(_module: Dropout, _input: Tuple[Tensor], output: Tensor) -> None:
                """
                Hook function to capture attention weights.
                'output' of attn.attn_drop is:
                    (batch_size, num_heads, sequence_length, sequence_length)
                """

                model.attention_weights_container.append(output.detach())

            # Iterate through each Transformer block and register a hook on its attention module
            for name, module in model.named_modules():
                if "attn_drop" in name:
                    module.register_forward_hook(hook_fn)

            return model
        except RuntimeError:
            LOGGER.error("Ran into error loading model!")
            raise

    # Load models to each GPU
    models: List[torch.nn.Module] = list(map(load_model_onto_gpu, range(torch.cuda.device_count())))

    def process_images_for_model(
        image_batch: List[RGBInt8ImageType], model: torch.nn.Module
    ) -> List[torch.Tensor]:
        """
        Preprocesses a batch of images and extracts feature vectors AND attention weights.

        :param image_batch: List of RGB images in NumPy format.
        :param model: The model to process the image batch.
        :return: A list of attention tensors from each
        layer (each tensor: batch_size, num_heads, 197, 197).
        """
        LOGGER.debug("Sent images for inference.")

        # Clear previous attention weights before new inference This is CRUCIAL to ensure you only
        # get attention for the current batch.
        model.attention_weights_container.clear()

        tensor_image_batch = torch.stack(
            [VIT_IMAGE_TRANSFORM(Image.fromarray(img)).to(torch.float16) for img in image_batch]
        )

        # Pin memory for better performance during transfer to GPU
        pinned = tensor_image_batch.pin_memory()

        # Move the input tensor to the correct GPU (same as the model's device)
        device = next(model.parameters()).device
        tensor_image_batch = pinned.to(device, non_blocking=True)

        # Disable gradient calculation and pass the batch through the model
        with torch.no_grad():
            LOGGER.debug(f"Sending images to {device}...")
            _output: torch.Tensor = model.forward_features(tensor_image_batch)
            return cast(List[Tensor], model.attention_weights_container)

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
        for _index, future in enumerate(as_completed(futures)):
            list_of_attention_tensors = future.result()
            # Convert to (batch, layers, heads, seq_len, seq_len)
            per_frame = torch.stack(list_of_attention_tensors, dim=1)

            for frame_attention in per_frame:
                yield frame_attention.cpu().numpy().astype(np.float16)

    # Create a single thread pool for all image batches
    with ThreadPoolExecutor(max_workers=len(models)) as e:
        for batch in more_itertools.chunked(frame_batches, len(models)):
            yield from images_to_feature_vectors(batch, e)
            # TODO: We can print reference counts here to look for memory leaks.


def _calculate_scores_vit_cls(packed: Tuple[int, npt.NDArray[np.float16]]) -> IndexScores:
    """
    Calculate scores from the CLS token embedding of an image and identify top pixels
    using attention-based patch saliency scores.

    :param packed: Tuple of (image_index, (full_vit_output, attention_saliency_scores_for_patches)).
                   full_vit_output: (197, 768) array (CLS + patches)
                   attention_saliency_scores_for_patches: (196,) array (saliency for each patch)
    :return: Calculated scores and index, including top pixel coordinates.
    """

    index, _stacked_attention_map = packed

    # TODO: Need to read the attention map and get the desired properties.

    return IndexScores(
        frame_index=index,
        entropy=1,
        variance=1,
        saliency=1,  # This is CLS embedding saliency
        energy=1,
        top_pixels=[],
    )


CONVERT_VIT_CLS = ConversionScoringFunctions(
    conversion=_compute_vectors_vit_cls, scoring=_calculate_scores_vit_cls
)
