"""
Defines the forward features method of going from frames to vectors.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from functools import partial
from typing import Callable, Iterator, List, Optional, Tuple, cast

import more_itertools
import numpy as np
import timm
import torch
import torchvision.transforms.functional as F
from numpy import typing as npt
from PIL import Image
from pympler.tracker import SummaryTracker
from torch import Tensor
from torch.nn.modules.dropout import Dropout
from torchvision import transforms

from content_aware_timelapse.frames_to_vectors.conversion_types import (
    ConversionPOIsFunctions,
    ConversionScoringFunctions,
    IndexPointsOfInterest,
    IndexScores,
    ScoreWeights,
    XYPoint,
)
from content_aware_timelapse.frames_to_vectors.vector_computation import gpu_model_management
from content_aware_timelapse.gpu_discovery import GPUDescription
from content_aware_timelapse.viderator.viderator_types import (
    ImageResolution,
    PILImage,
    RGBInt8ImageType,
)

LOGGER = logging.getLogger(__name__)

logging.getLogger("timm").setLevel(logging.ERROR)

VIT_SIDE_LENGTH = 224


def _create_padded_square_resizer(
    side_length: int = VIT_SIDE_LENGTH, fill_color: Tuple[int, int, int] = (123, 116, 103)
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


def _map_output_to_source(
    output_point: XYPoint, original_resolution: ImageResolution, side_length: int
) -> XYPoint:
    """
    Maps a point in the ViT output image to the coordinate space of the original input.
    Reverses the scale and pad operation (`_create_padded_square_resizer`) needed to send the
    image in for VIT inference.
    :param output_point: XYPoint of the output image.
    :param original_resolution: Size of the source image.
    :param side_length: VIT side length.
    :return: Remapped point.
    """

    scale = min(side_length / original_resolution.width, side_length / original_resolution.height)
    pad_x = (side_length - round(original_resolution.width * scale)) // 2
    pad_y = (side_length - round(original_resolution.height * scale)) // 2

    x_src = (output_point.x - pad_x) / scale
    y_src = (output_point.y - pad_y) / scale

    # Clamp to valid coordinates
    x_src = int(round(x_src))
    y_src = int(round(y_src))
    x_src = max(0, min(x_src, original_resolution.width - 1))
    y_src = max(0, min(y_src, original_resolution.height - 1))

    return XYPoint(x_src, y_src)


VIT_IMAGE_TRANSFORM = transforms.Compose(
    [
        _create_padded_square_resizer(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class VITOutputMode(int, Enum):
    """
    Different possible VIT output nodes.
    """

    CLS_TOKEN = 0
    ATTENTION_MAP = 1


def _compute_vectors_vit(
    output_mode: VITOutputMode,
    frame_batches: Iterator[List[RGBInt8ImageType]],
    gpus: Tuple[GPUDescription, ...],
    track_memory_usage: bool = False,
) -> Iterator[npt.NDArray[np.float16]]:
    """
    Computes new vectors from the input frames. Uses GPU acceleration if available.
    :param output_mode: Desired output mode, controls the format of the output vectors.
    :param frame_batches: Iterator of lists of frames to compute vectors. Frames are processed
    in batches, but output will be one vector per frame.
    :param gpus: GPUs to use for this computation.
    :param track_memory_usage: If given, debugging information about memory usage over time
    will be printed to identify memory leaks.
    :return: Iterator of vectors, one per input frame.
    """

    if output_mode == VITOutputMode.ATTENTION_MAP:
        timm.layers.set_fused_attn(False)

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

            if output_mode == VITOutputMode.ATTENTION_MAP:

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
    models: List[torch.nn.Module] = list(map(load_model_onto_gpu, gpus))

    def process_images_for_model(
        image_batch: List[RGBInt8ImageType], model: torch.nn.Module
    ) -> Tensor:
        """
        Preprocesses a batch of images and extracts feature vectors AND attention weights.

        :param image_batch: List of RGB images in NumPy format.
        :param model: The model to process the image batch.
        :return: A list of attention tensors from each
        layer (each tensor: batch_size, num_heads, 197, 197).
        """
        LOGGER.debug("Sent images for inference.")

        if output_mode == VITOutputMode.ATTENTION_MAP:
            # Clear previous attention weights before new inference This is CRUCIAL to ensure you
            # only get attention for the current batch.
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

            features = model.forward_features(tensor_image_batch)

            if output_mode == VITOutputMode.CLS_TOKEN:
                output: Tensor = features
            elif output_mode == VITOutputMode.ATTENTION_MAP:
                output = torch.stack(model.attention_weights_container, dim=1)
            else:
                raise ValueError("Unknown output_mode.")

            LOGGER.debug(f"Got back result. Shape: {output.shape}, ")

            return output

    def images_to_feature_vectors(
        image_batches: List[List[RGBInt8ImageType]], executor: ThreadPoolExecutor
    ) -> Iterator[npt.NDArray[np.float16]]:
        """
        Convert a list of RGB image batches into feature vectors using pretrained models on multiple
        GPUs.

        The images are preprocessed, converted to FP16, and passed through each model to obtain
        feature vectors. The final result is an iterator over the vectors, one per frame.
        """
        # Submit image batches to thread pool
        futures = [
            executor.submit(process_images_for_model, image_batch, model)
            for image_batch, model in zip(image_batches, models)
        ]

        # Collect results as they are completed
        for future in as_completed(futures):
            output_selection = future.result()

            # Move to CPU once, then yield one frame at a time
            output_cpu = output_selection[:, 0, :].cpu()
            for frame in output_cpu:
                yield frame.numpy()

    summary_tracker: Optional[SummaryTracker] = None

    if track_memory_usage:
        summary_tracker = SummaryTracker()  # type:ignore[no-untyped-call]

    with gpu_model_management.gpu_cleanup(models=models):

        # Create a single thread pool for all image batches
        # GPU parallel consumption is achieved in a hacky way here, by splitting the input batches
        # into n lists where n is the number of GPUS, then giving each CPU and sub list to a thread.
        with ThreadPoolExecutor(max_workers=len(models)) as e:
            for batch in more_itertools.chunked(frame_batches, len(models)):
                yield from images_to_feature_vectors(batch, e)
                if summary_tracker is not None:
                    summary_tracker.print_diff()  # type:ignore[no-untyped-call]


def _calculate_scores_vit_cls(packed: Tuple[int, npt.NDArray[np.float16]]) -> IndexScores:
    """
    Calculate scores from the CLS token embedding of an image.

    Metrics are reinterpreted for feature vectors:
        - Energy: Measures the L2 norm (magnitude) of the feature vector, indicating overall
                  feature activation strength.
        - Saliency: Measures the maximum activation value within the vector, pointing to the
                    strongest single feature.
        - Variance: Measures the spread of values in the vector, indicating diversity or
                    concentration of feature activations.
        - Entropy: Measures the distribution of (normalized positive) feature activations.
                   Its interpretation for 'interestingness' is less direct for feature vectors
                   compared to spatial attention maps, but can indicate feature sparsity/density.

    :param packed: Tuple of the image's index and its CLS token embedding.
    :return: Calculated scores and index.
    """

    index, cls_embedding = packed
    cls_embedding = cls_embedding.astype(np.float32)  # Ensure float32 for calculations

    # --- Re-interpreting Metrics for a CLS Feature Vector (768 dimensions) ---

    # Energy: L2 Norm of the CLS embedding. A higher norm generally means a stronger,
    # more confident, or more distinct feature representation.
    energy_score = np.linalg.norm(cls_embedding)

    # Indicates the strongest single feature component the model picked up.
    saliency_score = np.percentile(cls_embedding, 90)

    # Variance: Variance of the values in the CLS embedding.
    # High variance suggests some feature dimensions are highly active while others are not.
    # Low variance suggests more uniform feature activations.
    variance_score = np.var(cls_embedding)

    # Entropy: Entropy of the distribution of (normalized positive) feature values.
    # If the embedding has negative values, consider taking abs or handling carefully.
    # Here, we normalize only positive values to make it behave like a probability distribution.
    positive_values = cls_embedding[cls_embedding > 0]
    if positive_values.size > 0:
        # Normalize positive values to sum to 1 to form a probability distribution
        normalized_positive_values = positive_values / (np.sum(positive_values) + 1e-8)
        entropy_score = -np.sum(
            normalized_positive_values * np.log(normalized_positive_values + 1e-8)
        )
    else:
        # If no positive values, entropy is conventionally 0 (or adjust as needed)
        entropy_score = 0.0

    return IndexScores(
        frame_index=index,
        entropy=entropy_score,
        variance=float(variance_score),
        saliency=float(saliency_score),
        energy=float(energy_score),
    )


def _calculate_scores_vit_attention(packed: Tuple[int, npt.NDArray[np.float16]]) -> IndexScores:
    """
    Calculate scores from the attention map(s) of an image.

    Metrics are reinterpreted for attention maps:
        - Energy: Frobenius norm of the attention maps, indicating overall strength.
        - Saliency: 90th percentile of attention weights, highlighting strongest focus.
        - Variance: Spread of attention values; high = sparse focus, low = diffuse.
        - Entropy: Normalized entropy of attention distribution;
                   low = focused, high = diffuse.

    :param packed: Tuple of (frame_index, attention_map).
                   attention_map may be (layers, heads, seq_len, seq_len)
                   or already reduced. All dims are flattened here.
    :return: IndexScores with scalar metrics for the frame.
    """
    index, attention_map = packed

    # Flatten across all dimensions
    map_as_float = attention_map.astype(np.float32).ravel()

    # Energy: Frobenius norm (total activation strength)
    energy_score = np.linalg.norm(map_as_float)

    # Saliency: 90th percentile (strongest attention weight)
    saliency_score = np.percentile(map_as_float, 90)

    # Variance: spread of attention values
    variance_score = np.var(map_as_float)

    # Entropy: treat as probability distribution
    if map_as_float.size > 0:
        probs = map_as_float / (np.sum(map_as_float) + 1e-8)
        entropy_score = -np.sum(probs * np.log(probs + 1e-8))
    else:
        entropy_score = 0.0

    return IndexScores(
        frame_index=index,
        entropy=float(entropy_score),
        variance=float(variance_score),
        saliency=float(saliency_score),
        energy=float(energy_score),
    )


def _calculate_poi_vit_attention(  # pylint: disable=too-many-locals
    packed: Tuple[int, npt.NDArray[np.float16]],
    original_source_resolution: ImageResolution,
    num_interesting_points: int,
) -> IndexPointsOfInterest:
    """
    Calculate attention-based scalar scores + top-K interesting points, excluding edge patches.

    :param packed: Tuple of (frame_index, attention_map). Shape: (H,197,197) or (197,197)
    :param original_source_resolution: Width/height of original image (before ViT resize/pad)
    :param num_interesting_points: Number of interesting points to extract. This will be per frame.
    :return: Links the frame and the detected points of interest.
    """
    index, attention_map = packed
    attention_map = np.asarray(attention_map).astype(np.float32)

    # Average over heads if needed
    if attention_map.ndim == 3:  # (H,197,197)
        attention_map = attention_map.mean(axis=0)  # (197,197)

    # Compute per-patch scores (exclude CLS token)
    patch_attn = attention_map[1:, 1:]  # shape: (196, 196)
    num_patches = patch_attn.shape[0]

    incoming = patch_attn.sum(axis=0)
    outgoing = patch_attn.sum(axis=1)
    entropy = -(
        patch_attn
        / patch_attn.sum(axis=1, keepdims=True)
        * np.log(patch_attn / patch_attn.sum(axis=1, keepdims=True) + 1e-8)
    ).sum(axis=1)

    patch_scores = incoming + outgoing - entropy  # high attention + sharp distribution

    # Select top points, excluding edge patches
    points_to_select = min(num_interesting_points, num_patches)
    points_of_interest: List[XYPoint] = []

    if points_to_select > 0:
        grid_size = int(np.sqrt(num_patches))  # 14x14 for 196 patches
        top_indices = np.argsort(patch_scores)[::-1]  # sort descending

        for idx in top_indices:
            gx, gy = idx % grid_size, idx // grid_size

            # Map to ViT grid
            x_vit = (gx + 0.5) * (VIT_SIDE_LENGTH / grid_size)
            y_vit = (gy + 0.5) * (VIT_SIDE_LENGTH / grid_size)

            # Map to original image
            pt = _map_output_to_source(
                output_point=XYPoint(x=int(round(x_vit)), y=int(round(y_vit))),
                original_resolution=original_source_resolution,
                side_length=VIT_SIDE_LENGTH,
            )

            points_of_interest.append(pt)
            if len(points_of_interest) >= points_to_select:
                break

    return IndexPointsOfInterest(
        frame_index=index,
        points_of_interest=points_of_interest,
    )


CONVERT_SCORE_VIT_CLS = ConversionScoringFunctions(
    name="vit_cls_token",
    conversion=partial(_compute_vectors_vit, VITOutputMode.CLS_TOKEN),
    scoring=_calculate_scores_vit_cls,
    weights=ScoreWeights(low_entropy=0.5, variance=0.2, saliency=0.5, energy=0.7),
    max_side_length=VIT_SIDE_LENGTH,
)

CONVERT_SCORE_VIT_ATTENTION = ConversionScoringFunctions(
    name="vit_attention_map",
    conversion=partial(_compute_vectors_vit, VITOutputMode.ATTENTION_MAP),
    scoring=_calculate_scores_vit_attention,
    weights=ScoreWeights(low_entropy=0.4, variance=0.1, saliency=0.4, energy=0.2),
    max_side_length=VIT_SIDE_LENGTH,
)

CONVERT_POIS_VIT_ATTENTION = ConversionPOIsFunctions(
    name="vit_attention_map",
    conversion=partial(_compute_vectors_vit, VITOutputMode.ATTENTION_MAP),
    compute_pois=_calculate_poi_vit_attention,
    max_side_length=VIT_SIDE_LENGTH,
)
