"""
Uses clip embeddings as vectors.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterator, List, Tuple, cast

import clip
import more_itertools
import numpy as np
import torch
from numpy import typing as npt
from PIL import Image

from content_aware_timelapse.frames_to_vectors.conversion_types import (
    ConversionScoringFunctions,
    IndexScores,
    ScoreWeights,
)
from content_aware_timelapse.viderator.viderator_types import ImageResolution, RGBInt8ImageType

LOGGER = logging.getLogger(__name__)


def _compute_vectors_clip(
    frame_batches: Iterator[List[RGBInt8ImageType]],
) -> Iterator[npt.NDArray[np.float16]]:
    """
    Computes CLIP embeddings for input frames using GPU acceleration if available.

    :param frame_batches: Iterator of lists of frames to compute vectors.
    :return: Iterator of CLIP feature vectors, one per input frame.
    """

    LOGGER.debug(f"Detected {torch.cuda.device_count()} GPUs. Loading CLIP Model")

    def load_clip_model(
        gpu_index: int,
    ) -> Tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor]]:
        """
        Loads the CLIP model onto the specified GPU.

        :param gpu_index: Target GPU index.
        :return: The CLIP model.
        """
        LOGGER.debug(f"Loading CLIP model onto GPU index: {gpu_index}...")
        try:
            model, preprocess = clip.load("ViT-B/16", device=f"cuda:{gpu_index}")
            model.eval()
            return model, preprocess
        except RuntimeError:
            LOGGER.error("Error loading CLIP model!")
            raise

    # Load models onto each available GPU
    models = [load_clip_model(i) for i in range(torch.cuda.device_count())]

    def process_images_for_model(
        image_batch: List[RGBInt8ImageType],
        model: torch.nn.Module,
        preprocess: Callable[[Image.Image], torch.Tensor],
    ) -> torch.Tensor:
        """
        Preprocesses images and extracts CLIP feature vectors.

        :param image_batch: List of RGB images as NumPy arrays.
        :param model: The CLIP model.
        :param preprocess: The CLIP preprocessing pipeline.
        :return: Feature vector tensor.
        """
        LOGGER.debug("Processing images for CLIP inference.")

        # Convert images to PIL and preprocess
        tensor_batch = torch.stack([preprocess(Image.fromarray(img)) for img in image_batch])

        # Move to GPU
        device = next(model.parameters()).device
        tensor_batch = tensor_batch.to(device)

        with torch.no_grad():
            return cast(torch.Tensor, model.encode_image(tensor_batch).half())

    def images_to_feature_vectors(
        image_batches: List[List[RGBInt8ImageType]], executor: ThreadPoolExecutor
    ) -> Iterator[npt.NDArray[np.float16]]:
        """
        Converts batches of images into CLIP feature vectors using multiple GPUs.

        :param image_batches: List of batches of RGB images.
        :param executor: ThreadPoolExecutor for concurrent processing.
        :return: Iterator of feature vectors.
        """
        futures = [
            executor.submit(process_images_for_model, image_batch, model, preprocess)
            for (image_batch, (model, preprocess)) in zip(image_batches, models)
        ]

        for index, future in enumerate(as_completed(futures)):
            yield from future.result().cpu().numpy()
            LOGGER.debug(f"Processed image batch #{index} using CLIP.")

    # Thread pool for parallel GPU processing
    with ThreadPoolExecutor(max_workers=len(models)) as e:
        for batch in more_itertools.chunked(frame_batches, len(models)):
            yield from images_to_feature_vectors(batch, e)


def _calculate_scores_clip(  # pylint: disable=unused-argument
    packed: Tuple[int, npt.NDArray[np.float16]], original_source_resolution: ImageResolution
) -> IndexScores:
    """
    Calculate a combined score from various metrics of the CLIP embedding.

    Metrics:
        - Entropy: Measures how distributed the embedding values are.
        - Variance: Measures spread in embedding values.
        - Saliency: Measures maximum embedding value.
        - Energy: Measures total embedding magnitude (L2 norm).

    :param packed: Tuple of the index and the CLIP embedding vector.
    :param original_source_resolution: Provided by API, not consumed here yet.
    :return: Combined score and index.
    """

    index, embedding = packed
    embedding = embedding.astype(np.float32)  # Convert to higher precision for calculations

    # Normalize embedding
    norm_embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

    entropy_score = -np.sum(
        norm_embedding[norm_embedding > 0] * np.log(norm_embedding[norm_embedding > 0] + 1e-8)
    )

    return IndexScores(
        frame_index=index,
        entropy=entropy_score,
        variance=float(np.var(embedding)),
        saliency=float(np.max(embedding)),
        energy=float(np.linalg.norm(embedding)),
        interesting_points=[],
    )


CONVERT_CLIP = ConversionScoringFunctions(
    name="clip",
    conversion=_compute_vectors_clip,
    scoring=_calculate_scores_clip,
    weights=ScoreWeights(low_entropy=1, variance=1, saliency=1, energy=1),
)
