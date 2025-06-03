import torch
import clip
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from typing import Iterator, List
import more_itertools
import logging
import numpy.typing as npt

LOGGER = logging.getLogger(__name__)


def compute_vectors_clip(
        frame_batches: Iterator[List[np.ndarray]],
) -> Iterator[npt.NDArray[np.float16]]:
    """
    Computes CLIP embeddings for input frames using GPU acceleration if available.

    :param frame_batches: Iterator of lists of frames to compute vectors.
    :return: Iterator of CLIP feature vectors, one per input frame.
    """

    LOGGER.debug(f"Detected {torch.cuda.device_count()} GPUs. Loading CLIP Model")

    def load_clip_model(gpu_index: int) -> torch.nn.Module:
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
            image_batch: List[np.ndarray], model: torch.nn.Module, preprocess
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
            return model.encode_image(tensor_batch).half()

    def images_to_feature_vectors(
            image_batches: List[List[np.ndarray]], executor: ThreadPoolExecutor
    ) -> Iterator[np.ndarray]:
        """
        Converts batches of images into CLIP feature vectors using multiple GPUs.

        :param image_batches: List of batches of RGB images.
        :param executor: ThreadPoolExecutor for concurrent processing.
        :return: Iterator of feature vectors.
        """
        futures = [
            executor.submit(process_images_for_model, batch, model, preprocess)
            for (batch, (model, preprocess)) in zip(image_batches, models)
        ]

        for index, future in enumerate(as_completed(futures)):
            yield from future.result().cpu().numpy()
            LOGGER.debug(f"Processed image batch #{index} using CLIP.")

    # Thread pool for parallel GPU processing
    with ThreadPoolExecutor(max_workers=len(models)) as e:
        for batch in more_itertools.chunked(frame_batches, len(models)):
            yield from images_to_feature_vectors(batch, e)