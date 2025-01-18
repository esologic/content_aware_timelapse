"""
Computing the feature vectors is the most computationally expensive part of creating these
timelapses. This module is responsible for offloading that computation to the GPU to improve
throughput.

Additionally, this module is capable of writing these feature vectors to disk, so they don't have
to be re-computed for similar runs.

These files, called "vector file"s, are HDF5 files of numpy arrays.
"""

import hashlib
import io
import itertools
import json
import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator, List, NamedTuple, Optional, Tuple

import h5py
import more_itertools
import numpy as np
import numpy.typing as npt
import timm
import torch
from PIL import Image
from torchvision import transforms

from content_aware_timelapse.viderator.video_common import ImageSourceType, RGBInt8ImageType

LOGGER = logging.getLogger(__name__)

VECTORS_GROUP_NAME = "vectors"
SIGNATURE_ATTRIBUTE_NAME = "signature"


def create_videos_signature(video_paths: List[Path]) -> str:
    """
    Used to describe the contents of a vector file. Becomes an attribute in the HDF5 file.
    :param video_paths: Paths to the videos in the vector file.
    :return: A string, ready to be input to the HDF5 file.
    """

    def compute_partial_hash(video_path: Path) -> str:
        """
        Hashes the first 512KB of a video file.
        :param video_path: Path to the video.
        :return: Hash digest as a string.
        """

        sha256 = hashlib.sha256()

        with video_path.open("rb") as f:
            sha256.update(f.read(512_000))

        return sha256.hexdigest()

    # Convert dictionary to JSON string
    return json.dumps(
        {
            video.name: json.dumps(
                {
                    "file_size": video.stat().st_size,
                    "partial_sha256": compute_partial_hash(video),
                }
            )
            for video in sorted(video_paths, key=str)
        }
    )


class _LengthIterator(NamedTuple):
    """
    Intermediate type.
    Links the count of vectors on disk, in the file with an iterator that will emit those vectors.
    """

    length: int
    iterator: Iterator[npt.NDArray[np.float16]]


def _read_vector_file(vector_file: Path, input_signature: str) -> _LengthIterator:
    """
    Reads vectors from an HDF5 "vector file" as an iterator. When the iterator in the output is
    exhausted, it is auto-closed.

    If the input file is empty, the resulting count will be zero and the iterator will have
    no items in it.

    :param vector_file: Path to the HDF5 file.
    :param input_signature: Signature to validate against.
    :return: NT containing the number of vectors on disk and an iterator that produces them.
    """

    f = h5py.File(vector_file, "a")

    def iterate_from_file() -> Iterator[npt.NDArray[np.float16]]:
        """
        Yields each dataset in the vectors group.
        :return: An iterator of the datasets.
        """

        group = f[VECTORS_GROUP_NAME]

        for item in iter(group):
            dataset = group[item]
            yield dataset[()]

        # Automatically close the file when iteration is done.
        f.close()

    if (
        SIGNATURE_ATTRIBUTE_NAME in f.attrs.keys()
        and f.attrs[SIGNATURE_ATTRIBUTE_NAME] == input_signature
        and len(f.keys())
    ):
        completed_vectors = len(f[VECTORS_GROUP_NAME])
        return _LengthIterator(
            length=completed_vectors,
            iterator=iterate_from_file(),
        )
    else:
        # Close the file if signature doesn't match or file is empty.
        f.close()
        return _LengthIterator(
            length=0,
            iterator=iter([]),
        )


def _in_memory_hdf5(index_vector: Tuple[int, npt.NDArray[np.float16]]) -> Tuple[int, io.BytesIO]:
    """
    Create an in-memory HDF5 file of the input vector.
    :param index_vector: Packed input tuple of the frame's index in the video, and
    it's corresponding vectors.
    :return: A packed tuple of the input index and the bytesio containing the hdf5 file for copying.
    """

    index, vector = index_vector

    bytes_io = io.BytesIO()

    with h5py.File(bytes_io, "w") as f:
        f.create_dataset(
            name=str(index),
            shape=vector.shape,
            dtype=vector.dtype,
            data=vector,
            compression="gzip",
            compression_opts=9,
        )

    return index, bytes_io


def _write_vector_file_forward(
    vector_iterator: Iterator[npt.NDArray[np.float16]],
    vector_file: Path,
    input_signature: str,
) -> Iterator[npt.NDArray[np.float16]]:
    """
    Iterate through the input, writing the vectors to a vector file as we go. Unmodified vectors
    are then re-iterated in the output.
    :param vector_iterator: Input. Written to disk and then forwarded.
    :param vector_file: Output path. Note here:
        *   If this file already exists and contains vectors and a matching signature,
            it is assumed that the caller knows this, and is only inputting subsequent vectors.
            As a result, the input vectors are appended to the existing vectors in the group.
        *   If this file is empty, the index starts from zero.
    :param input_signature: Describes the source of the input vectors.
    :return: Iterator, forwarded from `vector_file`.
    :raises ValueError: If the vector file exists and the signature does not match the input.
    """

    with h5py.File(vector_file, "a") as f:

        if (
            SIGNATURE_ATTRIBUTE_NAME in f.attrs.keys()
            and f.attrs[SIGNATURE_ATTRIBUTE_NAME] == input_signature
            and len(f.keys())
        ):
            group = f[VECTORS_GROUP_NAME]

            if len(group):
                starting_index = max(map(int, group.keys())) + 1
            else:
                starting_index = 0
        elif (
            SIGNATURE_ATTRIBUTE_NAME in f.attrs.keys()
            and f.attrs[SIGNATURE_ATTRIBUTE_NAME] != input_signature
            and len(f.keys())
        ):
            raise ValueError("Can't write to vector file! Signature does not match.")
        else:
            f.attrs[SIGNATURE_ATTRIBUTE_NAME] = input_signature
            f.create_group(name=VECTORS_GROUP_NAME, track_order=True)
            starting_index = 0

        LOGGER.info(f"Starting to write vectors to: {vector_file}")

        # Doing this tee prevents the need to get the vector out of the pool twice.
        vectors_input, vectors_output = itertools.tee(vector_iterator, 2)

        with multiprocessing.Pool() as pool:

            for (index, bytes_io), vector in zip(
                pool.imap(_in_memory_hdf5, zip(itertools.count(starting_index), vectors_input)),
                vectors_output,
            ):

                with h5py.File(bytes_io) as input_file:
                    input_file.copy(input_file[str(index)], f[VECTORS_GROUP_NAME], str(index))

                f.flush()

                yield vector

    f.close()


def _compute_vectors(
    frame_batches: Iterator[List[RGBInt8ImageType]],
) -> Iterator[npt.NDArray[np.float16]]:
    """
    Computes new vectors from the input frames. Uses GPU acceleration if available.
    :param frame_batches: Iterator of lists of frames to compute vectors. Frames are processed
    in batches, but output will be one vector per frame.
    :return: Iterator of vectors, one per input frame.
    """

    LOGGER.debug("Loading Model...")

    # Load models to each GPU
    models = [
        timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        .eval()
        .half()
        .cuda(device=gpu_index)
        for gpu_index in range(torch.cuda.device_count())
    ]

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
            output = future.result()
            unpacked = output.cpu().numpy()
            LOGGER.debug(f"Got back image batch #{index} from GPU. Shape: {unpacked.shape}")
            yield from unpacked

    # Create a single thread pool for all image batches
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        # Use a single thread pool for all image batches
        yield from itertools.chain.from_iterable(
            map(
                lambda batch: images_to_feature_vectors(batch, executor),
                more_itertools.chunked(frame_batches, len(models)),
            )
        )


def frames_to_vectors(
    frames: ImageSourceType,
    intermediate_path: Optional[Path],
    input_signature: str,
    batch_size: int,
    total_input_frames: int,
) -> Iterator[npt.NDArray[np.float16]]:
    """
    Computes feature vectors from an input iterator or frames.
    Because this process is expensive, even with GPU, the intermediate vectors are written to disk
    to avoid re-doing the work.
    :param frames: Frames to process.
    :param intermediate_path: Vectors file to store intermediate results on disk.
    :param input_signature: Describes the video in `frames` to ensure we don't read from the wrong
    intermediate vectors from the input file.
    :param batch_size: Number of frames to process at once. Should try to utilize all GPU memory.
    :param total_input_frames: Number of frames in `frames`.
    :return: Vectors, one per input frame.
    """

    if intermediate_path is not None:

        intermediate = _read_vector_file(
            vector_file=intermediate_path, input_signature=input_signature
        )

        LOGGER.info(f"Read in {intermediate.length} intermediate vectors from file.")

        fresh_tensors: Iterator[npt.NDArray[np.float16]] = iter([])

        if intermediate.length < total_input_frames:

            LOGGER.info(f"Need to compute {total_input_frames-intermediate.length} new vectors.")

            # Skip to the unprocessed section of the input.
            unprocessed_frames: ImageSourceType = itertools.islice(
                frames, intermediate.length, None
            )

            # Compute new vectors, writing the results to disk.
            fresh_tensors = _compute_vectors(
                frame_batches=more_itertools.chunked(unprocessed_frames, batch_size)
            )

            fresh_tensors = _write_vector_file_forward(
                vector_iterator=fresh_tensors,
                vector_file=intermediate_path,
                input_signature=input_signature,
            )

        yield from itertools.chain.from_iterable(
            (
                intermediate.iterator,  # first output any vectors from disk.
                fresh_tensors,  # second, compute any vectors not found on disk.
            )
        )

    else:
        yield from _compute_vectors(frame_batches=more_itertools.chunked(frames, batch_size))
