"""
Computing the feature vectors is the most computationally expensive part of creating these
timelapses. This module is responsible for offloading that computation to the GPU to improve
throughput.

Additionally, this module is capable of writing these feature vectors to disk, so they don't have
to be re-computed for similar runs.

These files, called "vector file"s, are HDF5 files of numpy arrays.
"""

import hashlib
import itertools
import json
import logging
from pathlib import Path
from typing import Iterator, List, NamedTuple

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
            yield np.array(dataset)

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

    f = h5py.File(vector_file, "a")

    if (
        SIGNATURE_ATTRIBUTE_NAME in f.attrs.keys()
        and f.attrs[SIGNATURE_ATTRIBUTE_NAME] == input_signature
        and len(f.keys())
    ):
        group = f[VECTORS_GROUP_NAME]
        starting_index = max(map(int, group.keys())) + 1
    elif (
        SIGNATURE_ATTRIBUTE_NAME in f.attrs.keys()
        and f.attrs[SIGNATURE_ATTRIBUTE_NAME] != input_signature
        and len(f.keys())
    ):
        raise ValueError("Can't write to vector file! Signature does not match.")
    else:
        f.attrs[SIGNATURE_ATTRIBUTE_NAME] = input_signature
        group = f.create_group(name=VECTORS_GROUP_NAME, track_order=True)
        starting_index = 0

    for index, vector in zip(itertools.count(starting_index), vector_iterator):
        group.create_dataset(
            name=str(index),
            shape=vector.shape,
            dtype=vector.dtype,
            data=vector,
            compression="gzip",
            compression_opts=9,
        )
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

    # Load a pre-trained Vision Transformer model (ViT)
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    model.eval()  # Set to evaluation mode

    # Move the model to GPU if available and convert to FP16
    if torch.cuda.is_available():
        model = model.cuda()
        model = model.half()  # Half precision for speed if using GPU

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
        # Preprocess images and stack them into a batch tensor by converting each image to PIL
        # and then applying transforms.
        tensor_image_batch = torch.stack(
            [vit_transform(Image.fromarray(img)).to(torch.float16) for img in images]
        )

        # Pin memory for better performance during transfer to GPU
        pinned = tensor_image_batch.pin_memory()

        # Move the input tensor to GPU if available
        if torch.cuda.is_available():
            tensor_image_batch = pinned.cuda(non_blocking=True)

        # Disable gradient calculation and pass the batch through the model
        with torch.no_grad():
            outputs = model.forward_features(tensor_image_batch)

            # Extract feature vectors directly from the outputs
            attention_map = outputs.cpu().numpy()  # Convert to NumPy array

        # Return each feature vector directly
        return (array for array in attention_map)

    yield from itertools.chain.from_iterable(map(images_to_feature_vectors, frame_batches))


def frames_to_vectors(
    frames: ImageSourceType,
    intermediate_path: Path,
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

    intermediate = _read_vector_file(vector_file=intermediate_path, input_signature=input_signature)

    fresh_tensors: Iterator[npt.NDArray[np.float16]] = iter([])

    if intermediate.length < total_input_frames:
        # Skip to the unprocessed section of the input.
        unprocessed_frames: ImageSourceType = itertools.islice(frames, intermediate.length, None)

        # Compute new vectors, writing the results to disk.
        fresh_tensors = _compute_vectors(more_itertools.chunked(unprocessed_frames, batch_size))
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
