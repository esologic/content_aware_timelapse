import itertools
from pathlib import Path
from typing import Callable, Iterator, List, NamedTuple, Optional

import h5py
import more_itertools
import numpy as np
import numpy.typing as npt
import timm  # PyTorch image models
import torch
from PIL import Image
from torchvision import transforms

from content_aware_timelapse.viderator.video_common import (
    ImageSourceType,
    RGBInt8ImageType,
    VideoFrames,
)


class _CountIterator(NamedTuple):
    count: int
    iterator: Iterator[npt.NDArray[np.float16]]


def read_intermediate(intermediate_path: Path, input_signature: str) -> _CountIterator:
    """
    Read data from an HDF5 file and return an iterator using closures that auto-close the file.

    :param intermediate_path: Path to the HDF5 file.
    :param input_signature: Signature to validate against.
    :return: _CountIterator with the count and an iterator for the dataset.
    """

    dataset_name = "vectors"
    f = h5py.File(intermediate_path, "a")

    def iterator_closure() -> Iterator[npt.NDArray[np.float16]]:
        """

        :param hdf_file:
        :param dataset_name:
        :return:
        """

        dataset = f[dataset_name]
        dataset_iter = iter(dataset)

        for item in dataset_iter:
            yield item

        f.close()  # Automatically close the file when iteration is done

    if f.attrs["signature"] == input_signature and len(f.keys()):
        return _CountIterator(
            count=len(f[dataset_name]),
            iterator=iterator_closure(),  # Pass the iterator as a closure
        )
    else:
        f.close()  # Close the file if signature doesn't match or no keys present
        return _CountIterator(
            count=0,
            iterator=iter([]),
        )


def vector_iterator(
    frame_batches: Iterator[List[RGBInt8ImageType]],
) -> Iterator[npt.NDArray[np.uint8]]:
    """

    :param frame_batches:
    :return:
    """

    # Load a pre-trained Vision Transformer model (ViT)
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    model.eval()  # Set to evaluation mode

    # Move the model to GPU if available and convert to FP16
    if torch.cuda.is_available():
        model = model.cuda()

    model = model.half()

    vit_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # ViT expects 224x224 input
            transforms.ToTensor(),  # Convert to tensor and scale [0, 255] -> [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # ViT normalization
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
        # Preprocess images and stack them into a batch tensor
        tensor_image_batch = torch.stack(
            [vit_transform(Image.fromarray(img)).to(torch.float16) for img in images]
            # Convert each image to PIL and then apply transforms
        ).pin_memory()  # Pin memory for better performance during transfer to GPU

        # Move the input tensor to GPU if available
        if torch.cuda.is_available():
            tensor_image_batch = tensor_image_batch.cuda(non_blocking=True)

        # Disable gradient calculation and pass the batch through the model
        with torch.no_grad():
            outputs = model.forward_features(tensor_image_batch)

            # Extract feature vectors directly from the outputs
            attention_map = outputs.cpu().numpy()  # Convert to NumPy array

        # Return each feature vector directly
        return (array for array in attention_map)

    yield from itertools.chain.from_iterable(map(images_to_feature_vectors, frame_batches))


def frames_to_vectors(
    frames: Iterator[RGBInt8ImageType],
    intermediate_path: Path,
    input_signature: str,
    batch_size: int,
) -> Iterator[npt.NDArray[np.uint8]]:
    """

    :param frames:
    :param intermediate_path:
    :param input_signature:
    :param batch_size:
    :return:
    """

    intermediate = read_intermediate(intermediate_path, input_signature)

    fresh_tensors = vector_iterator(
        more_itertools.chunked(itertools.islice(frames, intermediate.count, None), batch_size)
    )

    yield from itertools.chain.from_iterable((intermediate.iterator, fresh_tensors))
