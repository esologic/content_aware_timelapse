"""
Types needed to describe the frames to vectors conversions.
"""

from typing import Iterator, List, NamedTuple, Protocol, Tuple

import numpy as np
import numpy.typing as npt

from content_aware_timelapse.frames_to_vectors.vector_scoring import IndexScores
from content_aware_timelapse.viderator.viderator_types import RGBInt8ImageType


class ConvertBatchesFunction(Protocol):
    """
    Describes functions that convert a batch of frames to vectors. This way multiple converters
    can be substituted.
    """

    def __call__(
        self, frame_batches: Iterator[List[RGBInt8ImageType]]
    ) -> Iterator[npt.NDArray[np.float16]]:
        """
        :param frame_batches: Iterator of batches (lists) of frames for conversion.
        :return: An iterator of the converted vectors.
        """


class ScoreVectorsFunction(Protocol):
    """
    Describes functions that convert vectors to numerical properties of the vectors.
    """

    def __call__(self, packed: Tuple[int, npt.NDArray[np.float16]]) -> IndexScores:
        """
        :param packed: A tuple, the index of the frame in the input and the calculated vectors
        for that frame.
        :return: An IndexScores, which are the numerical properties of the vectors.
        """


class ConversionScoringFunctions(NamedTuple):
    """
    Links a conversion function (which goes from images -> vectors) to a scoring function which
    determines the numerical score for each vector. These two components make up both halves of
    the images to scores pipeline.
    """

    conversion: ConvertBatchesFunction
    scoring: ScoreVectorsFunction
