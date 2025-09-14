"""
Types needed to describe the frames to vectors conversions.
"""

from typing import Iterator, List, NamedTuple, Optional, Protocol, Tuple, TypedDict

import numpy as np
import numpy.typing as npt

from content_aware_timelapse.viderator.viderator_types import (
    ImageResolution,
    RGBInt8ImageType,
    XYPoint,
)


class IndexScores(TypedDict):
    """
    Links the frame index of the input with a set of overall scores that describe the whole frame
    """

    frame_index: int
    entropy: float
    variance: float
    saliency: float
    energy: float


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


class ScoreWeights(NamedTuple):
    """
    Each of these values is a float between 0 and 1, and it is a multipler on the given column
    post normalization.
    """

    low_entropy: float
    variance: float
    saliency: float
    energy: float


class ConversionScoringFunctions(NamedTuple):
    """
    Links a conversion function (which goes from images -> vectors) to a scoring function which
    determines the numerical score for each vector. These two components make up both halves of
    the images to scores pipeline.
    """

    name: str
    conversion: ConvertBatchesFunction
    scoring: ScoreVectorsFunction
    weights: ScoreWeights
    max_side_length: Optional[int] = None
    """
    Allows pre-processing steps to shrink the input images to the exact side length, or nearby
    to decrease overall memory pressure, or pack more frames into memory before being processed.
    """


class IndexPointsOfInterest(TypedDict):
    """
    Links the frame index of the input with a list of points of interest within the image.
    This is consumed in auto-cropping to find the best region of the video.
    """

    frame_index: int
    points_of_interest: List[XYPoint]


class ScorePOIsFunction(Protocol):
    """
    Describes functions that convert vectors to points of interest in the source images.
    """

    def __call__(
        self,
        packed: Tuple[int, npt.NDArray[np.float16]],
        original_source_resolution: ImageResolution,
        num_interesting_points: int,
    ) -> IndexPointsOfInterest:
        """
        :param packed: A tuple, the index of the frame in the input and the vectors (an attention
        map) for that frame.
        :param original_source_resolution: Used to remap interesting locations back onto the full
        coordinate space of the source image.
        :param num_interesting_points: The number of points to generate per frame.
        :return: An NT that links the frame index with the list of interesting points.
        """


class ConversionPOIsFunctions(NamedTuple):
    """
    Similar to `ConversionScoringFunctions`, links the function that generates tensor
    with a function that creates the POIs.
    """

    name: str
    conversion: ConvertBatchesFunction
    compute_pois: ScorePOIsFunction
    max_side_length: Optional[int] = None
    """
    Allows pre-processing steps to shrink the input images to the exact side length, or nearby
    to decrease overall memory pressure, or pack more frames into memory before being processed.
    """
