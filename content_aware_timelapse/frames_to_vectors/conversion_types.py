"""
Types needed to describe the frames to vectors conversions.
"""

from typing import Iterator, List, Protocol

import numpy as np
import numpy.typing as npt

from content_aware_timelapse.viderator.video_common import RGBInt8ImageType


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
