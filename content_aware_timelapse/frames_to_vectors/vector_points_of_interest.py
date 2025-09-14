"""
Library code related to points of interest within inamges.
"""

from functools import partial
from pathlib import Path
from typing import Iterator, List, NamedTuple, Optional, Union

import numpy as np
import pandas as pd
from numpy import typing as npt
from tqdm import tqdm

import content_aware_timelapse.frames_to_vectors
from content_aware_timelapse.frames_to_vectors.conversion_types import (
    ConvertBatchesFunction,
    IndexPointsOfInterest,
    ScorePOIsFunction,
)
from content_aware_timelapse.viderator import image_common, video_common
from content_aware_timelapse.viderator.viderator_types import (
    ImageResolution,
    ImageSourceType,
    RectangleRegion,
    RGBInt8ImageType,
    XYPoint,
)


class _PointFrameCount(NamedTuple):
    """
    Dynamic point with its frequency across frames.
    """

    point: XYPoint
    frame_count: int


def _count_frames_filter(
    points_of_interest: List[IndexPointsOfInterest],
    total_frame_count: int,
    drop_frame_threshold: float,
) -> List[_PointFrameCount]:
    """

    :param points_of_interest:
    :param total_frame_count:
    :param drop_frame_threshold: Drop points that appear in more than this percent of frames.
    Float from 0-1
    :return:
    """

    # Flatten to DataFrame
    df = pd.DataFrame.from_records(
        (
            (pt.x, pt.y, frame["frame_index"])
            for frame in points_of_interest
            for pt in frame["points_of_interest"]
        ),
        columns=["x", "y", "frame_index"],
    )

    # Count in how many frames each (x, y) appears
    frames_per_point = (
        df.groupby(["x", "y"])["frame_index"].nunique().reset_index(name="frame_count")
    )

    # Keep only dynamic points
    dynamic_points_df = frames_per_point[
        (frames_per_point["frame_count"] / total_frame_count) < drop_frame_threshold
    ]

    # Convert to list of DynamicPoint
    result: List[_PointFrameCount] = [
        _PointFrameCount(XYPoint(row.x, row.y), int(row.frame_count))
        for row in dynamic_points_df.itertuples(index=False)
    ]

    return result


def _window_sum(
    integral: Union[
        npt.NDArray[np.float32],
        npt.NDArray[np.int32],
    ],
    top: int,
    left: int,
    h: int,
    w: int,
) -> float:
    """
    Compute the sum of values within a rectangular region using an integral image.

    Parameters
    ----------
    integral : numpy.ndarray
        The integral image (2D cumulative sum array). Can be of dtype ``float32``,
        ``int32``, or similar.
    top : int
        The top row index of the window (inclusive).
    left : int
        The left column index of the window (inclusive).
    h : int
        The height of the window in pixels.
    w : int
        The width of the window in pixels.

    Returns
    -------
    float
        The total sum of the values inside the window region.
    """
    bottom = top + h
    right = left + w
    total = integral[bottom - 1, right - 1]
    if top > 0:
        total -= integral[top - 1, right - 1]
    if left > 0:
        total -= integral[bottom - 1, left - 1]
    if top > 0 and left > 0:
        total += integral[top - 1, left - 1]
    return float(total)


def _top_regions(  # pylint: disable=too-many-locals,too-many-arguments,too-many-locals
    points: List[_PointFrameCount],
    image_size: ImageResolution,
    crop_resolution: ImageResolution,
    num_regions: int,
    alpha: float,
) -> List[RectangleRegion]:
    """
    Exhaustively find the top-k regions of size `region_size` anywhere in the image.
    Score each region as a weighted combination of:
      - normalized unique point count in the region
      - normalized sum of frame counts in the region

    alpha: relative weight of unique points vs frame counts.
           0.0 = only frame counts, 1.0 = only unique points.
    """
    if not points:
        return []

    width, height = image_size

    # Rasterize points into frame-count and unique-point maps
    frame_count_map = np.zeros((height, width), dtype=np.float32)
    unique_map = np.zeros((height, width), dtype=np.int32)

    for p in points:
        x, y = p.point.x, p.point.y
        if 0 <= x < width and 0 <= y < height:
            frame_count_map[y, x] += p.frame_count
            unique_map[y, x] = 1

    # Integral images for O(1) sums
    integral_frame = frame_count_map.cumsum(axis=0).cumsum(axis=1)
    integral_unique = unique_map.cumsum(axis=0).cumsum(axis=1)

    region_scores = []

    # Slide window over all positions
    for top in range(0, height - crop_resolution.height + 1):
        for left in range(0, width - crop_resolution.width + 1):
            frame_sum = _window_sum(
                integral_frame, top, left, crop_resolution.height, crop_resolution.width
            )
            unique_sum = _window_sum(
                integral_unique, top, left, crop_resolution.height, crop_resolution.width
            )
            if frame_sum > 0 or unique_sum > 0:
                region_scores.append((frame_sum, unique_sum, top, left))

    if not region_scores:
        return []

    # Extract max values for normalization
    max_frame = max(r[0] for r in region_scores)
    max_unique = max(r[1] for r in region_scores)

    best_regions = []
    for frame_sum, unique_sum, top, left in region_scores:
        frame_norm = frame_sum / max_frame if max_frame > 0 else 0
        unique_norm = unique_sum / max_unique if max_unique > 0 else 0
        score = alpha * unique_norm + (1 - alpha) * frame_norm
        best_regions.append(
            (
                score,
                RectangleRegion(
                    top, left, top + crop_resolution.height, left + crop_resolution.width
                ),
            )
        )

    # Sort by normalized blended score
    best_regions.sort(key=lambda x: x[0], reverse=True)
    return [r[1] for r in best_regions[:num_regions]]


class POICropResult(NamedTuple):
    """
    Output of the POI crop process.
    """

    winning_region: RectangleRegion
    cropped_to_region: ImageSourceType
    visualization: Optional[ImageSourceType]


def crop_to_pois(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    analysis_frames: ImageSourceType,
    output_frames: ImageSourceType,
    drawing_frames: Optional[ImageSourceType],
    intermediate_path: Path,
    input_signature: str,
    batch_size: int,
    total_input_frames: int,
    convert: ConvertBatchesFunction,
    compute: ScorePOIsFunction,
    original_resolution: ImageResolution,
    crop_resolution: ImageResolution,
) -> POICropResult:
    """

    :param analysis_frames:
    :param output_frames:
    :param drawing_frames:
    :param intermediate_path:
    :param input_signature:
    :param batch_size:
    :param total_input_frames:
    :param convert:
    :param compute:
    :param original_resolution:
    :param crop_resolution:
    :param visualization:
    :return:
    """

    vectors_for_pois: Iterator[npt.NDArray[np.float16]] = (
        content_aware_timelapse.frames_to_vectors.conversion.frames_to_vectors(
            frames=analysis_frames,
            intermediate_path=intermediate_path,
            input_signature=input_signature,
            batch_size=batch_size,
            total_input_frames=total_input_frames,
            convert_batches=convert,
        )
    )

    points_of_interest: List[IndexPointsOfInterest] = list(
        map(
            partial(
                compute, original_source_resolution=original_resolution, num_interesting_points=30
            ),
            tqdm(
                enumerate(vectors_for_pois),
                total=total_input_frames,
                unit="Frames",
                ncols=100,
                desc="Discovering POIs",
                maxinterval=1,
            ),
        )
    )

    winning_points_and_counts: List[_PointFrameCount] = _count_frames_filter(
        points_of_interest=points_of_interest,
        total_frame_count=total_input_frames,
        drop_frame_threshold=0.7,
    )

    winning_region = next(
        iter(
            _top_regions(
                points=winning_points_and_counts,
                image_size=original_resolution,
                crop_resolution=crop_resolution,
                num_regions=1,
                alpha=0.8,
            )
        )
    )

    cropped_to_region: ImageSourceType = video_common.crop_source(
        source=output_frames, region=winning_region
    )

    visualization: Optional[ImageSourceType] = None

    if drawing_frames is not None:

        winning_points = {
            points_and_counts.point for points_and_counts in winning_points_and_counts
        }

        def render_visualization_frame(
            frame: RGBInt8ImageType, frame_pois: IndexPointsOfInterest
        ) -> RGBInt8ImageType:
            """

            :param frame:
            :param frame_pois:
            :return:
            """

            return image_common.draw_regions_on_image(
                regions=[winning_region],
                image=image_common.draw_points_on_image(
                    points=[
                        point
                        for point in frame_pois["points_of_interest"]
                        if point in winning_points
                    ],
                    image=frame,
                ),
            )

        visualization = (
            render_visualization_frame(frame, score)
            for score, frame in zip(points_of_interest, drawing_frames)
        )

    return POICropResult(
        winning_region=winning_region,
        cropped_to_region=cropped_to_region,
        visualization=visualization,
    )
