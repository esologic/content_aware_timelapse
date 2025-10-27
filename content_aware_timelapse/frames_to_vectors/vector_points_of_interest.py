"""
Library code related to points of interest within inamges.
"""

from functools import partial
from typing import Iterator, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy import typing as npt
from tqdm import tqdm

from content_aware_timelapse.frames_to_vectors import conversion
from content_aware_timelapse.frames_to_vectors.conversion import IntermediateFileInfo
from content_aware_timelapse.frames_to_vectors.conversion_types import (
    ConversionPOIsFunctions,
    IndexPointsOfInterest,
)
from content_aware_timelapse.gpu_discovery import GPUDescription
from content_aware_timelapse.viderator.viderator_types import (
    ImageResolution,
    ImageSourceType,
    RectangleRegion,
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
    drop_frame_threshold: float,
) -> List[_PointFrameCount]:
    """
    Process the points of interest list, converting it to a list of points and the number of
    frames where that point appears.
    :param points_of_interest: A list of the POIs discovered for each input frame.
    :param drop_frame_threshold: Drop points that appear in more than this percent of frames.
    Float from 0-1
    :return: A list of points mapped to the number of frames they appear in.
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
        (frames_per_point["frame_count"] / len(points_of_interest)) < drop_frame_threshold
    ]

    # Convert to list of DynamicPoint
    result: List[_PointFrameCount] = [
        _PointFrameCount(XYPoint(row.x, row.y), int(row.frame_count))
        for row in dynamic_points_df.itertuples(index=False)
    ]

    return result


class _ScoredRegion(NamedTuple):
    """
    Links a region with its score
    """

    score: float
    region: RectangleRegion


def _score_region(
    integral_image: Union[
        npt.NDArray[np.float32],
        npt.NDArray[np.int32],
    ],
    region: RectangleRegion,
) -> float:
    """
    Compute the sum of values within a rectangular region of an integral image.

    :param integral_image: 2D cumulative sum array. This could be the sum of frame count, or the
    sum of bulk attention.
    :param region: The region to compute the sum of within the integral region.
    :return: The total sum of the values inside the window region.
    """

    total = integral_image[region.bottom - 1, region.right - 1]
    if region.top > 0:
        total -= integral_image[region.top - 1, region.right - 1]
    if region.left > 0:
        total -= integral_image[region.bottom - 1, region.left - 1]
    if region.top > 0 and region.left > 0:
        total += integral_image[region.top - 1, region.left - 1]

    return float(total)


def _iou(region_a: RectangleRegion, region_b: RectangleRegion) -> float:
    """Compute intersection-over-union between two regions."""
    inter_left = max(region_a.left, region_b.left)
    inter_top = max(region_a.top, region_b.top)
    inter_right = min(region_a.right, region_b.right)
    inter_bottom = min(region_a.bottom, region_b.bottom)

    if inter_right <= inter_left or inter_bottom <= inter_top:
        return 0.0

    inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)
    area_a = (region_a.right - region_a.left) * (region_a.bottom - region_a.top)
    area_b = (region_b.right - region_b.left) * (region_b.bottom - region_b.top)
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _top_regions(  # pylint: disable=too-many-locals,too-many-arguments
    points: List[_PointFrameCount],
    image_size: ImageResolution,
    region_resolution: ImageResolution,
    num_regions: int,
    alpha_points_frames: float,
) -> List[RectangleRegion]:
    """
    Exhaustively find the top regions of size `region_resolution` anywhere in the image. Score each
    region as a weighted combination of:
      - normalized unique point count in the region
      - normalized sum of frame counts in the region

    Selection process:
      1. Start with the highest scoring region.
      2. Iteratively add regions, always preferring zero-overlap regions.
      3. If no zero-overlap regions remain, pick regions that minimize overlap while maximizing
      score.
      4. Guarantees up to `num_regions` regions are returned if candidates exist.

    :param points: Points of interest within the frame.
    :param image_size: Size of the image.
    :param region_resolution: Desired size of the output region.
    :param num_regions: Number of regions to return.
    :param alpha_points_frames: Relative weight of unique points vs frame counts.
    :return: List of top `num_regions` RectangleRegion objects.
    """

    if not points:
        return []

    # Rasterize points into frame-count and unique-point maps
    frame_count_map = np.zeros((image_size.height, image_size.width), dtype=np.float32)
    unique_map = np.zeros((image_size.height, image_size.width), dtype=np.int32)

    for p in points:
        if 0 <= p.point.x < image_size.width and 0 <= p.point.y < image_size.height:
            frame_count_map[p.point.y, p.point.x] += p.frame_count
            unique_map[p.point.y, p.point.x] = 1

    # Integral images for O(1) sum calculations
    integral_frame = frame_count_map.cumsum(axis=0).cumsum(axis=1)
    integral_unique = unique_map.cumsum(axis=0).cumsum(axis=1)

    # Compute raw scores for all possible regions
    region_scores: List[_ScoredRegion] = []
    for top in range(0, image_size.height - region_resolution.height + 1):
        for left in range(0, image_size.width - region_resolution.width + 1):
            region = RectangleRegion(
                top=top,
                left=left,
                bottom=top + region_resolution.height,
                right=left + region_resolution.width,
            )
            frame_sum = _score_region(integral_image=integral_frame, region=region)
            unique_sum = _score_region(integral_image=integral_unique, region=region)

            if frame_sum > 0 or unique_sum > 0:
                region_scores.append(
                    _ScoredRegion(
                        score=alpha_points_frames * unique_sum
                        + (1 - alpha_points_frames) * frame_sum,
                        region=region,
                    )
                )

    if not region_scores:
        return []

    remaining = sorted(region_scores, key=lambda r: r.score, reverse=True)

    # Pre-populate with the top-scoring region first
    selected_regions: List[RectangleRegion] = [remaining.pop(0).region]

    # Iteratively pick remaining regions
    while len(selected_regions) < num_regions and remaining:

        # Partition candidates into zero-overlap vs overlapping
        zero_overlap_indices = [
            i
            for i, r in enumerate(remaining)
            if all(_iou(r.region, s) == 0.0 for s in selected_regions)
        ]

        if zero_overlap_indices:
            # Pick highest scoring zero-overlap region
            best_idx = max(zero_overlap_indices, key=lambda i: remaining[i].score)
        else:
            # Pick region with the highest score penalized by max overlap
            best_idx = max(
                range(len(remaining)),
                key=lambda i: remaining[i].score
                * (1 - max((_iou(remaining[i].region, s) for s in selected_regions), default=0.0)),
            )

        selected_regions.append(remaining[best_idx].region)
        remaining.pop(best_idx)

    return selected_regions[:num_regions]


def discover_poi_regions(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    analysis_frames: ImageSourceType,
    intermediate_info: Optional[IntermediateFileInfo],
    batch_size: int,
    source_frame_count: int,
    conversion_pois_functions: ConversionPOIsFunctions,
    original_resolution: ImageResolution,
    crop_resolution: ImageResolution,
    gpus: Tuple[GPUDescription, ...],
    num_regions: int,
    alpha_points_frames: float = 0.8,
) -> Tuple[RectangleRegion, ...]:
    """
    Returns a tuple of the most interesting regions in a frame.

    :param analysis_frames: These frames are fed to the GPU via the conversion function and will
    likely be scaled down. Pre-scaled and buffered images should be passed here to avoid waiting
    around.
    :param intermediate_info: If given, attention map vectors will be written to disk for faster
    re-runs.
    :param batch_size: Number of scaled input images to send to the GPU at once.
    :param source_frame_count: Number of frames in the input.
    :param conversion_pois_functions: Conversion POIs functions to compute and derive POIs from
    input frames.
    :param original_resolution: Resolution of the source.
    :param crop_resolution: Desired resolution to crop to.
    :param num_regions:
    :param alpha_points_frames:
    :param gpus: GPUs to use in the computation.
    :return: Output NT. Contains the region, the cropped output iterator and the visualization
    iterator.
    """

    vectors_for_pois: Iterator[npt.NDArray[np.float16]] = conversion.frames_to_vectors(
        frames=analysis_frames,
        intermediate_info=intermediate_info,
        batch_size=batch_size,
        total_input_frames=source_frame_count,
        convert_batches=conversion_pois_functions.conversion,
        gpus=gpus,
    )

    points_of_interest: List[IndexPointsOfInterest] = list(
        map(
            partial(
                conversion_pois_functions.compute_pois,
                original_source_resolution=original_resolution,
                num_interesting_points=30,
            ),
            tqdm(
                enumerate(vectors_for_pois),
                total=source_frame_count,
                unit="Frames",
                ncols=100,
                desc="Discovering POIs",
                maxinterval=1,
            ),
        )
    )

    winning_points_and_counts: List[_PointFrameCount] = _count_frames_filter(
        points_of_interest=points_of_interest,
        drop_frame_threshold=0.7,
    )

    return tuple(
        _top_regions(
            points=winning_points_and_counts,
            image_size=original_resolution,
            region_resolution=crop_resolution,
            num_regions=num_regions,
            alpha_points_frames=alpha_points_frames,
        )
    )
