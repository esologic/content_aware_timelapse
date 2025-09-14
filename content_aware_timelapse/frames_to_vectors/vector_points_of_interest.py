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


def _top_regions(  # pylint: disable=too-many-locals,too-many-arguments,too-many-locals
    points: List[_PointFrameCount],
    image_size: ImageResolution,
    region_resolution: ImageResolution,
    num_regions: int,
    alpha: float,
) -> List[RectangleRegion]:
    """
    Exhaustively find the top regions of size `region_size` anywhere in the image. Score each
    region as a weighted combination of:
      - normalized unique point count in the region
      - normalized sum of frame counts in the region

    :param points: Points of interest within the frame. This list of NTs contains the coordinates
    of those points and the number of frames they appear in.
    :param image_size: Size of the image that the POIs exist in.
    :param region_resolution: The desired size of the output region.
    :param num_regions: Every possible region in the image will be considered, but only the top
    number of regions will be returned.
    :param alpha: relative weight of unique points vs frame counts. 0.0 = only frame counts,
    1.0 = only unique points.
    :return: A list of regions sorted by score with the highest scoring region first.
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

    # Integral images for O(1) sums
    integral_frame = frame_count_map.cumsum(axis=0).cumsum(axis=1)
    integral_unique = unique_map.cumsum(axis=0).cumsum(axis=1)

    region_scores: List[Tuple[float, float, RectangleRegion]] = []

    # Slide window over all positions
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
                region_scores.append((frame_sum, unique_sum, region))

    if not region_scores:
        return []

    # Extract max values for normalization
    max_frame = max(r[0] for r in region_scores)
    max_unique = max(r[1] for r in region_scores)

    scored_regions: List[_ScoredRegion] = []

    for frame_sum, unique_sum, region in region_scores:
        frame_norm = frame_sum / max_frame if max_frame > 0 else 0
        unique_norm = unique_sum / max_unique if max_unique > 0 else 0
        score = alpha * unique_norm + (1 - alpha) * frame_norm
        scored_regions.append(
            _ScoredRegion(
                score=score,
                region=region,
            )
        )

    # Sort by normalized blended score
    scored_regions.sort(key=lambda x: x.score, reverse=True)
    return [r.region for r in scored_regions[:num_regions]]


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
    intermediate_info: Optional[IntermediateFileInfo],
    batch_size: int,
    total_input_frames: int,
    convert: ConvertBatchesFunction,
    compute: ScorePOIsFunction,
    original_resolution: ImageResolution,
    crop_resolution: ImageResolution,
) -> POICropResult:
    """
    Crops an input video to a region of that video that is the most interesting using points of
    interest analysis.

    The input video is given at least twice in `analysis_frames`, `output_frames`, and
    `drawing_frames`. The contents of these iterators should be pretty much the same, and they
    should all have the same length.

    :param analysis_frames: These frames are fed to the GPU via the conversion function and will
    likely be scaled down. Pre-scaled and buffered images should be passed here to avoid waiting
    around.
    :param output_frames: These are the frames that are actually cropped and forwarded in the
    output.
    :param drawing_frames: Should be another copy of `output_frames` that is consumed in
    visualization. If this is not given the `visualization` field in the output will also
    be None.
    :param intermediate_info: Describes the vectors file if given.
    :param batch_size:
    :param total_input_frames:
    :param convert:
    :param compute:
    :param original_resolution:
    :param crop_resolution:
    :return:
    """

    vectors_for_pois: Iterator[npt.NDArray[np.float16]] = conversion.frames_to_vectors(
        frames=analysis_frames,
        intermediate_info=intermediate_info,
        batch_size=batch_size,
        total_input_frames=total_input_frames,
        convert_batches=convert,
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
        drop_frame_threshold=0.7,
    )

    winning_region = next(
        iter(
            _top_regions(
                points=winning_points_and_counts,
                image_size=original_resolution,
                region_resolution=crop_resolution,
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
