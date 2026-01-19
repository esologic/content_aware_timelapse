"""
Test of the POI determination process.
"""

from pathlib import Path
from test.assets import SAMPLE_TIMELAPSE_INPUT_PATH
from test.test_viderator import viderator_test_common

import more_itertools
import pytest

from content_aware_timelapse.frames_to_vectors import vector_points_of_interest
from content_aware_timelapse.frames_to_vectors.conversion_types import ConversionPOIsFunctions
from content_aware_timelapse.frames_to_vectors.vector_computation.compute_vectors_vit import (
    CONVERT_POIS_VIT_ATTENTION,
)
from content_aware_timelapse.gpu_discovery import discover_gpus
from content_aware_timelapse.viderator import (
    frames_in_video,
    image_common,
    iterator_on_disk,
    video_common,
)
from content_aware_timelapse.viderator.viderator_types import (
    ImageResolution,
    RectangleRegion,
    XYPoint,
)


@pytest.mark.integration
@pytest.mark.skip()
@pytest.mark.parametrize("pois_functions", [CONVERT_POIS_VIT_ATTENTION])
def test_crop_to_pois_visualization_check(pois_functions: ConversionPOIsFunctions) -> None:
    """
    Uses the visualization output to verify that the resulting crop decision looks good.
    :param pois_functions: POIs determination function that is under test.
    :return: None
    """

    video_frames = frames_in_video.frames_in_video_opencv(
        video_path=SAMPLE_TIMELAPSE_INPUT_PATH,
    )

    analysis_frames, output_frames, visualization_frames = iterator_on_disk.tee_disk_cache(
        iterator=video_frames.frames,
        copies=2,
        serializer=iterator_on_disk.HDF5_COMPRESSED_SERIALIZER,
    )

    winning_region = next(
        iter(
            vector_points_of_interest.discover_poi_regions(
                analysis_frames=analysis_frames,
                intermediate_info=None,
                batch_size=100,
                source_frame_count=video_frames.total_frame_count,
                conversion_pois_functions=pois_functions,
                original_resolution=video_frames.original_resolution,
                crop_resolution=ImageResolution(500, 500),
                gpus=discover_gpus(),
                num_regions=1,
            )
        )
    )

    cropped = video_common.display_frame_forward_opencv(
        source=video_common.crop_source(
            source=output_frames,
            region=winning_region,
        ),
        window_name="Cropped",
    )

    visualization = video_common.display_frame_forward_opencv(
        source=(
            image_common.draw_regions_on_image(
                regions=[winning_region],
                image=frame,
            )
            for frame in visualization_frames
        ),
        window_name="Visualization",
    )

    more_itertools.consume(visualization)
    more_itertools.consume(cropped)


def _run_top_regions_test(  # pylint: disable=too-many-positional-arguments,protected-access
    image_size: ImageResolution,
    region_resolution: ImageResolution,
    num_regions: int,
    expected_regions: list[RectangleRegion],
    output_path: Path,
) -> None:
    """
    Runs a region selection test with a fixed cluster of points and visualizes the result.

    The point configuration is:
    - A dense cluster of points in the top-left.
    - A few points near that cluster.
    - A single isolated point in the bottom-right.
    """

    points = [
        # Main Cluster of points in the top left
        XYPoint(0, 0),
        XYPoint(1, 1),
        XYPoint(10, 10),
        XYPoint(0, 10),
        XYPoint(10, 0),
        # Single point in the bottom right, with at least a 100x100 box of free space around it
        XYPoint(299, 299),
        # Two points near the cluster as well
        XYPoint(90, 90),
        XYPoint(90, 95),
    ]

    regions = vector_points_of_interest._top_regions(
        points=[
            vector_points_of_interest._PointFrameCount(point=point, frame_count=1)
            for point in points
        ],
        image_size=image_size,
        region_resolution=region_resolution,
        num_regions=num_regions,
        alpha_points_frames=1,  # Equal weighting by frame count.
    )

    # Visualization for eyeball review
    image_common.save_rgb_image(
        path=output_path,
        image=image_common.draw_points_on_image(
            points=points,
            radius=1,
            image=image_common.draw_regions_on_image(
                regions=regions,
                image=next(
                    viderator_test_common.create_black_frames_iterator(
                        image_resolution=image_size, count=1
                    )
                ),
                width=1,
            ),
        ),
    )

    assert len(regions) == len(expected_regions)
    assert regions == expected_regions


def test__top_regions_no_overlap(artifact_root: Path) -> None:
    """
    Creates a few clusters of points and then draws the best regions.

    Many points in the top left, several points near the cluster, and then a single point well
    away from the cluster in the bottom right.

    Because the points near the cluster would create a region that intersects with the region
    created by the top left cluster, a region containing a single point in the bottom right is
    created instead.
    :param artifact_root: Test fixture that provides an optionally persisted directory to write
    test assets to.
    :return: None
    """
    _run_top_regions_test(
        image_size=ImageResolution(300, 300),
        region_resolution=ImageResolution(100, 100),
        num_regions=2,
        expected_regions=[
            RectangleRegion(top=0, left=0, bottom=100, right=100),
            RectangleRegion(top=200, left=200, bottom=300, right=300),
        ],
        output_path=artifact_root / "./test_viz_no_overlap.png",
    )
