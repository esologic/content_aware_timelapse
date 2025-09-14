"""
Test of the POI determination process.
"""

from test.assets import SAMPLE_TIMELAPSE_INPUT_PATH

import more_itertools
import pytest

from content_aware_timelapse.frames_to_vectors import vector_points_of_interest
from content_aware_timelapse.frames_to_vectors.conversion_types import ConversionPOIsFunctions
from content_aware_timelapse.frames_to_vectors.vector_computation.compute_vectors_vit import (
    CONVERT_POIS_VIT_ATTENTION,
)
from content_aware_timelapse.viderator import frames_in_video, iterator_on_disk, video_common
from content_aware_timelapse.viderator.viderator_types import ImageResolution


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
        iterator=video_frames.frames, copies=2
    )

    pio_crop_result = vector_points_of_interest.crop_to_pois(
        analysis_frames=analysis_frames,
        output_frames=output_frames,
        drawing_frames=visualization_frames,
        intermediate_info=None,
        batch_size=100,
        total_input_frames=video_frames.total_frame_count,
        convert=pois_functions.conversion,
        compute=pois_functions.compute_pois,
        original_resolution=video_frames.original_resolution,
        crop_resolution=ImageResolution(500, 500),
    )

    cropped = video_common.display_frame_forward_opencv(
        source=pio_crop_result.cropped_to_region, window_name="Cropped"
    )
    visualization = video_common.display_frame_forward_opencv(
        source=pio_crop_result.visualization, window_name="Visualization"
    )

    more_itertools.consume(visualization)
    more_itertools.consume(cropped)
