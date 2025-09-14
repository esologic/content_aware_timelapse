"""
Main functionality, defines the pipeline.
"""

import itertools
import json
import logging
from functools import partial
from pathlib import Path
from typing import List, NamedTuple, Optional, cast

import more_itertools
from tqdm import tqdm

import content_aware_timelapse.viderator.frames_in_video
from content_aware_timelapse.frames_to_vectors.conversion import IntermediateFileInfo
from content_aware_timelapse.frames_to_vectors.conversion_types import (
    ConversionPOIsFunctions,
    ConversionScoringFunctions,
)
from content_aware_timelapse.frames_to_vectors.vector_points_of_interest import (
    POICropResult,
    crop_to_pois,
)
from content_aware_timelapse.frames_to_vectors.vector_scoring import reduce_frames_by_score
from content_aware_timelapse.vector_file import create_videos_signature
from content_aware_timelapse.viderator import image_common, iterator_common, iterator_on_disk
from content_aware_timelapse.viderator.video_common import VideoFrames
from content_aware_timelapse.viderator.viderator_types import (
    AspectRatio,
    ImageResolution,
    ImageSourceType,
)

LOGGER = logging.getLogger(__name__)


class _FramesCountResolution(NamedTuple):
    """
    Intermediate type for keeping track of the total number of frames in an iterator.
    """

    total_frame_count: int
    frames: ImageSourceType
    original_resolution: ImageResolution


def calculate_output_frames(duration: float, output_fps: float) -> int:
    """
    Canonical function to do this math.
    :param duration: Desired length of the video in seconds.
    :param output_fps: FPS of output.
    :return: Round number for output frames.
    """

    return int(duration * output_fps)


def load_input_videos(input_files: List[Path], tqdm_desc: str) -> _FramesCountResolution:
    """
    Helper function to combine the input videos.
    :param input_files: List of input videos.
    :param tqdm_desc: The resulting `.frames` field in the output iterator will be wrapped
    in a TQDM with this description.
    :return: NT containing the total frame count and a joined iterator of all the input
    frames.
    """

    input_video_frames: List[VideoFrames] = list(
        map(content_aware_timelapse.viderator.frames_in_video.frames_in_video_opencv, input_files)
    )

    all_input_frames = itertools.chain.from_iterable(
        [video_frames.frames for video_frames in input_video_frames]
    )

    total_frame_count = sum((video_frames.total_frame_count for video_frames in input_video_frames))

    input_resolutions = [video_frames.original_resolution for video_frames in input_video_frames]

    if len(input_video_frames) > 1:
        raise ValueError("Input videos have different resolutions.")

    return _FramesCountResolution(
        total_frame_count=total_frame_count,
        frames=cast(
            ImageSourceType,
            tqdm(
                all_input_frames,
                total=total_frame_count,
                unit="Frames",
                ncols=100,
                desc=tqdm_desc,
                maxinterval=1,
            ),
        ),
        original_resolution=next(iter(input_resolutions)),
    )


def preload_and_scale(
    frames_count_resolution: _FramesCountResolution,
    max_side_length: Optional[int],
    buffer_size: int,
) -> _FramesCountResolution:
    """
    Wraps a few library functions to optionally scale and pre-load input videos from disk into
    RAM for faster processing. See the underlying docs for `iterator_common.preload_into_memory`
    for the conditions of that load optimization.
    :param frames_count_resolution: Source.
    :param max_side_length: If given, the source frames will be scaled such that maximum side
    length is scaled to this value while maintaining the source aspect ratio.
    :param buffer_size: Number of frames to preload into memory. If 0, no preloading will occur.
    :return: NT that switches the `frames` field of the input with the modified buffer result.
    """

    output_frames = frames_count_resolution.frames

    if max_side_length is not None:
        output_frames = map(
            partial(
                image_common.resize_image_max_side,
                max_side_length=max_side_length,
                delete=True,
            ),
            output_frames,
        )

    if buffer_size > 0:
        output_frames = iterator_common.preload_into_memory(
            source=output_frames, buffer_size=buffer_size, fill_buffer_before_yield=True
        )

    return _FramesCountResolution(
        frames=output_frames,
        total_frame_count=frames_count_resolution.total_frame_count,
        original_resolution=frames_count_resolution.original_resolution,
    )


def create_timelapse_score(  # pylint: disable=too-many-locals,too-many-positional-arguments
    input_files: List[Path],
    output_path: Path,
    duration: float,
    output_fps: float,
    batch_size: int,
    buffer_size: int,
    conversion_scoring_functions: ConversionScoringFunctions,
    deselection_radius_frames: int,
    vectors_path: Optional[Path],
    plot_path: Optional[Path],
) -> None:
    """
    Main function to create the timelapse, defines the video processing pipeline.

    :param input_files: Videos from user to convert.
    :param output_path: Path to write the resulting output video to.
    :param duration: Desired duration for the output video in seconds.
    :param output_fps: Desired output video fps.
    :param batch_size: The number of frames from the input video to vectorize at once. The
    vectorization functions are responsible for doing the parallel loads onto the GPU.
    :param buffer_size: The number of frames to load into an in-memory buffer. This makes sure
    the GPUs have fast access to more frames rather than have the GPU waiting on disk/network IO.
    :param conversion_scoring_functions: A tuple of callables that contain the vectorization
    function and the function to score those vectors. Lets us swap the backend between different
    CV processes.
    :param deselection_radius_frames: Frames surrounding high scoring frames removed to
    prevent clustering. This is the number of frames before/after a high scoring one that are
    slightly decreased in score.
    :param vectors_path: Path to the vectors file. This file is an on-disk copy of each of the
    vectorization results for the input frames. This lets us recover from a run that has to be
    ended early, or lets us re-run selection without having to burn more compute.
    :param plot_path: If given, a visualization of the selection process is written to this file
    to aid in debugging.
    :return: None
    """

    frames_count_resolution = preload_and_scale(
        frames_count_resolution=load_input_videos(
            input_files=input_files, tqdm_desc="Reading Score Frames"
        ),
        max_side_length=conversion_scoring_functions.max_side_length,
        buffer_size=buffer_size,
    )

    more_itertools.consume(
        reduce_frames_by_score(
            scoring_frames=frames_count_resolution.frames,
            output_frames=load_input_videos(
                input_files=input_files, tqdm_desc="Reading Output Frames"
            ).frames,
            source_frame_count=frames_count_resolution.total_frame_count,
            intermediate_info=(
                IntermediateFileInfo(
                    path=vectors_path,
                    signature=create_videos_signature(
                        video_paths=input_files, modifications_salt=None
                    ),
                )
                if vectors_path is not None
                else None
            ),
            output_path=output_path,
            num_output_frames=calculate_output_frames(duration=duration, output_fps=output_fps),
            output_fps=output_fps,
            batch_size=batch_size,
            conversion_scoring_functions=conversion_scoring_functions,
            deselection_radius_frames=deselection_radius_frames,
            plot_path=plot_path,
        )
    )


def create_timelapse_crop_score(  # pylint: disable=too-many-locals,too-many-positional-arguments,too-many-arguments,unused-argument,unused-variable
    input_files: List[Path],
    output_path: Path,
    duration: float,
    output_fps: float,
    batch_size_pois: int,
    batch_size_scores: int,
    scaled_frames_buffer_size: int,
    conversion_pois_functions: ConversionPOIsFunctions,
    conversion_scoring_functions: ConversionScoringFunctions,
    aspect_ratio: AspectRatio,
    scoring_deselection_radius_frames: int,
    pois_vectors_path: Optional[Path],
    scores_vectors_path: Optional[Path],
    plot_path: Optional[Path],
) -> None:
    """

    :param input_files:
    :param output_path:
    :param duration:
    :param output_fps:
    :param batch_size_pois:
    :param batch_size_scores:
    :param scaled_frames_buffer_size:
    :param conversion_pois_functions:
    :param conversion_scoring_functions:
    :param aspect_ratio:
    :param pois_vectors_path:
    :param scores_vectors_path:
    :param plot_path:
    :return:
    """

    input_signature_pois = create_videos_signature(video_paths=input_files, modifications_salt=None)

    source = load_input_videos(input_files=input_files, tqdm_desc="Reading POI Frames")

    original_frames_for_poi_analysis, original_frames_for_poi_cropping = (
        iterator_on_disk.tee_disk_cache(
            iterator=source.frames, copies=1, serializer=iterator_on_disk.HDF5_SERIALIZER
        )
    )

    pois_frames_count_resolution = preload_and_scale(
        frames_count_resolution=_FramesCountResolution(
            frames=original_frames_for_poi_analysis,
            total_frame_count=source.total_frame_count,
            original_resolution=source.original_resolution,
        ),
        max_side_length=conversion_pois_functions.max_side_length,
        buffer_size=scaled_frames_buffer_size,
    )

    crop_resolution = image_common.largest_fitting_region(
        source_resolution=source.original_resolution, aspect_ratio=aspect_ratio
    )

    poi_crop_result: POICropResult = crop_to_pois(
        analysis_frames=pois_frames_count_resolution.frames,
        output_frames=original_frames_for_poi_cropping,
        drawing_frames=None,
        intermediate_info=(
            IntermediateFileInfo(
                path=pois_vectors_path,
                signature=create_videos_signature(video_paths=input_files, modifications_salt=None),
            )
            if pois_vectors_path is not None
            else None
        ),
        batch_size=batch_size_pois,
        total_input_frames=pois_frames_count_resolution.total_frame_count,
        convert=conversion_pois_functions.conversion,
        compute=conversion_pois_functions.compute_pois,
        original_resolution=pois_frames_count_resolution.original_resolution,
        crop_resolution=crop_resolution,
    )

    # We need to consume the resulting cropped image source twice, so it is cached to disk
    # because the cropped frames could be very large leading to memory pressure.

    cropped_frames_for_scoring, cropped_frames_for_output = iterator_on_disk.tee_disk_cache(
        iterator=poi_crop_result.cropped_to_region,
        copies=1,
        serializer=iterator_on_disk.HDF5_SERIALIZER,
    )

    scoring_frames_count_resolution = preload_and_scale(
        frames_count_resolution=_FramesCountResolution(
            frames=cropped_frames_for_scoring,
            total_frame_count=pois_frames_count_resolution.total_frame_count,
            original_resolution=crop_resolution,
        ),
        max_side_length=conversion_scoring_functions.max_side_length,
        buffer_size=scaled_frames_buffer_size,
    )

    output_frames_count_resolution = preload_and_scale(
        frames_count_resolution=_FramesCountResolution(
            frames=cropped_frames_for_output,
            total_frame_count=pois_frames_count_resolution.total_frame_count,
            original_resolution=crop_resolution,
        ),
        max_side_length=None,  # Don't scale the output frames.
        buffer_size=0,  # Could buffer here but these are the full sized frames.
    )

    more_itertools.consume(
        reduce_frames_by_score(
            scoring_frames=scoring_frames_count_resolution.frames,
            output_frames=output_frames_count_resolution.frames,
            source_frame_count=pois_frames_count_resolution.total_frame_count,
            intermediate_info=(
                IntermediateFileInfo(
                    path=scores_vectors_path,
                    signature=create_videos_signature(
                        video_paths=input_files,
                        modifications_salt=json.dumps(
                            {"winning_region": poi_crop_result.winning_region}
                        ),
                    ),
                )
                if scores_vectors_path is not None
                else None
            ),
            output_path=output_path,
            num_output_frames=calculate_output_frames(duration=duration, output_fps=output_fps),
            output_fps=output_fps,
            batch_size=batch_size_scores,
            conversion_scoring_functions=conversion_scoring_functions,
            deselection_radius_frames=scoring_deselection_radius_frames,
            plot_path=plot_path,
        )
    )
