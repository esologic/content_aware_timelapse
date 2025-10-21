"""
Main functionality, defines the pipeline.
"""

import itertools
import json
import logging
from functools import partial
from pathlib import Path
from typing import List, NamedTuple, Optional, Set, Tuple, cast

import more_itertools
from tqdm import tqdm

from content_aware_timelapse.frames_to_vectors import vector_scoring
from content_aware_timelapse.frames_to_vectors.conversion import IntermediateFileInfo
from content_aware_timelapse.frames_to_vectors.conversion_types import (
    ConversionPOIsFunctions,
    ConversionScoringFunctions,
)
from content_aware_timelapse.frames_to_vectors.vector_points_of_interest import discover_poi_regions
from content_aware_timelapse.frames_to_vectors.vector_scoring import (
    ScoredFrames,
    reduce_frames_by_score,
)
from content_aware_timelapse.gpu_discovery import GPUDescription
from content_aware_timelapse.vector_file import create_videos_signature
from content_aware_timelapse.viderator import (
    frames_in_video,
    image_common,
    iterator_common,
    video_common,
)
from content_aware_timelapse.viderator.video_common import VideoFrames
from content_aware_timelapse.viderator.viderator_types import (
    AspectRatio,
    ImageResolution,
    ImageSourceType,
    RectangleRegion,
)

LOGGER = logging.getLogger(__name__)


class _CombinedVideos(NamedTuple):
    """
    Intermediate type for keeping track of the total number of frames in an iterator.
    """

    total_frame_count: int
    frames: ImageSourceType
    original_resolution: ImageResolution
    original_fps: float


def calculate_output_frames(duration: float, output_fps: float) -> int:
    """
    Canonical function to do this math.
    :param duration: Desired length of the video in seconds.
    :param output_fps: FPS of output.
    :return: Round number for output frames.
    """

    return int(duration * output_fps)


def load_input_videos(input_files: List[Path], tqdm_desc: Optional[str] = None) -> _CombinedVideos:
    """
    Helper function to combine the input videos.

    :param input_files: List of input videos.
    :param tqdm_desc: If provided, wrap the combined frame iterator in a tqdm progress bar
    with this description. If None, tqdm is disabled.
    :return: NT containing the total frame count and a joined iterator of all the input
    frames.
    """

    input_video_frames: List[VideoFrames] = list(
        map(frames_in_video.frames_in_video_opencv, input_files)
    )

    all_input_frames = itertools.chain.from_iterable(
        video_frames.frames for video_frames in input_video_frames
    )

    total_frame_count = sum(video_frames.total_frame_count for video_frames in input_video_frames)

    input_resolutions: Set[ImageResolution] = {
        video_frames.original_resolution for video_frames in input_video_frames
    }

    input_fps: Set[float] = {video_frames.original_fps for video_frames in input_video_frames}

    if len(input_resolutions) > 1:
        raise ValueError(f"Input videos have different resolutions: {input_video_frames}")

    # Conditionally wrap in tqdm
    if tqdm_desc is not None:
        frames_iter = tqdm(
            all_input_frames,
            total=total_frame_count,
            unit="Frames",
            ncols=100,
            desc=tqdm_desc,
            maxinterval=1,
        )
    else:
        frames_iter = all_input_frames

    return _CombinedVideos(
        total_frame_count=total_frame_count,
        frames=cast(ImageSourceType, frames_iter),
        original_resolution=next(iter(input_resolutions)),
        original_fps=next(iter(input_fps)),
    )


def preload_and_scale(
    video_source: _CombinedVideos,
    max_side_length: Optional[int],
    buffer_size: int,
) -> _CombinedVideos:
    """
    Wraps a few library functions to optionally scale and pre-load input videos from disk into
    RAM for faster processing. See the underlying docs for `iterator_common.preload_into_memory`
    for the conditions of that load optimization.
    :param video_source: Source.
    :param max_side_length: If given, the source frames will be scaled such that maximum side
    length is scaled to this value while maintaining the source aspect ratio.
    :param buffer_size: Number of frames to preload into memory. If 0, no preloading will occur.
    :return: NT that switches the `frames` field of the input with the modified buffer result.
    """

    output_frames = video_source.frames

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

    return _CombinedVideos(
        frames=output_frames,
        total_frame_count=video_source.total_frame_count,
        original_resolution=video_source.original_resolution,
        original_fps=video_source.original_fps,
    )


def create_timelapse_score(  # pylint: disable=too-many-locals,too-many-positional-arguments,too-many-arguments
    input_files: List[Path],
    output_path: Path,
    duration: float,
    output_fps: float,
    batch_size: int,
    buffer_size: int,
    conversion_scoring_functions: ConversionScoringFunctions,
    deselection_radius_frames: int,
    audio_paths: List[Path],
    gpus: Tuple[GPUDescription, ...],
    vectors_path: Optional[Path],
    plot_path: Optional[Path],
    best_frame_path: Optional[Path],
) -> None:
    """
    Library backend for the UI function. See docs in `reduce_frames_by_score` or docs in click
    CLI function for reference.

    :param input_files: See docs in library or click.
    :param output_path: See docs in library or click.
    :param duration: See docs in library or click.
    :param output_fps: See docs in library or click.
    :param batch_size: See docs in library or click.
    :param buffer_size: See docs in library or click.
    :param conversion_scoring_functions: See docs in library or click.
    :param deselection_radius_frames: See docs in library or click.
    :param audio_paths: See docs in library or click.
    :param gpus: See docs in library or click.
    :param vectors_path: See docs in library or click.
    :param plot_path: See docs in library or click.
    :param best_frame_path: See docs in library or click.
    :return: None
    """

    frames_count_resolution = preload_and_scale(
        video_source=load_input_videos(input_files=input_files, tqdm_desc="Reading Score Frames"),
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
            audio_paths=audio_paths,
            plot_path=plot_path,
            gpus=gpus,
            best_frame_path=best_frame_path,
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
    audio_paths: List[Path],
    gpus: Tuple[GPUDescription, ...],
    layout_matrix: List[List[int]],
    pois_vectors_path: Optional[Path],
    scores_vectors_path: Optional[Path],
    plot_path: Optional[Path],
) -> None:
    """
    Library backend for the UI function. See docs in `reduce_frames_by_score`, `crop_to_pois` or
    docs in click CLI function for complete reference.

    :param input_files: See docs in library or click.
    :param output_path: See docs in library or click.
    :param duration: See docs in library or click.
    :param output_fps: See docs in library or click.
    :param batch_size_pois: See docs in library or click.
    :param batch_size_scores: See docs in library or click.
    :param scaled_frames_buffer_size: See docs in library or click.
    :param conversion_pois_functions: See docs in library or click.
    :param conversion_scoring_functions: See docs in library or click.
    :param aspect_ratio: See docs in library or click.
    :param scoring_deselection_radius_frames: See docs in library or click.
    :param audio_paths: See docs in library or click.
    :param gpus: See docs in library or click.
    :param layout_matrix: See docs in library or click.
    :param pois_vectors_path: See docs in library or click.
    :param scores_vectors_path: See docs in library or click.
    :param plot_path: See docs in library or click.
    :return: None
    """

    num_regions: int = len(list(itertools.chain.from_iterable(layout_matrix)))

    primary_source = load_input_videos(input_files=input_files, tqdm_desc="Reading POI Frames")

    crop_resolution = image_common.largest_fitting_region(
        source_resolution=primary_source.original_resolution, aspect_ratio=aspect_ratio
    )

    # Input video is read twice, this could be optimized.

    poi_regions: Tuple[RectangleRegion, ...] = discover_poi_regions(
        analysis_frames=preload_and_scale(
            video_source=_CombinedVideos(
                frames=primary_source.frames,
                total_frame_count=primary_source.total_frame_count,
                original_resolution=primary_source.original_resolution,
                original_fps=primary_source.original_fps,
            ),
            max_side_length=conversion_pois_functions.max_side_length,
            buffer_size=scaled_frames_buffer_size,
        ).frames,
        intermediate_info=(
            IntermediateFileInfo(
                path=pois_vectors_path,
                signature=create_videos_signature(video_paths=input_files, modifications_salt=None),
            )
            if pois_vectors_path is not None
            else None
        ),
        batch_size=batch_size_pois,
        source_frame_count=primary_source.total_frame_count,
        conversion_pois_functions=conversion_pois_functions,
        original_resolution=primary_source.original_resolution,
        crop_resolution=crop_resolution,
        gpus=gpus,
        num_regions=num_regions,
    )

    cropped_to_best_region: ImageSourceType = video_common.crop_source(
        source=load_input_videos(  # Don't scale or buffer output frames.
            input_files=input_files, tqdm_desc="Reading Crop Output Frames"
        ).frames,
        region=next(iter(poi_regions)),
    )

    scored_frames: ScoredFrames = vector_scoring.discover_high_scoring_frames(
        scoring_frames=preload_and_scale(
            video_source=_CombinedVideos(
                frames=tqdm(
                    cropped_to_best_region,
                    total=primary_source.total_frame_count,
                    unit="Frames",
                    ncols=100,
                    desc="Reading Cropped Frames for Scoring",
                    maxinterval=1,
                ),
                total_frame_count=primary_source.total_frame_count,
                original_resolution=crop_resolution,
                original_fps=primary_source.original_fps,
            ),
            max_side_length=conversion_scoring_functions.max_side_length,
            buffer_size=scaled_frames_buffer_size,
        ).frames,
        source_frame_count=primary_source.total_frame_count,
        intermediate_info=(
            IntermediateFileInfo(
                path=scores_vectors_path,
                signature=create_videos_signature(
                    video_paths=input_files,
                    modifications_salt=json.dumps({"winning_regions": list(poi_regions)}),
                ),
            )
            if scores_vectors_path is not None
            else None
        ),
        num_output_frames=calculate_output_frames(duration=duration, output_fps=output_fps),
        batch_size=batch_size_scores,
        conversion_scoring_functions=conversion_scoring_functions,
        deselection_radius_frames=scoring_deselection_radius_frames,
        plot_path=plot_path,
        gpus=gpus,
    )

    def output_frames() -> ImageSourceType:
        """
        Creates the output frames.
        :return: Iterator of output frames to be written to disk.
        """

        for frame_index, input_frame in enumerate(
            load_input_videos(
                input_files=input_files,
            ).frames
        ):
            if frame_index in scored_frames.winning_indices:

                yield image_common.reshape_from_regions(
                    image=input_frame,
                    prioritized_poi_regions=poi_regions,
                    layout_matrix=layout_matrix,
                )

    video_common.write_source_to_disk_consume(
        source=output_frames(),
        video_path=output_path,
        video_fps=output_fps,
        audio_paths=audio_paths,
        high_quality=False,
    )
