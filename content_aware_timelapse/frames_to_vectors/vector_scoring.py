"""
Turn vectorized images to numerical scores, then pick the best images for the output.
"""

import itertools
import logging
from pathlib import Path
from typing import Iterator, List, NamedTuple, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import typing as npt
from sklearn import preprocessing

from content_aware_timelapse.frames_to_vectors import conversion
from content_aware_timelapse.frames_to_vectors.conversion import IntermediateFileInfo
from content_aware_timelapse.frames_to_vectors.conversion_types import (
    ConversionScoringFunctions,
    IndexScores,
    ScoreWeights,
)
from content_aware_timelapse.gpu_discovery import GPUDescription
from content_aware_timelapse.viderator import image_common, video_common
from content_aware_timelapse.viderator.viderator_types import ImageSourceType, RGBInt8ImageType

LOGGER = logging.getLogger(__name__)


def _apply_radius_deselection(
    indices_sorted_by_score: List[int],
    num_output_frames: int,
    initial_radius: int,
    minimum_radius: int = 0,
) -> List[int]:
    """
    Tries to remove frames nearby to frames with high scores to avoid clustering.

    :param indices_sorted_by_score: Vector (frame) indices sorted by score, with the most
    interesting frames first.
    :param num_output_frames: The desired number of output frames requested by the user.
    :param initial_radius: The number of frames before/after high scoring frames that should be
    dropped. The function will half this value until `num_output_frames` is reached.
    :param minimum_radius: The smallest value the `initial_radius` can shink to.
    :return: A list of frame indices.
    """

    all_indices = np.array(sorted(indices_sorted_by_score))

    def try_deselection(selection_radius: int) -> List[int]:
        """
        Runs the selection process given the current radius value.
        :param selection_radius: The number of frames to drop before/after high value frames.
        :return: A list of suitable indices.
        """

        suppression_mask = np.full(len(all_indices), True)
        local_selection = []

        for idx in indices_sorted_by_score:

            if not suppression_mask[all_indices == idx][0]:
                continue

            local_selection.append(idx)

            suppression_mask &= np.abs(all_indices - idx) > selection_radius

            if len(local_selection) == num_output_frames:
                break

        return local_selection

    # Try decreasing radii until enough frames are selected
    radius = initial_radius  # Initial suppression radius
    selected_indices = []

    while radius >= minimum_radius:
        selected_indices = try_deselection(radius)

        if len(selected_indices) >= num_output_frames:
            break

        radius = max(radius // 2, minimum_radius)

    return selected_indices[:num_output_frames]  # Just in case we over-selected


class ScoredFrames(NamedTuple):
    """
    Intermediate type.
    """

    best_frame_index: int
    winning_indices: List[int]


def _score_and_sort_frames(
    score_weights: ScoreWeights,
    index_scores: List[IndexScores],
    num_output_frames: int,
    deselection_radius_frames: int,
    plot_path: Optional[Path],
) -> ScoredFrames:
    """
    Selects the top `num_output_frames` while avoiding clusters.

    :param score_weights: Weights applied to the different properties after they are normalized from
    0-1. The overall score is the sum of the weighted properties.
    :param index_scores: List of dictionaries containing frame index and feature scores.
    :param num_output_frames: Number of frames to select.
    :param deselection_radius_frames: Frames surrounding high scoring frames removed to
    prevent clustering. This is the number of frames before/after a high scoring one that are
    slightly decreased in score.
    :param plot_path: If given, creates an overall visualization of the math that went into
    selecting the frames.
    :return: List of selected frame indices.
    """

    def _basic_score(index_score: IndexScores) -> float:
        """
        Computes an overall score for a row given the weighted components of the underlying score.
        :param index_score: From the row, the input parts of the score.
        :return: The overall score.
        """

        return (
            (index_score["entropy"] * score_weights.low_entropy)
            + (index_score["variance"] * score_weights.variance)
            + (index_score["saliency"] * score_weights.saliency)
            + (index_score["energy"] * score_weights.energy)
        )

    raw_df = pd.DataFrame.from_records(index_scores).set_index("frame_index")

    # Normalize the score columns
    scaler = preprocessing.MinMaxScaler()
    scaled_df = pd.DataFrame(
        scaler.fit_transform(raw_df), columns=raw_df.columns, index=raw_df.index
    )

    scaled_df["entropy"] = 1.0 - scaled_df["entropy"]  # lower entropy = better

    scaled_df["overall_score"] = scaled_df.apply(_basic_score, axis=1)

    winning_indices = _apply_radius_deselection(
        indices_sorted_by_score=scaled_df["overall_score"]
        .sort_values(ascending=False)
        .index.to_numpy(),
        num_output_frames=num_output_frames,
        initial_radius=deselection_radius_frames,
    )

    if plot_path is not None:
        score_axes, other_axes = scaled_df.plot(
            kind="line",
            subplots=[("overall_score",), ("entropy", "variance", "saliency", "energy")],
            title="Frame Scores",
            xlabel="Frame Index",
            ylabel="Score",
            sharey=True,
            figsize=(12, 6),
            grid=True,
        )

        score_axes.set_title(label="Overall Score", fontsize="small")
        score_axes.legend().set_visible(False)

        other_axes.set_title(label="Score Composition", fontsize="small")

        for i in winning_indices:
            score_axes.axvline(i, color="lime", alpha=0.5)

        plt.tight_layout()
        plt.savefig(plot_path)

    return ScoredFrames(
        best_frame_index=int(scaled_df["overall_score"].idxmax()), winning_indices=winning_indices
    )


def discover_high_scoring_frames(  # pylint: disable=too-many-positional-arguments
    scoring_frames: ImageSourceType,
    intermediate_info: Optional[IntermediateFileInfo],
    batch_size: int,
    source_frame_count: int,
    conversion_scoring_functions: ConversionScoringFunctions,
    gpus: Tuple[GPUDescription, ...],
    num_output_frames: int,
    deselection_radius_frames: int,
    plot_path: Optional[Path],
) -> ScoredFrames:
    """
    Function to only get the list of high scoring frames in the input.

    :param scoring_frames: These frames are fed to the GPU via the conversion function and will
    likely be scaled down. Pre-scaled and buffered images should be passed here to avoid waiting
    around.
    :param source_frame_count: Number of frames in the input.
    :param intermediate_info: If given, vectors will be written to disk for faster
    re-runs.
    :param num_output_frames: Number of frames to select.
    :param batch_size: Number of scaled input images to send to the GPU at once.
    :param conversion_scoring_functions: Functions to compute vectors then score them.
    :param deselection_radius_frames: Frames surrounding high scoring frames removed to
    prevent clustering. This is the number of frames before/after a high scoring one that are
    slightly decreased in score.
    :param gpus: Describes the GPUs we're allowed to use for this computation.
    :param plot_path: If given, a visualization of the math that went into scores will be written
    to this path.
    :return:
    """

    vectors: Iterator[npt.NDArray[np.float16]] = conversion.frames_to_vectors(
        frames=scoring_frames,
        intermediate_info=intermediate_info,
        batch_size=batch_size,
        total_input_frames=source_frame_count,
        convert_batches=conversion_scoring_functions.conversion,
        gpus=gpus,
    )

    LOGGER.debug("Starting to sort output vectors by score.")

    score_indexes: List[IndexScores] = list(
        map(
            conversion_scoring_functions.scoring,
            enumerate(vectors),
        )
    )

    scored_sorted = _score_and_sort_frames(
        score_weights=conversion_scoring_functions.weights,
        index_scores=score_indexes,
        num_output_frames=num_output_frames,
        deselection_radius_frames=deselection_radius_frames,
        plot_path=plot_path,
    )

    return scored_sorted


def reduce_frames_by_score(  # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    scoring_frames: ImageSourceType,
    output_frames: ImageSourceType,
    source_frame_count: int,
    intermediate_info: Optional[IntermediateFileInfo],
    output_path: Path,
    num_output_frames: int,
    output_fps: float,
    batch_size: int,
    conversion_scoring_functions: ConversionScoringFunctions,
    deselection_radius_frames: int,
    gpus: Tuple[GPUDescription, ...],
    audio_paths: List[Path],
    plot_path: Optional[Path],
    best_frame_path: Optional[Path],
) -> ImageSourceType:
    """
    Uses input scoring function to reduce the input frames to the desired count by selecting
    the highest scoring frames.

    :param scoring_frames: These frames are fed to the GPU via the conversion function and will
    likely be scaled down. Pre-scaled and buffered images should be passed here to avoid waiting
    around.
    :param output_frames: These are the frames that are actually cropped and forwarded in the
    output.
    :param source_frame_count: Number of frames in the input.
    :param intermediate_info: If given, vectors will be written to disk for faster
    re-runs.
    :param output_path: Desired output location of the video.
    :param num_output_frames: Number of frames to select.
    :param output_fps: Set as the FPS of the output video.
    :param batch_size: Number of scaled input images to send to the GPU at once.
    :param conversion_scoring_functions: Functions to compute vectors then score them.
    :param deselection_radius_frames: Frames surrounding high scoring frames removed to
    prevent clustering. This is the number of frames before/after a high scoring one that are
    slightly decreased in score.
    :param gpus: Describes the GPUs we're allowed to use for this computation.
    :param audio_paths: If given, the audio files will be written to the output video.
    :param plot_path: If given, a visualization of the math that went into scores will be written
    to this path.
    :param best_frame_path: If given, the highest scoring frame in the input will be written to
    this path. Good for thumbnails.
    :return: Selected frames.
    """

    scored_sorted = discover_high_scoring_frames(
        scoring_frames=scoring_frames,
        intermediate_info=intermediate_info,
        batch_size=batch_size,
        source_frame_count=source_frame_count,
        conversion_scoring_functions=conversion_scoring_functions,
        gpus=gpus,
        num_output_frames=num_output_frames,
        deselection_radius_frames=deselection_radius_frames,
        plot_path=plot_path,
    )

    most_interesting_indices: Set[int] = set(sorted(scored_sorted.winning_indices))

    final_frame_index: int = max(most_interesting_indices)

    # Slice the frames to include the frame at final_frame_index
    # Ensure the frame at final_frame_index is included, the plus one.
    sliced_frames: ImageSourceType = itertools.islice(output_frames, None, final_frame_index + 1)

    def filter_frame(index_frame: Tuple[int, RGBInt8ImageType]) -> bool:
        """
        Hijacks the enumeration of the output frames to write the best frame.
        :param index_frame: Indices and the contents of the frames.
        :return: Filter output, True if the frame is interesting False if otherwise.
        """

        frame_index, frame = index_frame

        if frame_index == scored_sorted.best_frame_index and best_frame_path is not None:
            image_common.save_rgb_image(path=best_frame_path, image=frame)

        return frame_index in most_interesting_indices

    most_interesting_frames: ImageSourceType = (
        index_frame[1]
        for index_frame in filter(
            filter_frame,
            enumerate(sliced_frames),
        )
    )

    return video_common.write_source_to_disk_forward(
        source=most_interesting_frames,
        video_path=output_path,
        video_fps=output_fps,
        audio_paths=audio_paths,
        high_quality=False,
    )
