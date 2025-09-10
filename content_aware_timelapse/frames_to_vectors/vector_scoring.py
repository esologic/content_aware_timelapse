"""
Turn vectorized images to numerical scores, then pick the best images for the output.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing

from content_aware_timelapse.frames_to_vectors.conversion_types import IndexScores, ScoreWeights


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


def _score_and_sort_frames(
    score_weights: ScoreWeights,
    index_scores: List[IndexScores],
    num_output_frames: int,
    plot_path: Optional[Path],
) -> List[int]:
    """
    Selects the top `num_output_frames` while avoiding clusters.

    :param score_weights: Weights applied to the different properties after they are normalized from
    0-1. The overall score is the sum of the weighted properties.
    :param index_scores: List of dictionaries containing frame index and feature scores.
    :param num_output_frames: Number of frames to select.
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
        initial_radius=1000,
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

    return winning_indices


def select_frames(
    score_weights: ScoreWeights,
    index_scores: List[IndexScores],
    num_output_frames: int,
    plot_path: Optional[Path] = None,
) -> List[int]:
    """
    Selects the top `num_output_frames` while avoiding clusters.

    :param score_weights: Weights applied to the different properties after they are normalized from
    0-1. The overall score is the sum of the weighted properties.
    :param index_scores: List of dictionaries containing frame index and feature scores.
    :param num_output_frames: Number of frames to select.
    :param plot_path: If given, a visualization representing why each frame was chosen will be
    created and written to this path.
    :return: List of selected frame indices, sorted numerically not by their score.
    """

    return sorted(
        _score_and_sort_frames(
            score_weights=score_weights,
            index_scores=index_scores,
            num_output_frames=num_output_frames,
            plot_path=plot_path,
        )
    )
