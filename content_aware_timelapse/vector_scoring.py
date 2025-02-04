"""
Turn vectorized images to numerical scores, then pick the best images for the output.
"""

from typing import List, Tuple, TypedDict

import numpy as np
import pandas as pd
from numpy import typing as npt
from sklearn import preprocessing


class IndexScores(TypedDict):
    """
    Intermediate type for linking the Euclidean distance between the next frame and the index
    of the frame.
    """

    frame_index: int
    entropy: float
    variance: float
    saliency: float
    energy: float
    center_bias: float


def calculate_scores(packed: Tuple[int, npt.NDArray[np.float16]]) -> IndexScores:
    """
    Calculate a combined score from various metrics of the attention map.

    Metrics:
        - Entropy: Measures how distributed the attention is.
        - Variance: Measures spread in attention.
        - Saliency: Measures maximum attention.
        - Energy: Measures total attention.
        - Center-bias: Favors attention focused near the center.

    :param packed: Tuple of the index and the attention map.
    :return: Combined score and index.
    """
    index, attention_map = packed
    attention_map = attention_map.astype(np.float32)

    # Normalize the attention map
    normalized_map = attention_map / (np.sum(attention_map) + 1e-8)

    entropy_score = -np.sum(
        normalized_map[normalized_map > 0] * np.log(normalized_map[normalized_map > 0] + 1e-8)
    )

    variance_score = np.var(attention_map)
    saliency_score = np.max(attention_map)
    energy_score = np.sum(attention_map)

    # Center-bias
    h, w = attention_map.shape
    y, x = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
    center_weights = np.exp(-(x**2 + y**2))
    center_bias_score = np.sum(attention_map * center_weights)

    return IndexScores(
        frame_index=index,
        entropy=entropy_score,
        variance=float(variance_score),
        saliency=float(saliency_score),
        energy=float(energy_score),
        center_bias=center_bias_score,
    )


def select_frames(index_scores: List[IndexScores], num_output_frames: int) -> List[int]:
    """
    Selects the top `num_output_frames` while avoiding clusters.

    :param index_scores: List of dictionaries containing frame index and feature scores.
    :param num_output_frames: Number of frames to select.
    :return: List of selected frame indices.
    """

    def _basic_score(index_score: IndexScores) -> float:
        return (
            0.2 * (1 - index_score["entropy"])
            + 0.3 * index_score["variance"]
            + 0.2 * index_score["saliency"]
            + 0.2 * index_score["energy"]
            + 0.1 * index_score["center_bias"]
        )

    raw_df = pd.DataFrame.from_records(index_scores).set_index("frame_index")

    # Normalize the score columns
    scaler = preprocessing.MinMaxScaler()
    scaled_df = pd.DataFrame(
        scaler.fit_transform(raw_df), columns=raw_df.columns, index=raw_df.index
    )
    scaled_df["overall_score"] = scaled_df.apply(_basic_score, axis=1)

    sorted_df = scaled_df.sort_values(by="overall_score", ascending=False)
    selected_indices: List[int] = []
    scores = sorted_df["overall_score"].copy()

    penalty_radius = max(1, len(sorted_df) // num_output_frames)  # Ensure at least 1

    while len(selected_indices) < num_output_frames and not scores.empty:
        best_frame = int(scores.idxmax())  # Get highest-score frame
        selected_indices.append(best_frame)

        # Find its position in sorted_df
        best_frame_pos = sorted_df.index.get_loc(best_frame)

        # Define a safe penalty window within sorted order
        nearby_start = max(0, best_frame_pos - penalty_radius)
        nearby_end = min(len(sorted_df), best_frame_pos + penalty_radius)

        # Extract nearby frame indices from sorted order
        nearby_frames = sorted_df.index[nearby_start:nearby_end].to_list()

        # Apply penalty (only to existing frames in `scores`)
        scores.loc[scores.index.intersection(nearby_frames)] *= 0.5

        # Drop the selected frame
        scores = scores.drop(best_frame)

    return sorted(selected_indices)
