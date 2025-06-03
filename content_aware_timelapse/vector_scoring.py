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


def calculate_scores_old(packed: Tuple[int, npt.NDArray[np.float16]]) -> IndexScores:
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


def calculate_scores(packed: Tuple[int, npt.NDArray[np.float16]]) -> IndexScores:
    """
    Calculate a combined score from various metrics of the CLIP embedding.

    Metrics:
        - Entropy: Measures how distributed the embedding values are.
        - Variance: Measures spread in embedding values.
        - Saliency: Measures maximum embedding value.
        - Energy: Measures total embedding magnitude (L2 norm).

    :param packed: Tuple of the index and the CLIP embedding vector.
    :return: Combined score and index.
    """
    index, embedding = packed
    embedding = embedding.astype(np.float32)  # Convert to higher precision for calculations

    # Normalize embedding
    norm_embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

    entropy_score = -np.sum(
        norm_embedding[norm_embedding > 0] * np.log(norm_embedding[norm_embedding > 0] + 1e-8)
    )

    variance_score = np.var(embedding)
    saliency_score = np.max(embedding)
    energy_score = np.linalg.norm(embedding)  # L2 norm as energy measure

    return IndexScores(
        frame_index=index,
        entropy=entropy_score,
        variance=float(variance_score),
        saliency=float(saliency_score),
        energy=float(energy_score),
        center_bias=0.0,  # Center-bias not applicable for 1D embeddings
    )


def select_frames(index_scores: List[IndexScores], num_output_frames: int) -> List[int]:
    """
    Selects the top `num_output_frames` while avoiding clusters.

    :param index_scores: List of dictionaries containing frame index and feature scores.
    :param num_output_frames: Number of frames to select.
    :return: List of selected frame indices.
    """

    def _basic_score(index_score: IndexScores) -> float:
        """

        :param index_score:
        :return:
        """
        return index_score["saliency"]

    raw_df = pd.DataFrame.from_records(index_scores).set_index("frame_index")

    # Normalize the score columns
    scaler = preprocessing.MinMaxScaler()
    scaled_df = pd.DataFrame(
        scaler.fit_transform(raw_df), columns=raw_df.columns, index=raw_df.index
    )
    scaled_df["overall_score"] = scaled_df.apply(_basic_score, axis=1)

    sorted_df = scaled_df.sort_values(by="overall_score", ascending=False).head(num_output_frames)

    return sorted(sorted_df.index.to_list())
