"""
Turn vectorized images to numerical scores, then pick the best images for the output.
"""

from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

import matplotlib.pyplot as plt
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


def calculate_scores(packed: Tuple[int, npt.NDArray[np.float16]]) -> IndexScores:
    """
    Calculate scores from the CLS token embedding of an image.

    Metrics are reinterpreted for feature vectors:
        - Energy: Measures the L2 norm (magnitude) of the feature vector, indicating overall
                  feature activation strength.
        - Saliency: Measures the maximum activation value within the vector, pointing to the
                    strongest single feature.
        - Variance: Measures the spread of values in the vector, indicating diversity or
                    concentration of feature activations.
        - Entropy: Measures the distribution of (normalized positive) feature activations.
                   Its interpretation for 'interestingness' is less direct for feature vectors
                   compared to spatial attention maps, but can indicate feature sparsity/density.

    :param packed: Tuple of the image's index and its CLS token embedding.
    :return: Calculated scores and index.
    """

    index, cls_embedding = packed
    cls_embedding = cls_embedding.astype(np.float32)  # Ensure float32 for calculations

    # --- Re-interpreting Metrics for a CLS Feature Vector (768 dimensions) ---

    # Energy: L2 Norm of the CLS embedding. A higher norm generally means a stronger,
    # more confident, or more distinct feature representation.
    energy_score = np.linalg.norm(cls_embedding)

    # Indicates the strongest single feature component the model picked up.
    saliency_score = np.percentile(cls_embedding, 90)

    # Variance: Variance of the values in the CLS embedding.
    # High variance suggests some feature dimensions are highly active while others are not.
    # Low variance suggests more uniform feature activations.
    variance_score = np.var(cls_embedding)

    # Entropy: Entropy of the distribution of (normalized positive) feature values.
    # If the embedding has negative values, consider taking abs or handling carefully.
    # Here, we normalize only positive values to make it behave like a probability distribution.
    positive_values = cls_embedding[cls_embedding > 0]
    if positive_values.size > 0:
        # Normalize positive values to sum to 1 to form a probability distribution
        normalized_positive_values = positive_values / (np.sum(positive_values) + 1e-8)
        entropy_score = -np.sum(
            normalized_positive_values * np.log(normalized_positive_values + 1e-8)
        )
    else:
        # If no positive values, entropy is conventionally 0 (or adjust as needed)
        entropy_score = 0.0

    return IndexScores(
        frame_index=index,
        entropy=entropy_score,
        variance=float(variance_score),
        saliency=float(saliency_score),
        energy=float(energy_score),
    )


def _score_and_sort_frames(
    index_scores: List[IndexScores], num_output_frames: int, plot_path: Optional[Path]
) -> List[int]:
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

        return (
            (index_score["entropy"] * 0.25)
            + (index_score["variance"] * 0.3)
            + (index_score["saliency"] * 0.2)
            + (index_score["energy"] * 0.1)
        )

    raw_df = pd.DataFrame.from_records(index_scores).set_index("frame_index")

    # Normalize the score columns
    scaler = preprocessing.MinMaxScaler()
    scaled_df = pd.DataFrame(
        scaler.fit_transform(raw_df), columns=raw_df.columns, index=raw_df.index
    )
    scaled_df["overall_score"] = scaled_df.apply(_basic_score, axis=1)

    sorted_df = scaled_df.sort_values(by="overall_score", ascending=False)

    winning_indices: List[int] = sorted_df.head(num_output_frames).index.to_list()

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
    index_scores: List[IndexScores], num_output_frames: int, plot_path: Optional[Path] = None
) -> List[int]:
    """
    Selects the top `num_output_frames` while avoiding clusters.

    :param index_scores: List of dictionaries containing frame index and feature scores.
    :param num_output_frames: Number of frames to select.
    :param plot_path: If given, a visualization representing why each frame was chosen will be
    created and written to this path.
    :return: List of selected frame indices.
    """

    return sorted(
        _score_and_sort_frames(
            index_scores=index_scores,
            num_output_frames=num_output_frames,
            plot_path=plot_path,
        )
    )
