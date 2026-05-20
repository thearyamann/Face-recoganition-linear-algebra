"""Mathematical helper functions used across the teaching demos."""

from __future__ import annotations

import numpy as np


def normalize_image(matrix: np.ndarray) -> np.ndarray:
    """Normalize an image matrix to the range [0, 1]."""
    array = np.asarray(matrix, dtype=np.float32)
    if array.size == 0:
        raise ValueError("Cannot normalize an empty image matrix.")
    if array.max() > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def flatten_matrix(matrix: np.ndarray) -> np.ndarray:
    """Flatten a 2D image matrix into a 1D feature vector."""
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2D matrix to flatten, but received shape {array.shape}."
        )
    return array.reshape(-1)


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the Euclidean distance between two vectors."""
    vector_x = np.asarray(x, dtype=np.float32).reshape(-1)
    vector_y = np.asarray(y, dtype=np.float32).reshape(-1)
    if vector_x.shape != vector_y.shape:
        raise ValueError(
            "Euclidean distance requires vectors with the same shape. "
            f"Received {vector_x.shape} and {vector_y.shape}."
        )
    return float(np.linalg.norm(vector_x - vector_y))

