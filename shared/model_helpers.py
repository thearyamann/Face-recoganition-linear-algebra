"""Model training and persistence helpers for PCA and KNN demos."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from shared.utils import PROJECT_ROOT, ensure_output_directories


def train_knn(
    X: np.ndarray, y: np.ndarray, n_neighbors: int = 3
) -> KNeighborsClassifier:
    """Train a KNN classifier with a safe neighbor count."""
    if X.size == 0 or y.size == 0:
        raise ValueError("Cannot train KNN on an empty dataset.")
    if len(np.unique(y)) < 2:
        raise ValueError(
            "KNN face recognition needs at least 2 identities in the dataset."
        )

    effective_neighbors = min(n_neighbors, len(X))
    model = KNeighborsClassifier(
        n_neighbors=effective_neighbors,
        weights="distance",
        metric="euclidean",
    )
    model.fit(X, y)
    return model


def train_pca(X: np.ndarray, n_components: int = 50) -> PCA:
    """Train PCA while respecting the maximum valid component count."""
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("PCA requires a non-empty 2D dataset matrix.")
    max_components = min(X.shape[0], X.shape[1], n_components)
    if max_components < 1:
        raise ValueError("PCA needs at least one valid principal component.")

    model = PCA(
        n_components=max_components,
        svd_solver="auto",
        whiten=True,
        random_state=42,
    )
    model.fit(X)
    return model


def save_model(model: Any, path: str | Path) -> None:
    """Save a model object to disk using joblib."""
    ensure_output_directories()
    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def load_model(path: str | Path) -> Any:
    """Load a model object from disk using joblib."""
    model_path = Path(path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)
