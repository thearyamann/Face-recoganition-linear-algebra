"""Concept 06: Reducing dimensionality with Principal Component Analysis (PCA).

Mathematical idea:
    z = W^T x

Why it matters in AI:
    A 4096-dimensional face vector is large. PCA finds a smaller set of directions
    that preserves the most important variation in the data.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from shared.dataset import load_dataset
from shared.model_helpers import save_model, train_pca
from shared.utils import PROJECT_ROOT, ensure_output_directories, save_message_figure


def main() -> int:
    """Run the PCA dimensionality reduction demo."""
    print("\n[06] PCA Dimension Reduction")
    print("PCA compresses each face vector while keeping important structure.\n")

    ensure_output_directories()
    output_image = PROJECT_ROOT / "outputs/images/06_pca_dimension_reduction.png"
    output_model = PROJECT_ROOT / "outputs/models/pca.pkl"

    try:
        X, _, _ = load_dataset()
    except (FileNotFoundError, ValueError) as error:
        print(f"[06] {error}")
        save_message_figure(output_image, "06 PCA Dimension Reduction", str(error))
        return 0

    if len(X) < 2:
        message = "PCA needs at least 2 images so it can learn variation between samples."
        print(f"[06] {message}")
        save_message_figure(output_image, "06 PCA Dimension Reduction", message)
        return 0

    pca = train_pca(X, n_components=50)
    Z = pca.transform(X)
    explained_variance = float(np.sum(pca.explained_variance_ratio_))

    print(f"[06] Original shape: {X.shape}")
    print(f"[06] Reduced shape: {Z.shape}")
    print(f"[06] PCA components used: {pca.n_components_}")
    print(f"[06] Total explained variance: {explained_variance:.4f}")

    save_model(pca, output_model)
    print(f"[06] Saved PCA model to: {output_model}")

    figure, axis = plt.subplots(figsize=(10, 4.5))
    component_indices = np.arange(1, len(pca.explained_variance_ratio_) + 1)
    axis.bar(component_indices, pca.explained_variance_ratio_, color="tab:green")
    axis.set_title("Explained Variance by Principal Component")
    axis.set_xlabel("Principal component index")
    axis.set_ylabel("Explained variance ratio")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_image, dpi=150)
    plt.close(figure)

    print(f"[06] Saved visual output to: {output_image}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
