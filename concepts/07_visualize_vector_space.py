"""Concept 07: Visualizing face vectors in a lower-dimensional space.

Mathematical idea:
    Project 4096-dimensional face vectors into 2D with PCA.

Why it matters in AI:
    We cannot draw 4096 dimensions directly, but we can visualize the main patterns
    in 2D to understand how faces cluster in vector space.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from shared.dataset import load_dataset
from shared.utils import PROJECT_ROOT, ensure_output_directories, save_message_figure


def main() -> int:
    """Run the 2D vector space visualization demo."""
    print("\n[07] Visualize Vector Space")
    print("We use PCA to place high-dimensional face vectors onto a 2D plot.\n")

    ensure_output_directories()
    output_path = PROJECT_ROOT / "outputs/plots/07_pca_2d.png"

    try:
        X, y, label_names = load_dataset()
    except (FileNotFoundError, ValueError) as error:
        print(f"[07] {error}")
        save_message_figure(output_path, "07 Visualize Vector Space", str(error))
        return 0

    if len(X) < 2:
        message = "At least 2 images are needed to create a 2D PCA visualization."
        print(f"[07] {message}")
        save_message_figure(output_path, "07 Visualize Vector Space", message)
        return 0

    projection_model = PCA(n_components=min(2, len(X), X.shape[1]), random_state=42)
    projection = projection_model.fit_transform(X)

    if projection.shape[1] < 2:
        message = "The dataset is too small to form a full 2D PCA plot."
        print(f"[07] {message}")
        save_message_figure(output_path, "07 Visualize Vector Space", message)
        return 0

    figure, axis = plt.subplots(figsize=(8, 6))
    for sample_index, point in enumerate(projection):
        label = label_names[y[sample_index]]
        axis.scatter(point[0], point[1], s=60)
        axis.text(point[0] + 0.01, point[1] + 0.01, label, fontsize=9)

    axis.set_title("Faces Projected into 2D PCA Space")
    axis.set_xlabel("Principal Component 1")
    axis.set_ylabel("Principal Component 2")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    print(f"[07] Saved 2D PCA plot to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
