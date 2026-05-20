"""Concept 04: Measuring similarity with Euclidean distance.

Mathematical idea:
    d(x, y) = sqrt(sum((x_i - y_i)^2))

Why it matters in AI:
    Face recognition starts by comparing how close two face vectors are.
    Smaller distance usually means the faces are more similar.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt

from shared.dataset import load_dataset
from shared.math_helpers import euclidean_distance
from shared.utils import IMAGE_SIZE, PROJECT_ROOT, ensure_output_directories, save_message_figure


def main() -> int:
    """Run the Euclidean distance demo."""
    print("\n[04] Euclidean Distance")
    print("We compare two faces by measuring the distance between their vectors.\n")

    ensure_output_directories()
    output_path = PROJECT_ROOT / "outputs/images/04_euclidean_distance.png"

    try:
        X, y, label_names = load_dataset()
    except (FileNotFoundError, ValueError) as error:
        print(f"[04] {error}")
        save_message_figure(output_path, "04 Euclidean Distance", str(error))
        return 0

    if len(X) < 2:
        message = "At least 2 face images are needed to compare Euclidean distance."
        print(f"[04] {message}")
        save_message_figure(output_path, "04 Euclidean Distance", message)
        return 0

    first_vector = X[0]
    second_vector = X[1]
    distance = euclidean_distance(first_vector, second_vector)

    first_matrix = first_vector.reshape(IMAGE_SIZE)
    second_matrix = second_vector.reshape(IMAGE_SIZE)

    same_identity = y[0] == y[1]
    print(f"[04] Face 1 label: {label_names[y[0]]}")
    print(f"[04] Face 2 label: {label_names[y[1]]}")
    print(f"[04] Euclidean distance: {distance:.4f}")
    if same_identity:
        print("[04] These two samples come from the same identity folder.")
    else:
        print("[04] These two samples come from different identity folders.")
    print(
        "[04] Interpretation: smaller distances mean the two face vectors are closer in "
        "the 4096-dimensional space."
    )

    figure, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(first_matrix, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title(f"Face 1: {label_names[y[0]]}")
    axes[0].axis("off")
    axes[1].imshow(second_matrix, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"Face 2: {label_names[y[1]]}")
    axes[1].axis("off")
    figure.suptitle(f"Euclidean Distance = {distance:.4f}", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    print(f"[04] Saved visual output to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
