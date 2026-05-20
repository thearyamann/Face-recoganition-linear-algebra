"""Concept 02: Flattening an image matrix into a feature vector.

Mathematical idea:
    x in R^4096

Why it matters in AI:
    Many machine learning models expect a 1D list of numbers, not a 2D grid.
    Flattening turns the image matrix into a vector that algorithms can compare.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from shared.math_helpers import flatten_matrix
from shared.utils import (
    PROJECT_ROOT,
    ensure_output_directories,
    get_first_image,
    load_and_preprocess,
    save_message_figure,
)


def main() -> int:
    """Run the flatten-to-vector demo."""
    print("\n[02] Flatten Matrix to Vector")
    print(
        "Each face image is converted into a 4096-dimensional vector. "
        "This vector is the mathematical representation of the face.\n"
    )

    ensure_output_directories()
    output_path = PROJECT_ROOT / "outputs/images/02_flatten_to_vector.png"

    try:
        image_path = get_first_image()
        matrix = load_and_preprocess(image_path)
    except (FileNotFoundError, ValueError) as error:
        print(f"[02] {error}")
        save_message_figure(output_path, "02 Flatten to Vector", str(error))
        return 0

    vector = flatten_matrix(matrix)
    print(f"[02] Loaded image: {image_path}")
    print(f"[02] Matrix shape: {matrix.shape}")
    print(f"[02] Vector length: {len(vector)}")
    print("[02] First 20 vector values:")
    print(np.array_str(vector[:20], precision=3, suppress_small=True))

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(matrix, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("64x64 Matrix")
    axes[0].axis("off")

    axes[1].plot(vector, color="tab:blue", linewidth=1.2)
    axes[1].set_title("Flattened 4096-D Vector")
    axes[1].set_xlabel("Vector index")
    axes[1].set_ylabel("Intensity")
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    print(f"[02] Saved visual output to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
