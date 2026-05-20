"""Concept 01: A face image can be represented as a matrix.

Mathematical idea:
    A in R^(64 x 64)

Why it matters in AI:
    Computers do not see a face as "eyes, nose, and mouth" at first.
    They see a grid of pixel intensities, which is a matrix.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from shared.utils import (
    PROJECT_ROOT,
    ensure_output_directories,
    get_first_image,
    load_and_preprocess,
    save_message_figure,
)


def main() -> int:
    """Run the image-as-matrix demo."""
    print("\n[01] Image as Matrix")
    print("Each face image starts as a matrix of numbers.")
    print("After preprocessing, the face becomes a 64x64 grayscale matrix.\n")

    ensure_output_directories()
    output_path = PROJECT_ROOT / "outputs/images/01_image_as_matrix.png"

    try:
        image_path = get_first_image()
        matrix = load_and_preprocess(image_path)
    except (FileNotFoundError, ValueError) as error:
        print(f"[01] {error}")
        save_message_figure(output_path, "01 Image as Matrix", str(error))
        return 0

    print(f"[01] Loaded image: {image_path}")
    print(f"[01] Matrix shape: {matrix.shape}")
    print("[01] Top-left 5x5 block:")
    print(np.array_str(matrix[:5, :5], precision=3, suppress_small=True))

    figure, axis = plt.subplots(figsize=(6, 6))
    image_plot = axis.imshow(matrix, cmap="gray", vmin=0.0, vmax=1.0)
    axis.set_title("A Face Image as a 64x64 Matrix")
    axis.set_xlabel("Column index")
    axis.set_ylabel("Row index")
    figure.colorbar(image_plot, ax=axis, fraction=0.046, pad=0.04, label="Intensity")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    print(f"[01] Saved visual output to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
