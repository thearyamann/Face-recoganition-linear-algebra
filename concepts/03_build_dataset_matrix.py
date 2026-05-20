"""Concept 03: Building a dataset matrix from many face vectors.

Mathematical idea:
    X in R^(n x 4096)

Why it matters in AI:
    A machine learning dataset is a matrix where each row is one example.
    For face recognition, each row is one flattened face image.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from shared.dataset import load_dataset
from shared.utils import PROJECT_ROOT, ensure_output_directories, save_message_figure


def main() -> int:
    """Run the dataset matrix demo."""
    print("\n[03] Build Dataset Matrix")
    print("We stack face vectors row by row to create a dataset matrix X.\n")

    ensure_output_directories()
    output_path = PROJECT_ROOT / "outputs/images/03_build_dataset_matrix.png"

    try:
        X, y, label_names = load_dataset()
    except (FileNotFoundError, ValueError) as error:
        print(f"[03] {error}")
        save_message_figure(output_path, "03 Build Dataset Matrix", str(error))
        return 0

    print(f"[03] Dataset matrix shape: {X.shape}")
    print(f"[03] Label vector shape: {y.shape}")
    print("[03] Label distribution:")
    for label_index, count in sorted(Counter(y.tolist()).items()):
        print(f"  - {label_names[label_index]}: {count} sample(s)")

    preview_rows = min(10, len(X))
    figure, axis = plt.subplots(figsize=(12, 5))
    heatmap = axis.imshow(X[:preview_rows], aspect="auto", cmap="viridis")
    axis.set_title("Dataset Matrix Preview")
    axis.set_xlabel("Feature index (0 to 4095)")
    axis.set_ylabel("Sample index")
    axis.set_yticks(range(preview_rows))
    axis.set_yticklabels([label_names[y[index]] for index in range(preview_rows)])
    figure.colorbar(heatmap, ax=axis, label="Normalized intensity")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    print(f"[03] Saved visual output to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
