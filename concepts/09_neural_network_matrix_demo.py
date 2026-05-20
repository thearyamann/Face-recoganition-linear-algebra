"""Concept 09: A neural network layer is a matrix multiplication plus a bias.

Mathematical idea:
    y = W x + b

Why it matters in AI:
    Deep learning systems build on linear algebra. Each layer combines inputs
    with weights and biases to produce a new representation.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from shared.utils import PROJECT_ROOT, ensure_output_directories


def main() -> int:
    """Run the neural network matrix demo."""
    print("\n[09] Neural Network Matrix Demo")
    print("A neural network layer applies matrix multiplication followed by a bias add.\n")

    ensure_output_directories()
    output_path = PROJECT_ROOT / "outputs/images/09_neural_network_matrix_demo.png"

    rng = np.random.default_rng(seed=42)
    input_size = 8
    output_size = 4

    x = rng.normal(size=(input_size, 1)).astype(np.float32)
    W = rng.normal(size=(output_size, input_size)).astype(np.float32)
    b = rng.normal(size=(output_size, 1)).astype(np.float32)
    y = W @ x + b

    print(f"[09] Input vector x shape: {x.shape}")
    print(f"[09] Weight matrix W shape: {W.shape}")
    print(f"[09] Bias vector b shape: {b.shape}")
    print(f"[09] Output vector y shape: {y.shape}")
    print("[09] Output values:")
    print(np.array_str(y.reshape(-1), precision=3, suppress_small=True))

    figure, axes = plt.subplots(1, 4, figsize=(12, 4))
    matrices = [(x, "Input x"), (W, "Weights W"), (b, "Bias b"), (y, "Output y")]
    for axis, (matrix, title) in zip(axes, matrices):
        image = axis.imshow(matrix, aspect="auto", cmap="coolwarm")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    figure.suptitle("Neural Network Computation: y = W x + b", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    print(f"[09] Saved visual output to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
