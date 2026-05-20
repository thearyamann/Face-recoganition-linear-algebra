"""Concept 05: Face recognition with K-Nearest Neighbors (KNN).

Mathematical idea:
    Predict the label of a face using the labels of the nearest vectors.

Why it matters in AI:
    KNN is an intuitive classifier. It recognizes a new face by asking:
    "Which known faces are closest to this one?"
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
from shared.model_helpers import save_model, train_knn
from shared.utils import IMAGE_SIZE, PROJECT_ROOT, ensure_output_directories, save_message_figure


def main() -> int:
    """Run the KNN face recognition demo."""
    print("\n[05] KNN Face Recognition")
    print("KNN predicts the identity of a face using nearby examples in vector space.\n")

    ensure_output_directories()
    output_path = PROJECT_ROOT / "outputs/images/05_knn_face_recognition.png"
    model_path = PROJECT_ROOT / "outputs/models/knn.pkl"

    try:
        X, y, label_names = load_dataset()
    except (FileNotFoundError, ValueError) as error:
        print(f"[05] {error}")
        save_message_figure(output_path, "05 KNN Face Recognition", str(error))
        return 0

    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        message = (
            "KNN face recognition needs at least 2 identity folders in `data/`. "
            "Add another person so the classifier has labels to compare."
        )
        print(f"[05] {message}")
        save_message_figure(output_path, "05 KNN Face Recognition", message)
        return 0

    test_index = len(X) - 1
    X_train = X[:test_index]
    y_train = y[:test_index]
    X_test = X[test_index : test_index + 1]
    y_test = y[test_index]

    if len(np.unique(y_train)) < 2:
        message = (
            "After holding out one sample, the remaining training set still needs at least 2 identities. "
            "Add more images for each person."
        )
        print(f"[05] {message}")
        save_message_figure(output_path, "05 KNN Face Recognition", message)
        return 0

    model = train_knn(X_train, y_train, n_neighbors=1)
    predicted_label = int(model.predict(X_test)[0])

    print(f"[05] Training matrix shape: {X_train.shape}")
    print(f"[05] Test vector shape: {X_test.shape}")
    print(f"[05] Actual label: {label_names[y_test]}")
    print(f"[05] Predicted label: {label_names[predicted_label]}")

    save_model(model, model_path)
    print(f"[05] Saved KNN model to: {model_path}")

    figure, axis = plt.subplots(figsize=(5, 5))
    axis.imshow(X_test[0].reshape(IMAGE_SIZE), cmap="gray", vmin=0.0, vmax=1.0)
    axis.set_title(
        f"Predicted: {label_names[predicted_label]}\nActual: {label_names[y_test]}"
    )
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    print(f"[05] Saved visual output to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
