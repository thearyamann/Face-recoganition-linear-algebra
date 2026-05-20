"""Concept 08: Live face recognition with webcam input, PCA, and KNN.

Mathematical idea:
    image matrix -> vector -> PCA projection -> nearest-neighbor classification

Why it matters in AI:
    This script connects the linear algebra pipeline to a real-time AI system.
    The webcam face is turned into numbers, compressed with PCA, and classified with KNN.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from shared.dataset import load_dataset
from shared.math_helpers import flatten_matrix
from shared.model_helpers import save_model, train_knn, train_pca
from shared.utils import (
    IMAGE_SIZE,
    PROJECT_ROOT,
    ensure_output_directories,
    get_face_cascade,
    preprocess_image_array,
)

MODEL_DIR = PROJECT_ROOT / "outputs/models"
KNN_MODEL_PATH = MODEL_DIR / "knn.pkl"
PCA_MODEL_PATH = MODEL_DIR / "pca.pkl"
DEFAULT_UNKNOWN_DISTANCE_THRESHOLD = 3.0


def estimate_unknown_threshold(X_projected: np.ndarray, y: np.ndarray) -> float:
    """Estimate a reasonable unknown threshold from nearest same-label distances."""
    same_label_distances: list[float] = []
    for sample_index, sample in enumerate(X_projected):
        label_mask = y == y[sample_index]
        label_indices = np.where(label_mask)[0]
        label_indices = label_indices[label_indices != sample_index]
        if len(label_indices) == 0:
            continue
        comparison_vectors = X_projected[label_indices]
        distances = np.linalg.norm(comparison_vectors - sample, axis=1)
        same_label_distances.append(float(np.min(distances)))

    if not same_label_distances:
        return DEFAULT_UNKNOWN_DISTANCE_THRESHOLD

    distance_array = np.asarray(same_label_distances, dtype=np.float32)
    threshold = float(np.mean(distance_array) + 1.5 * np.std(distance_array))
    return max(threshold, DEFAULT_UNKNOWN_DISTANCE_THRESHOLD)


def prepare_models() -> tuple[Any, Any, list[str], float]:
    """Train PCA and KNN models from the current dataset for best live accuracy."""
    X, y, label_names = load_dataset()
    if len(np.unique(y)) < 2:
        raise ValueError(
            "Live recognition needs at least 2 identity folders in `data/`. "
            "Add another person before using the webcam demo."
        )

    print("[08] Training fresh PCA and KNN models from the current dataset...")
    pca = train_pca(X, n_components=50)
    X_projected = pca.transform(X)
    knn = train_knn(X_projected, y, n_neighbors=3)
    unknown_threshold = estimate_unknown_threshold(X_projected, y)

    save_model(pca, PCA_MODEL_PATH)
    save_model(knn, KNN_MODEL_PATH)
    print(f"[08] Saved PCA model to: {PCA_MODEL_PATH}")
    print(f"[08] Saved KNN model to: {KNN_MODEL_PATH}")
    print(f"[08] PCA feature size: {pca.n_components_}")
    print(f"[08] Estimated unknown threshold: {unknown_threshold:.2f}")
    return pca, knn, label_names, unknown_threshold


def main() -> int:
    """Run the live webcam face recognition demo."""
    print("\n[08] Live Webcam Recognition")
    print("This demo recognizes faces in real time using matrices, vectors, PCA, and KNN.")
    print("The camera preview is mirrored left-to-right for a natural laptop-camera view.")
    print("Press 'q' in the webcam window to quit.\n")

    ensure_output_directories()

    try:
        pca, knn, label_names, unknown_threshold = prepare_models()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"[08] {error}")
        return 0

    try:
        face_cascade = get_face_cascade()
    except RuntimeError as error:
        print(f"[08] {error}")
        return 1

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("[08] Could not open the webcam. Check camera permissions and try again.")
        return 0

    print("[08] Webcam opened successfully. Starting recognition loop...")

    while True:
        success, frame = camera.read()
        if not success:
            print("[08] Failed to read a frame from the webcam.")
            break

        # Mirror the preview so it behaves like a typical laptop selfie camera.
        frame = cv2.flip(frame, 1)
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = face_cascade.detectMultiScale(
            grayscale_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        for (x_coord, y_coord, width, height) in detections:
            padding_x = int(width * 0.25)
            padding_y = int(height * 0.25)
            x_start = max(0, x_coord - padding_x)
            y_start = max(0, y_coord - padding_y)
            x_end = min(frame.shape[1], x_coord + width + padding_x)
            y_end = min(frame.shape[0], y_coord + height + padding_y)

            face_region = frame[y_start:y_end, x_start:x_end]
            matrix = preprocess_image_array(face_region)
            vector = flatten_matrix(matrix)
            projected = pca.transform(vector.reshape(1, -1))
            predicted_index = int(knn.predict(projected)[0])
            neighbor_distances, _ = knn.kneighbors(projected, return_distance=True)
            distance = float(neighbor_distances[0][0])

            if distance > unknown_threshold:
                predicted_name = "Unknown"
                box_color = (0, 0, 255)
            else:
                predicted_name = label_names[predicted_index]
                box_color = (0, 200, 0)

            cv2.rectangle(
                frame, (x_coord, y_coord), (x_coord + width, y_coord + height), box_color, 2
            )
            overlays = [
                f"Identity: {predicted_name}",
                f"Matrix: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
                f"Vector: {IMAGE_SIZE[0] * IMAGE_SIZE[1]}",
                f"PCA: {pca.n_components_}",
                f"Distance: {distance:.2f}",
                f"Threshold: {unknown_threshold:.2f}",
            ]

            for line_index, text in enumerate(overlays):
                position = (x_coord, max(20, y_coord - 60 + line_index * 18))
                cv2.putText(
                    frame,
                    text,
                    position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    box_color,
                    1,
                    cv2.LINE_AA,
                )

        cv2.imshow("How AI Recognizes Faces Using Linear Algebra", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[08] Quit requested by user.")
            break

    camera.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
