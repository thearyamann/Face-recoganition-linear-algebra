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
from shared.model_helpers import load_model, save_model, train_knn, train_pca
from shared.utils import (
    IMAGE_SIZE,
    PROJECT_ROOT,
    ensure_output_directories,
    preprocess_image_array,
)

MODEL_DIR = PROJECT_ROOT / "outputs/models"
KNN_MODEL_PATH = MODEL_DIR / "knn.pkl"
PCA_MODEL_PATH = MODEL_DIR / "pca.pkl"
UNKNOWN_DISTANCE_THRESHOLD = 6.0


def prepare_models() -> tuple[Any, Any, list[str]]:
    """Load or train the PCA and KNN models required for webcam recognition."""
    X, y, label_names = load_dataset()
    if len(np.unique(y)) < 2:
        raise ValueError(
            "Live recognition needs at least 2 identity folders in `data/`. "
            "Add another person before using the webcam demo."
        )

    pca = None
    knn = None

    if PCA_MODEL_PATH.exists():
        try:
            pca = load_model(PCA_MODEL_PATH)
            print(f"[08] Loaded PCA model from: {PCA_MODEL_PATH}")
        except Exception as error:  # pragma: no cover - defensive logging
            print(f"[08] Could not load PCA model: {error}")

    if pca is None:
        pca = train_pca(X, n_components=50)
        save_model(pca, PCA_MODEL_PATH)
        print(f"[08] Trained and saved PCA model to: {PCA_MODEL_PATH}")

    X_projected = pca.transform(X)

    if KNN_MODEL_PATH.exists():
        try:
            candidate = load_model(KNN_MODEL_PATH)
            if getattr(candidate, "n_features_in_", None) == X_projected.shape[1]:
                knn = candidate
                print(f"[08] Loaded KNN model from: {KNN_MODEL_PATH}")
            else:
                print(
                    "[08] Existing KNN model uses a different feature space. "
                    "Retraining it for PCA features."
                )
        except Exception as error:  # pragma: no cover - defensive logging
            print(f"[08] Could not load KNN model: {error}")

    if knn is None:
        knn = train_knn(X_projected, y, n_neighbors=1)
        save_model(knn, KNN_MODEL_PATH)
        print(f"[08] Trained and saved KNN model to: {KNN_MODEL_PATH}")

    return pca, knn, label_names


def get_face_cascade() -> cv2.CascadeClassifier:
    """Load the default OpenCV Haar cascade for frontal face detection."""
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f"Could not load Haar cascade from: {cascade_path}")
    return cascade


def main() -> int:
    """Run the live webcam face recognition demo."""
    print("\n[08] Live Webcam Recognition")
    print("This demo recognizes faces in real time using matrices, vectors, PCA, and KNN.")
    print("The camera preview is mirrored left-to-right for a natural laptop-camera view.")
    print("Press 'q' in the webcam window to quit.\n")

    ensure_output_directories()

    try:
        pca, knn, label_names = prepare_models()
    except (FileNotFoundError, ValueError) as error:
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
            face_region = frame[y_coord : y_coord + height, x_coord : x_coord + width]
            matrix = preprocess_image_array(face_region)
            vector = flatten_matrix(matrix)
            projected = pca.transform(vector.reshape(1, -1))
            predicted_index = int(knn.predict(projected)[0])
            neighbor_distances, _ = knn.kneighbors(projected, return_distance=True)
            distance = float(neighbor_distances[0][0])

            if distance > UNKNOWN_DISTANCE_THRESHOLD:
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
