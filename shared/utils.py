"""General utility helpers for image loading, preprocessing, and output folders."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np

from shared.math_helpers import normalize_image

IMAGE_SIZE: tuple[int, int] = (64, 64)
SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FACE_CASCADE: cv2.CascadeClassifier | None = None


def ensure_output_directories() -> None:
    """Create output directories used by the project if they do not exist."""
    for relative_path in ("outputs/images", "outputs/plots", "outputs/models"):
        (PROJECT_ROOT / relative_path).mkdir(parents=True, exist_ok=True)


def iter_image_files(data_dir: str | Path = "data") -> Iterable[Path]:
    """Yield supported image files under the dataset directory."""
    root = Path(data_dir)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def get_face_cascade() -> cv2.CascadeClassifier:
    """Load and cache the OpenCV Haar cascade used for face-focused preprocessing."""
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade from: {cascade_path}")
        _FACE_CASCADE = cascade
    return _FACE_CASCADE


def crop_to_largest_face(image: np.ndarray, padding_ratio: float = 0.35) -> np.ndarray:
    """Crop around the largest detected face, with padding for forehead and chin."""
    if image.ndim == 2:
        grayscale = image
    else:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detections = get_face_cascade().detectMultiScale(
        grayscale,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    if len(detections) == 0:
        return image

    x_coord, y_coord, width, height = max(detections, key=lambda rect: rect[2] * rect[3])
    padding_x = int(width * padding_ratio)
    padding_y = int(height * padding_ratio)

    x_start = max(0, x_coord - padding_x)
    y_start = max(0, y_coord - padding_y)
    x_end = min(image.shape[1], x_coord + width + padding_x)
    y_end = min(image.shape[0], y_coord + height + padding_y)
    return image[y_start:y_end, x_start:x_end]


def preprocess_image_array(image: np.ndarray) -> np.ndarray:
    """Convert a raw image array to a normalized grayscale 64x64 matrix."""
    if image is None or image.size == 0:
        raise ValueError("Received an empty image array for preprocessing.")

    if image.ndim == 2:
        grayscale = image
    elif image.ndim == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(
            "Image arrays must be either grayscale (2D) or color (3D). "
            f"Received shape {image.shape}."
        )

    # Focus on the face when possible so background pixels matter less.
    face_focused = crop_to_largest_face(image) if max(image.shape[:2]) > 96 else image
    if face_focused.ndim == 2:
        grayscale = face_focused
    else:
        grayscale = cv2.cvtColor(face_focused, cv2.COLOR_BGR2GRAY)

    # Normalize lighting differences to make webcam and training photos more comparable.
    grayscale = cv2.equalizeHist(grayscale)
    resized = cv2.resize(grayscale, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return normalize_image(resized)


def load_and_preprocess(image_path: str | Path) -> np.ndarray:
    """Load an image from disk and convert it to a normalized 64x64 matrix."""
    path = Path(image_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(
            f"OpenCV could not read the image at {path}. "
            "Please verify that the file is a valid image."
        )
    return preprocess_image_array(image)


def get_first_image(data_dir: str | Path = "data") -> Path:
    """Return the first available image in the dataset."""
    image_files = list(iter_image_files(data_dir))
    if not image_files:
        raise FileNotFoundError(
            "No images were found in the dataset. Add files to "
            "`data/<person_name>/` and run the script again."
        )
    return image_files[0]


def save_message_figure(output_path: str | Path, title: str, message: str) -> None:
    """Save a simple placeholder figure with a message."""
    ensure_output_directories()
    path = Path(output_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.axis("off")
    axis.set_title(title, fontsize=14, pad=12)
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
