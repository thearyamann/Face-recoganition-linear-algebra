"""Concept 10: Collect a face dataset directly from the webcam.

Mathematical idea:
    Better training vectors come from better input coverage.

Why it matters in AI:
    Face recognition improves when the training dataset contains multiple
    views of the same person, such as center, left turn, and right turn.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from shared.utils import PROJECT_ROOT, ensure_output_directories

TARGET_LABEL = "Aryamann"
TARGET_IMAGE_COUNT = 8
CAPTURE_INTERVAL_SECONDS = 1.0
FACE_CROP_SCALE = 2.0
POSE_SEQUENCE = [
    "Look straight at the camera",
    "Turn your face slightly left",
    "Turn your face slightly right",
    "Look slightly up",
    "Look slightly down",
    "Turn slightly left and smile",
    "Turn slightly right and relax",
    "Look straight again",
]


def get_face_cascade() -> cv2.CascadeClassifier:
    """Load the default frontal-face Haar cascade."""
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f"Could not load Haar cascade from: {cascade_path}")
    return cascade


def get_next_image_path(target_dir: Path) -> Path:
    """Return the next available image path inside the target dataset folder."""
    existing_numbers: list[int] = []
    for path in target_dir.glob("img*.jpg"):
        try:
            existing_numbers.append(int(path.stem.replace("img", "")))
        except ValueError:
            continue

    next_number = max(existing_numbers, default=0) + 1
    return target_dir / f"img{next_number}.jpg"


def main() -> int:
    """Capture multiple webcam face images into the Aryamann dataset folder."""
    print("\n[10] Collect Webcam Dataset")
    print(
        "This helper collects several face images from your webcam and saves them "
        "directly into the Aryamann dataset folder."
    )
    print(
        "The preview is mirrored left-to-right, and the script guides you through "
        "small face rotations to improve recognition accuracy."
    )
    print("Saved images now include a wider crop so your face is not zoomed in too much.")
    print("Press 'q' at any time to quit.\n")

    ensure_output_directories()
    target_dir = PROJECT_ROOT / "data" / TARGET_LABEL
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        face_cascade = get_face_cascade()
    except RuntimeError as error:
        print(f"[10] {error}")
        return 1

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("[10] Could not open the webcam. Check camera permissions and try again.")
        return 0

    print(f"[10] Images will be saved in: {target_dir}")
    print(f"[10] Target capture count: {TARGET_IMAGE_COUNT}\n")

    captured_count = 0
    last_capture_time = 0.0

    while captured_count < TARGET_IMAGE_COUNT:
        success, frame = camera.read()
        if not success:
            print("[10] Failed to read a frame from the webcam.")
            break

        frame = cv2.flip(frame, 1)
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = face_cascade.detectMultiScale(
            grayscale_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(90, 90),
        )

        instruction = POSE_SEQUENCE[min(captured_count, len(POSE_SEQUENCE) - 1)]
        overlay_lines = [
            f"Collecting dataset for: {TARGET_LABEL}",
            f"Image {captured_count + 1} of {TARGET_IMAGE_COUNT}",
            instruction,
            f"Crop scale: {FACE_CROP_SCALE:.1f}x face box",
            "Hold steady when the green box appears",
            "Press q to quit",
        ]

        best_face = None
        if len(detections) > 0:
            best_face = max(detections, key=lambda rect: rect[2] * rect[3])
            x_coord, y_coord, width, height = best_face
            cv2.rectangle(
                frame,
                (x_coord, y_coord),
                (x_coord + width, y_coord + height),
                (0, 200, 0),
                2,
            )

        for line_index, text in enumerate(overlay_lines):
            cv2.putText(
                frame,
                text,
                (20, 30 + line_index * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if best_face is not None and time.time() - last_capture_time >= CAPTURE_INTERVAL_SECONDS:
            x_coord, y_coord, width, height = best_face
            center_x = x_coord + width // 2
            center_y = y_coord + height // 2
            crop_size = int(max(width, height) * FACE_CROP_SCALE)
            half_crop = crop_size // 2

            x_start = max(0, center_x - half_crop)
            y_start = max(0, center_y - half_crop)
            x_end = min(frame.shape[1], center_x + half_crop)
            y_end = min(frame.shape[0], center_y + half_crop)

            face_crop = frame[y_start:y_end, x_start:x_end]
            output_path = get_next_image_path(target_dir)
            cv2.imwrite(str(output_path), face_crop)

            captured_count += 1
            last_capture_time = time.time()
            print(f"[10] Saved image {captured_count}/{TARGET_IMAGE_COUNT}: {output_path.name}")

        cv2.imshow("Collect Aryamann Webcam Dataset", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[10] Quit requested by user.")
            break

    camera.release()
    cv2.destroyAllWindows()

    if captured_count == TARGET_IMAGE_COUNT:
        print("\n[10] Dataset capture complete.")
        print("[10] Next step: retrain the models and test live recognition.")
    else:
        print(f"\n[10] Capture stopped after {captured_count} image(s).")

    print("[10] Suggested next commands:")
    print("  python concepts/05_knn_face_recognition.py")
    print("  python concepts/06_pca_dimension_reduction.py")
    print("  python concepts/08_live_webcam_recognition.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
