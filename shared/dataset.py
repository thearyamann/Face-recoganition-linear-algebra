"""Dataset loading utilities for face images stored in person-specific folders."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from shared.math_helpers import flatten_matrix
from shared.utils import PROJECT_ROOT, load_and_preprocess


def load_dataset(data_dir: str | Path = "data") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load face images into a dataset matrix X and label vector y."""
    root = Path(data_dir)
    if not root.is_absolute():
        root = PROJECT_ROOT / root

    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}. "
            "Create folders such as `data/Aryamann/` and add images inside."
        )

    person_directories = sorted(path for path in root.iterdir() if path.is_dir())
    if not person_directories:
        raise FileNotFoundError(
            "No identity folders were found in `data/`. "
            "Create folders like `data/Aryamann/` and place face images inside them."
        )

    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    label_names: list[str] = []
    unreadable_files: list[Path] = []

    for label_index, person_dir in enumerate(person_directories):
        image_paths = sorted(
            path
            for path in person_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        if not image_paths:
            continue

        label_names.append(person_dir.name)
        for image_path in image_paths:
            try:
                matrix = load_and_preprocess(image_path)
            except (FileNotFoundError, ValueError):
                unreadable_files.append(image_path)
                continue
            feature_rows.append(flatten_matrix(matrix))
            labels.append(len(label_names) - 1)

    if not feature_rows:
        message = (
            "No valid images could be loaded from the dataset. "
            "Check that your files are readable images inside `data/<person_name>/`."
        )
        if unreadable_files:
            message += f" Unreadable files: {', '.join(str(path.name) for path in unreadable_files[:5])}."
        raise ValueError(message)

    X = np.vstack(feature_rows).astype(np.float32)
    y = np.asarray(labels, dtype=np.int32)

    if unreadable_files:
        preview = ", ".join(path.name for path in unreadable_files[:5])
        print(
            "[dataset] Skipped unreadable image files:",
            preview if len(unreadable_files) <= 5 else f"{preview}, ...",
        )

    distribution = Counter(y.tolist())
    print("[dataset] Loaded dataset summary:")
    for index, count in sorted(distribution.items()):
        print(f"  - {label_names[index]}: {count} image(s)")

    return X, y, label_names

