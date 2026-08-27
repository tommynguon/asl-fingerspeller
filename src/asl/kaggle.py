from __future__ import annotations

from pathlib import Path

import cv2

from asl.extract import extract_from_bgr, extract_from_path
from asl.collect import append_row

KAGGLE_LABEL_MAP = {
    "space": "SPACE",
    "del": "DELETE",
    "nothing": "NOTHING",
}


def iter_kaggle_images(root: Path):
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        raw = folder.name
        label = KAGGLE_LABEL_MAP.get(raw.lower(), raw.upper())
        if label in {"J", "Z"}:
            continue
        for image in folder.glob("*.jpg"):
            yield label, image
        for image in folder.glob("*.png"):
            yield label, image


def ingest_kaggle(root: str | Path, limit_per_label: int = 400) -> int:
    root = Path(root)
    if not root.exists():
        raise SystemExit(f"kaggle folder not found: {root}")
    counts: dict[str, int] = {}
    n = 0
    hands = None
    try:
        import mediapipe as mp

        hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
        )
        for label, path in iter_kaggle_images(root):
            if counts.get(label, 0) >= limit_per_label:
                continue
            feats = extract_from_path(path, hands=hands)
            if feats is None:
                continue
            append_row(label, "kaggle", feats)
            counts[label] = counts.get(label, 0) + 1
            n += 1
            if n % 200 == 0:
                print(f"... {n} rows")
    finally:
        if hands is not None:
            hands.close()
    _ = cv2
    print(f"ingested {n} landmark rows from {root}")
    return n
