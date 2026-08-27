from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from asl.features import flatten_features


def _hands():
    import mediapipe as mp

    return mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
        model_complexity=1,
    )


def extract_from_bgr(frame: np.ndarray, hands=None) -> np.ndarray | None:
    """Return 63-d normalized features, or None if no hand."""
    if frame is None or frame.size == 0:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    close = False
    if hands is None:
        hands = _hands()
        close = True
    try:
        result = hands.process(rgb)
    finally:
        if close:
            hands.close()
    if not result.multi_hand_landmarks:
        return None
    return flatten_features(result.multi_hand_landmarks[0])


def extract_from_path(path: str | Path, hands=None) -> np.ndarray | None:
    image = cv2.imread(str(path))
    return extract_from_bgr(image, hands=hands)
