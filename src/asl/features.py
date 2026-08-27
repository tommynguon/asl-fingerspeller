from __future__ import annotations

import numpy as np

# MediaPipe Hands: 21 landmarks. Wrist = 0, middle finger MCP = 9.
WRIST = 0
MIDDLE_MCP = 9
N_LANDMARKS = 21
FEATURE_DIM = N_LANDMARKS * 3


def landmarks_to_array(landmarks) -> np.ndarray:
    """Accept MediaPipe landmark list or (21, 3) array. Returns float32 (21, 3)."""
    if hasattr(landmarks, "landmark"):
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)
    else:
        pts = np.asarray(landmarks, dtype=np.float32)
    if pts.shape != (N_LANDMARKS, 3):
        raise ValueError(f"expected {(N_LANDMARKS, 3)}, got {pts.shape}")
    return pts


def normalize_landmarks(points: np.ndarray) -> np.ndarray:
    """Wrist at origin, scaled by palm size (wrist → middle MCP)."""
    pts = landmarks_to_array(points)
    origin = pts[WRIST].copy()
    centered = pts - origin
    scale = float(np.linalg.norm(centered[MIDDLE_MCP]))
    if scale < 1e-6:
        scale = 1.0
    return (centered / scale).astype(np.float32)


def flatten_features(points: np.ndarray) -> np.ndarray:
    return normalize_landmarks(points).reshape(FEATURE_DIM)


def feature_names() -> list[str]:
    axes = ("x", "y", "z")
    return [f"l{i}_{axis}" for i in range(N_LANDMARKS) for axis in axes]
