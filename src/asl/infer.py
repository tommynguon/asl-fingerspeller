from __future__ import annotations

from collections import Counter, deque
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "best.joblib"


class Smoother:
    def __init__(self, window: int = 8, min_votes: int = 5):
        self.window = window
        self.min_votes = min_votes
        self.buf: deque[str] = deque(maxlen=window)
        self.last_emitted: str | None = None

    def push(self, label: str, nothing: str = "NOTHING") -> str | None:
        if label == nothing:
            self.buf.clear()
            return None
        self.buf.append(label)
        if len(self.buf) < self.min_votes:
            return None
        winner, count = Counter(self.buf).most_common(1)[0]
        if count < self.min_votes:
            return None
        if winner == self.last_emitted:
            return None
        self.last_emitted = winner
        self.buf.clear()
        return winner


def apply_letter(buffer: list[str], token: str) -> list[str]:
    if token == "SPACE":
        buffer.append(" ")
    elif token == "DELETE":
        if buffer:
            buffer.pop()
    elif token not in {"NOTHING", "J", "Z"}:
        buffer.append(token)
    return buffer
    if not path.exists():
        return None
    return joblib.load(path)


def predict_label(model, features: np.ndarray) -> tuple[str, float]:
    X = np.asarray(features, dtype=np.float32).reshape(1, -1)
    label = str(model.predict(X)[0])
    conf = 1.0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        conf = float(np.max(proba))
    elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("clf"), "predict_proba"):
        proba = model.predict_proba(X)[0]
        conf = float(np.max(proba))
    return label, conf
