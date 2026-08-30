from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from asl import LABELS
from asl.collect import CSV_PATH
from asl.features import FEATURE_DIM, N_LANDMARKS, feature_names

UPSTREAM_COLUMNS = [f"lm{i}_{axis}" for i in range(N_LANDMARKS) for axis in ("x", "y", "z")]
LABEL_MAP = {"del": "DELETE", "space": "SPACE", "nothing": "NOTHING"}


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    points = values.reshape(-1, N_LANDMARKS, 3).astype(np.float32)
    centered = points - points[:, :1, :]
    scale = np.linalg.norm(centered[:, 9, :], axis=1)
    scale[scale < 1e-6] = 1.0
    return (centered / scale[:, None, None]).reshape(-1, FEATURE_DIM)


def import_landmark_csv(
    source_path: str | Path,
    *,
    output_path: str | Path = CSV_PATH,
    limit_per_label: int | None = 1200,
    random_state: int = 42,
) -> int:
    """Import MediaPipe lm0_x...lm20_z rows into the project's normalized schema."""
    source_path = Path(source_path)
    output_path = Path(output_path)
    df = pd.read_csv(source_path)
    missing = [column for column in [*UPSTREAM_COLUMNS, "label"] if column not in df.columns]
    if missing:
        raise ValueError(f"landmark CSV is missing columns: {', '.join(missing[:5])}")

    labels = df["label"].astype(str).str.strip()
    labels = labels.map(lambda value: LABEL_MAP.get(value.lower(), value.upper()))
    df = df.assign(label=labels)
    df = df[df["label"].isin(LABELS)].copy()
    if df.empty:
        raise ValueError("landmark CSV contains no supported labels")

    if limit_per_label is not None:
        if limit_per_label < 1:
            raise ValueError("limit_per_label must be positive")
        df = pd.concat(
            [
                group.sample(n=min(limit_per_label, len(group)), random_state=random_state)
                for _, group in df.groupby("label", sort=True)
            ],
            ignore_index=True,
        )

    features = _normalize_rows(df[UPSTREAM_COLUMNS].to_numpy(dtype=np.float32))
    output = pd.DataFrame(features, columns=feature_names())
    output.insert(0, "source", "online_landmarks")
    output.insert(0, "label", df["label"].to_numpy())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return len(output)
