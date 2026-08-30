from __future__ import annotations

import numpy as np
import pandas as pd

from asl.features import feature_names
from asl.import_landmarks import UPSTREAM_COLUMNS, import_landmark_csv


def test_import_landmarks_maps_filters_balances_and_normalizes(tmp_path):
    rows = []
    for label in ("A", "A", "A", "del", "del", "J"):
        points = np.zeros((21, 3), dtype=np.float32)
        points[:, 0] = np.arange(21)
        points[:, 1] = np.arange(21) * 2
        rows.append({**dict(zip(UPSTREAM_COLUMNS, points.reshape(-1))), "label": label})

    source = tmp_path / "source.csv"
    output = tmp_path / "landmarks.csv"
    pd.DataFrame(rows).to_csv(source, index=False)

    count = import_landmark_csv(source, output_path=output, limit_per_label=2)
    result = pd.read_csv(output)

    assert count == 4
    assert list(result.columns) == ["label", "source", *feature_names()]
    assert set(result["label"]) == {"A", "DELETE"}
    assert result.groupby("label").size().to_dict() == {"A": 2, "DELETE": 2}
    assert np.allclose(result[["l0_x", "l0_y", "l0_z"]], 0.0)
    palm = result[["l9_x", "l9_y", "l9_z"]].to_numpy()
    assert np.allclose(np.linalg.norm(palm, axis=1), 1.0)
