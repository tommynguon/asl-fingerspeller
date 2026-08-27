import numpy as np
import pandas as pd

from asl.features import FEATURE_DIM, feature_names
from asl.infer import Smoother
from asl.train import load_dataset, train


def test_smoother_emits_stable_letter():
    s = Smoother(window=6, min_votes=4)
    assert s.push("NOTHING") is None
    emitted = [s.push("A") for _ in range(6)]
    assert "A" in emitted
    assert s.push("A") is None


def test_train_on_tiny_csv(tmp_path):
    names = feature_names()
    rows = []
    rng = np.random.default_rng(0)
    for i, label in enumerate(["A", "B", "C", "SPACE"]):
        for _ in range(20):
            vec = rng.normal(loc=i, scale=0.05, size=FEATURE_DIM)
            rows.append({"label": label, "source": "synth", **dict(zip(names, vec))})
    csv_path = tmp_path / "landmarks.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    metrics = train(csv_path, model_dir=tmp_path / "models")
    assert metrics["accuracy"] >= 0.9
    assert metrics["best_model"] in {"random_forest", "svm", "mlp"}
