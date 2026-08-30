import numpy as np
import pandas as pd
import joblib
from sklearn.dummy import DummyClassifier

from asl.features import FEATURE_DIM, feature_names
from asl.infer import Smoother, load_model, predict_label
from asl.train import load_dataset, train


def test_smoother_emits_stable_letter():
    s = Smoother(window=6, min_votes=4)
    assert s.push("NOTHING") is None
    emitted = [s.push("A") for _ in range(6)]
    assert "A" in emitted
    assert s.push("A") is None


def test_smoother_allows_repeated_letter_after_release():
    s = Smoother(window=4, min_votes=3)
    assert [s.push("L") for _ in range(3)][-1] == "L"
    assert s.push("NOTHING") is None
    assert [s.push("L") for _ in range(3)][-1] == "L"


def test_model_round_trip_and_prediction(tmp_path):
    X = np.vstack([np.zeros(FEATURE_DIM), np.ones(FEATURE_DIM)]).astype(np.float32)
    model = DummyClassifier(strategy="constant", constant="A").fit(X, np.array(["A", "B"]))
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    loaded = load_model(path)
    assert loaded is not None
    label, confidence = predict_label(loaded, np.zeros(FEATURE_DIM))
    assert label == "A"
    assert confidence == 1.0


def test_load_model_returns_none_for_missing_path(tmp_path):
    assert load_model(tmp_path / "missing.joblib") is None


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
