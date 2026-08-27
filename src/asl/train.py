from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "landmarks.csv"
MODEL_DIR = ROOT / "models"


def load_dataset(path: Path = CSV_PATH) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m asl ingest-kaggle <folder>  or  python -m asl collect --label A"
        )
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c.startswith("l") and c[1:2].isdigit()]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label"].astype(str).to_numpy()
    source = df["source"].astype(str).to_numpy() if "source" in df.columns else np.array(["unknown"] * len(df))
    return X, y, source


def _models() -> dict:
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=24, n_jobs=-1, random_state=42
        ),
        "svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=8, gamma="scale")),
            ]
        ),
        "mlp": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        max_iter=400,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def train(path: Path = CSV_PATH, model_dir: Path = MODEL_DIR) -> dict:
    X, y, source = load_dataset(path)
    X_train, X_test, y_train, y_test, src_train, src_test = train_test_split(
        X, y, source, test_size=0.2, random_state=42, stratify=y
    )
    scores = {}
    fitted = {}
    for name, model in _models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, pred))
        scores[name] = acc
        fitted[name] = (model, pred)
        print(f"{name}: {acc:.4f}")

    best_name = max(scores, key=scores.get)
    best_model, best_pred = fitted[best_name]
    webcam_mask = src_test == "webcam"
    webcam_acc = None
    if webcam_mask.any():
        webcam_acc = float(accuracy_score(y_test[webcam_mask], best_model.predict(X_test[webcam_mask])))

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_dir / "best.joblib")
    labels = sorted(set(y.tolist()))
    cm = confusion_matrix(y_test, best_pred, labels=labels)
    metrics = {
        "best_model": best_name,
        "accuracy": scores[best_name],
        "scores": scores,
        "n_samples": int(len(y)),
        "n_test": int(len(y_test)),
        "webcam_test_accuracy": webcam_acc,
        "labels": labels,
        "classification_report": classification_report(y_test, best_pred, output_dict=True, zero_division=0),
        "confusion_matrix": cm.tolist(),
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(cm)
        ax.set_title(f"{best_name} ({scores[best_name]:.3f})")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        fig.tight_layout()
        fig.savefig(model_dir / "confusion.png", dpi=120)
        plt.close(fig)
    except Exception as exc:
        print("confusion plot skipped:", exc)
    print(f"wrote {model_dir / 'best.joblib'}  best={best_name} acc={scores[best_name]:.4f}")
    return metrics


def main() -> int:
    train()
    return 0
