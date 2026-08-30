from __future__ import annotations

import csv
from pathlib import Path

import cv2

from asl import LABELS
from asl.extract import _hands, extract_from_bgr
from asl.features import feature_names

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "landmarks.csv"
WEBCAM_DIR = ROOT / "data" / "webcam"


def _ensure_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "source", *feature_names()])


def append_row(label: str, source: str, features, path: Path = CSV_PATH) -> None:
    label = label.upper()
    if label not in LABELS:
        raise ValueError(f"unknown label {label!r}")
    if len(features) != len(feature_names()):
        raise ValueError(f"expected {len(feature_names())} features, got {len(features)}")
    _ensure_csv(path)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([label, source, *features.tolist()])


def collect_webcam(label: str, n: int = 80, camera: int = 0) -> int:
    if label not in LABELS:
        raise SystemExit(f"unknown label {label!r}. use one of: {', '.join(LABELS)}")
    WEBCAM_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit("could not open webcam")
    saved = 0
    hands = _hands(static=False)
    print(f"Collecting {n} frames for {label}. Press q to stop.")
    try:
        while saved < n:
            ok, frame = cap.read()
            if not ok:
                break
            vis = frame.copy()
            cv2.putText(
                vis,
                f"{label} {saved}/{n}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            feats = extract_from_bgr(frame, hands=hands)
            if feats is not None:
                append_row(label, "webcam", feats)
                saved += 1
            cv2.imshow("asl-collect", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
    print(f"saved {saved} rows to {CSV_PATH}")
    return saved
