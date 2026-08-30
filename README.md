# ASL Finger-Speller

A real-time computer-vision application that recognizes static American Sign Language (ASL) finger-spelling hand shapes and composes them into English text.

> This is a finger-spelling classifier, not a full ASL translator. Version 1.0 recognizes 24 static letters (A-Y, excluding motion-based J and Z) plus SPACE, DELETE, and NOTHING.

## Features

- Extracts 21 three-dimensional hand landmarks from webcam frames with MediaPipe
- Normalizes 63 landmark coordinates relative to the wrist and palm size
- Trains and compares Random Forest, RBF SVM, and multilayer perceptron classifiers
- Saves the best model and a JSON evaluation report with per-class metrics
- Stabilizes live predictions with temporal majority-vote smoothing
- Supports repeated letters, spaces, deletion, and clearing composed text
- Provides both an OpenCV desktop demo and a responsive Flask web interface
- Includes unit and endpoint tests plus a Python 3.11/3.12 GitHub Actions matrix
- Ships a trained RBF SVM with 99.23% accuracy on a stratified 6,240-row holdout

## Tech stack

Python, MediaPipe, OpenCV, scikit-learn, NumPy, Pandas, Flask, Joblib, Matplotlib, Pytest, HTML, CSS, JavaScript, and GitHub Actions.

## Pipeline

```text
Webcam or image
  -> MediaPipe hand detection (21 landmarks)
  -> wrist-centered, palm-scaled feature vector (63 values)
  -> scikit-learn classifier
  -> temporal vote window
  -> letter or control token
  -> composed text
```

## Setup

Python 3.11 or 3.12 is required. MediaPipe does not currently support Python 3.14.

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Prepare data and train

Collect webcam examples for a label:

```bash
asl collect --label A -n 80
```

Or convert the Kaggle ASL Alphabet training set into normalized landmark rows:

```bash
asl ingest-kaggle /path/to/asl_alphabet_train --limit 400
```

For a faster reproducible path, import a CSV containing MediaPipe columns
`lm0_x` through `lm20_z` and a `label` column:

```bash
asl ingest-landmarks /path/to/asl_landmark_features.csv --limit 1200
```

Then train all three classifiers:

```bash
asl train
```

Training writes `models/best.joblib`, `models/metrics.json`, and `models/confusion.png`.
The included v1 artifact was trained on 31,200 balanced landmark rows (1,200 per
class) derived from the ASL Alphabet dataset. The stratified 80/20 comparison
scored Random Forest 98.72%, RBF SVM 99.23%, and MLP 98.99%; the SVM was saved.
See [MODEL_CARD.md](MODEL_CARD.md) for scope and evaluation limitations.

## Run

Desktop webcam demo:

```bash
asl demo
```

Web interface:

```bash
asl web
```

Open <http://127.0.0.1:5056>. Hold a letter until it locks. Show NOTHING between repeated letters, use SPACE to insert a space, and use DELETE to backspace.

The health endpoint is available at <http://127.0.0.1:5056/health> and reports whether a trained model was found.

## Test

```bash
python -m pytest
```

## Responsible description

This project recognizes a constrained set of isolated finger-spelling hand shapes. It does not interpret ASL grammar, facial expressions, body movement, or continuous signing, and should not be described as translating American Sign Language.

## Training data

The v1 model uses normalized landmarks published by
[KhaledEisa/Sign-Language-ML](https://github.com/KhaledEisa/Sign-Language-ML),
which documents the source images as the
[ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).
The imported CSV is not redistributed in this repository.

## License

MIT
