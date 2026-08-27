# ASL Finger-Speller

Webcam **American Sign Language finger-spelling** → English letters → spelled words.

This is **not** a full ASL translator. It classifies static hand shapes (A–Y) plus SPACE / DELETE. **J** and **Z** are motion signs and are out of v1.

## How it works

1. MediaPipe Hands finds 21 landmarks
2. Features are wrist-centered and scaled by palm size (63 floats)
3. scikit-learn (Random Forest / SVM / MLP) picks the letter
4. A short vote window turns jittery frames into one stable character

## Setup (Python 3.12 — not 3.14)

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Collect → train → demo

```bat
python -m asl collect --label A -n 80
python -m asl ingest-kaggle path\to\asl_alphabet_train
python -m asl train
python -m asl demo
python -m asl web
```

`ingest-kaggle` expects the [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) train folder (subfolders `A`, `B`, …). Raw images are gitignored; only `data/landmarks.csv` should be committed.

Webcam demo: hold a letter until it locks, SPACE inserts a space, DELETE backspaces, `c` clears, `q` quits.

Flask UI: http://127.0.0.1:5056

## Tests

```bat
python -m pytest tests
```

## Resume (fill numbers after `asl train`)

- Built a real-time ASL finger-spelling app (webcam → MediaPipe landmarks → sklearn classifier → spelled text).
- Wrist-normalized 21 3D landmarks; compared Random Forest, SVM, and MLP on a held-out set (**put accuracy here**).
- Added temporal majority-vote smoothing and SPACE/DELETE so letters compose into words.

Do not write “translates American Sign Language.”

## License

MIT
