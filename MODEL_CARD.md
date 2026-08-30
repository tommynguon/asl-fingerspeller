# ASL Finger-Speller v1 Model Card

## Intended use

This model recognizes 24 static ASL finger-spelling letters (A-Y excluding the
motion-based J and Z) plus SPACE and DELETE from a single detected hand. A frame
with no detected hand acts as the NOTHING/release state in the application.

It is a learning and portfolio project, not an accessibility aid, interpreter,
or full ASL translator.

## Training data

- Source landmarks: [KhaledEisa/Sign-Language-ML](https://github.com/KhaledEisa/Sign-Language-ML)
- Documented image source: [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- Imported rows: 31,200 (1,200 deterministic samples per supported class)
- Features: 21 MediaPipe hand landmarks, wrist-centered and scaled by the
  wrist-to-middle-finger-MCP distance, producing 63 numeric features

The source landmark CSV is not redistributed here. Reproduce the processed CSV
with `asl ingest-landmarks ... --limit 1200`.

## Evaluation

The data was split 80/20 with stratification and random state 42. Results on the
6,240-row holdout were:

- Random Forest: 98.72%
- RBF SVM: 99.23% (selected)
- Multilayer perceptron: 98.99%

These are same-dataset holdout results. They may overstate performance on new
cameras, backgrounds, skin tones, hand proportions, orientations, or signing
styles. A person-independent external test set and broader webcam collection
would be required before making generalization claims.

## Artifacts

- `models/best.joblib`: compressed scikit-learn RBF SVM pipeline
- `models/metrics.json`: aggregate and per-class metrics plus confusion matrix
- `models/confusion.png`: holdout confusion-matrix image
