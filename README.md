# ASL Finger-Speller

Real-time **ASL finger-spelling** from a webcam: hand landmarks in, English letters (and spelled words) out.

This is not a full American Sign Language translator. It recognizes static finger-spelled letters (A–Y), plus SPACE / DELETE, so you can spell words. J and Z are motion signs and are out of scope for v1.

## Stack

Python 3.12, MediaPipe Hands, scikit-learn, OpenCV, Flask.

## Setup

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Status

Scaffold only. Landmark extraction, training, and the live demo land in later commits.

## License

MIT
