from __future__ import annotations

import sys
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flask import Flask, Response, render_template

from asl.infer import apply_letter
from asl.extract import _hands, extract_from_bgr
from asl.infer import Smoother, load_model, predict_label

app = Flask(__name__, template_folder="templates", static_folder="static")
STATE = {"buffer": [], "letter": "NOTHING", "conf": 0.0}
STATE_LOCK = Lock()
SMOOTHER = Smoother()


def gen_frames():
    import cv2

    model = load_model()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("could not open webcam")
    hands = _hands(static=False)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            feats = extract_from_bgr(frame, hands=hands)
            letter, conf = "NOTHING", 0.0
            if model is not None and feats is not None:
                letter, conf = predict_label(model, feats)
            with STATE_LOCK:
                token = SMOOTHER.push(letter)
                if token:
                    apply_letter(STATE["buffer"], token)
                STATE["letter"] = letter
                STATE["conf"] = conf
                word = "".join(STATE["buffer"])
            cv2.putText(frame, f"{letter} {conf:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, word[-40:], (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            _, jpeg = cv2.imencode(".jpg", frame)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
    finally:
        hands.close()
        cap.release()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/video")
def video():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/state")
def state():
    with STATE_LOCK:
        return {
            "letter": STATE["letter"],
            "confidence": STATE["conf"],
            "text": "".join(STATE["buffer"]),
        }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": load_model() is not None}


@app.post("/clear")
def clear():
    with STATE_LOCK:
        STATE["buffer"].clear()
        STATE["letter"] = "NOTHING"
        STATE["conf"] = 0.0
        SMOOTHER.reset()
    return {"ok": True}


def run(host: str = "127.0.0.1", port: int = 5056) -> None:
    app.run(host=host, port=port, debug=False, threaded=True)
