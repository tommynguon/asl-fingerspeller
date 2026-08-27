from __future__ import annotations

from asl.extract import _hands, extract_from_bgr
from asl.infer import Smoother, apply_letter, load_model, predict_label


def run_demo(camera: int = 0) -> int:
    import cv2
    model = load_model()
    if model is None:
        raise SystemExit("No model yet. Collect data, then: python -m asl train")
    smoother = Smoother()
    buffer: list[str] = []
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit("could not open webcam")
    hands = _hands(static=False)
    print("q quit  |  c clear word")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            feats = extract_from_bgr(frame, hands=hands)
            letter, conf = "NOTHING", 0.0
            if feats is not None:
                letter, conf = predict_label(model, feats)
                token = smoother.push(letter)
                if token:
                    apply_letter(buffer, token)
            word = "".join(buffer)
            vis = frame.copy()
            cv2.putText(vis, f"{letter} {conf:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
            cv2.putText(vis, word[-40:], (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imshow("asl-fingerspeller", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                buffer.clear()
                smoother.last_emitted = None
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0
