from __future__ import annotations

import argparse

from asl.collect import collect_webcam
from asl.kaggle import ingest_kaggle
from asl.train import train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="asl", description="ASL finger-spelling recognizer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    collect = sub.add_parser("collect", help="Record webcam samples for one label")
    collect.add_argument("--label", required=True)
    collect.add_argument("-n", type=int, default=80)
    collect.add_argument("--camera", type=int, default=0)

    kaggle = sub.add_parser("ingest-kaggle", help="Convert Kaggle ASL images to landmarks CSV")
    kaggle.add_argument("root", help="Path to asl_alphabet_train (folders A, B, ...)")
    kaggle.add_argument("--limit", type=int, default=400)

    sub.add_parser("train", help="Train RF/SVM/MLP and save models/best.joblib")

    args = parser.parse_args(argv)
    if args.cmd == "collect":
        collect_webcam(args.label.upper(), n=args.n, camera=args.camera)
        return 0
    if args.cmd == "ingest-kaggle":
        ingest_kaggle(args.root, limit_per_label=args.limit)
        return 0
    if args.cmd == "train":
        train()
        return 0
    return 1
