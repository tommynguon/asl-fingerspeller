from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="asl",
        description="ASL finger-spelling recognizer (webcam → letters).",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="help",
        help="collect | train | demo (implemented in later commits)",
    )
    args = parser.parse_args(argv)
    print(f"asl-fingerspeller scaffold. next: {args.command}")
    print("Install Python 3.12, then: py -3.12 -m venv .venv")
    return 0
