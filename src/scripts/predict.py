#!/usr/bin/env python
"""CLI prediction wrapper.

Usage
-----
    python scripts/predict.py --text "Congratulations! You won a prize. Click [URL]."
    python scripts/predict.py --file path/to/email.txt
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phishguard.config import load_config
from phishguard.inference.predictor import PhishGuardPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="PhishGuard prediction CLI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Email body to classify.")
    group.add_argument("--file", type=str, help="Path to a plain-text email file.")
    args = parser.parse_args()

    config = load_config()
    ckpt = config["output"]["best_checkpoint_dir"]

    if not Path(ckpt).exists():
        print(f"[predict] Checkpoint not found: {ckpt}\nRun scripts/train.py first.")
        sys.exit(1)

    predictor = PhishGuardPredictor(
        checkpoint_dir=ckpt,
        max_length=config["model"]["max_seq_length"],
    )

    text = args.text if args.text else Path(args.file).read_text(encoding="utf-8")
    result = predictor.predict(text)

    print(f"\nVerdict    : {result['label'].upper()}")
    print(f"Confidence : {result['confidence']:.1%}")
    print("\nTop attention tokens:")
    top = sorted(result["token_scores"], key=lambda x: x["score"], reverse=True)[:10]
    for t in top:
        print(f"  {t['token']:20s}  {t['score']:.4f}")


if __name__ == "__main__":
    main()
