"""CLI wrapper around policy training.

This script simply calls `policy/train_dpo.py` with appropriate arguments.
It is provided for convenience.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..policy import train_dpo


def main() -> None:
    parser = argparse.ArgumentParser(description="Train JANUS gating model")
    parser.add_argument("--data", type=str, required=True, help="JSONL labels file")
    parser.add_argument("--out", type=str, default="policy/model.json", help="Output model path")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs")
    args = parser.parse_args()
    X, y = train_dpo.prepare_dataset(Path(args.data))
    W, b = train_dpo.train_softmax(X, y, lr=args.lr, epochs=args.epochs)
    train_dpo.save_model(Path(args.out), W, b)
    print(f"Model saved to {args.out}")


if __name__ == "__main__":
    main()