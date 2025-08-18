"""Train a simple gating policy using multinomial logistic regression.

This script implements a bare‑bones training loop for a multi‑class
logistic regression model (softmax classifier) using only `numpy`.  It is
intended as a starting point and demonstration of how to train the JANUS
controller on labelled data.  For better performance you should use a
proper ML library (e.g. PyTorch, JAX) and explore advanced fine‑tuning
techniques such as DPO/ORPO and cost‑aware bandits.

The script expects a JSONL file of labelled examples (see
`policy/dataset.py` for the schema) and outputs a model definition file
containing learned weights and bias.  Use `policy/controller.py` to load
the model for inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np

from .dataset import prepare_dataset, ID_TO_ACTION


def softmax(z: np.ndarray) -> np.ndarray:
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def cross_entropy_loss(probs: np.ndarray, y: np.ndarray) -> float:
    n = y.shape[0]
    # Clip to avoid log(0)
    probs = np.clip(probs, 1e-12, 1.0)
    log_likelihood = -np.log(probs[np.arange(n), y])
    return float(np.mean(log_likelihood))


def train_softmax(X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Train a multinomial logistic regression model.

    Parameters
    ----------
    X: np.ndarray
        Feature matrix of shape `(n_samples, n_features)`.
    y: np.ndarray
        Integer labels of shape `(n_samples,)`.
    lr: float
        Learning rate.
    epochs: int
        Number of training epochs.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The weight matrix and bias vector.
    """
    n_samples, n_features = X.shape
    n_classes = y.max() + 1
    W = np.zeros((n_features, n_classes))
    b = np.zeros((n_classes,))

    for epoch in range(epochs):
        # Forward
        logits = X @ W + b
        probs = softmax(logits)
        loss = cross_entropy_loss(probs, y)

        # Gradient
        y_one_hot = np.zeros_like(probs)
        y_one_hot[np.arange(n_samples), y] = 1.0
        grad_logits = (probs - y_one_hot) / n_samples
        grad_W = X.T @ grad_logits
        grad_b = grad_logits.sum(axis=0)

        # Update
        W -= lr * grad_W
        b -= lr * grad_b

        # Optionally print training progress
        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch+1}/{epochs}, loss={loss:.4f}")

    return W, b


def save_model(path: Path, W: np.ndarray, b: np.ndarray) -> None:
    """Save the trained model to a JSON file."""
    model = {
        "weights": W.tolist(),
        "bias": b.tolist(),
        "id_to_action": ID_TO_ACTION,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(model, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train JANUS gating policy")
    parser.add_argument("--data", type=str, required=True, help="Path to JSONL labelled data")
    parser.add_argument("--out", type=str, required=True, help="Output model file (JSON)")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    args = parser.parse_args()

    X, y = prepare_dataset(Path(args.data))
    W, b = train_softmax(X, y, lr=args.lr, epochs=args.epochs)
    save_model(Path(args.out), W, b)
    print(f"Model saved to {args.out}")


if __name__ == "__main__":
    main()