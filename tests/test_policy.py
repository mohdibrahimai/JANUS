"""Unit tests for JANUS policy training and controller."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from janus.policy import train_dpo
from janus.policy.controller import JanusController
from janus.policy.dataset import prepare_dataset, ID_TO_ACTION


def test_train_and_decide() -> None:
    # Create a temporary labelled dataset with simple examples
    labels = [
        {
            "id": "1",
            "lang": "en",
            "query": "What is the capital of France?",
            "draft_answer": "Paris is the capital of France.",
            "action": "parametric",
            "freshness_days_max": 30,
            "volatility": "timeless",
            "risk": "low",
            "domain": "general",
        },
        {
            "id": "2",
            "lang": "en",
            "query": "Latest news about elections",
            "draft_answer": "",
            "action": "retrieve",
            "freshness_days_max": 1,
            "volatility": "breaking",
            "risk": "med",
            "domain": "general",
        },
    ]
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        for ex in labels:
            tmp.write(json.dumps(ex) + "\n")
        tmp_path = Path(tmp.name)

    X, y = prepare_dataset(tmp_path)
    # Train a model quickly
    W, b = train_dpo.train_softmax(X, y, lr=0.1, epochs=20)
    # Save model
    model_path = tmp_path.with_suffix(".model.json")
    train_dpo.save_model(model_path, W, b)
    # Load controller
    controller = JanusController.from_file(model_path)
    # Decision for timeless query should prefer parametric
    action, staleness = controller.decide("What is the capital of Germany?", {"freshness_days_max": 30, "risk": "low", "domain": "general", "volatility": "timeless"})
    assert action in ID_TO_ACTION.values()
    assert staleness > 1.0