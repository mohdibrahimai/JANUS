"""Dataset utilities for training JANUS gating policies.

This module provides functions to load labelled data from JSONL files and
convert them into feature matrices for model training.  It supports
extracting features from queries using the functions in
`features/query_signals.py` and encoding categorical fields such as risk
and domain.

The dataset class returns pairs `(x, y)` where `x` is a numeric feature
vector and `y` is an integer label encoding the chosen action.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

from ..features.query_signals import compute_signals


# Define a mapping from action names to integer classes.  These are the
# classes predicted by the gating policy.  If you change the list, be sure
# to update downstream code accordingly.
ACTION_TO_ID: Dict[str, int] = {
    "parametric": 0,
    "retrieve": 1,
    "compute": 2,
    "clarify": 3,
    "escalate": 4,
    "abstain": 5,
}
ID_TO_ACTION = {v: k for k, v in ACTION_TO_ID.items()}


def encode_risk(risk: str) -> float:
    return {"low": 0.0, "med": 0.5, "high": 1.0}.get(risk, 0.0)


def encode_domain(domain: str) -> float:
    mapping = {
        "general": 0.0,
        "medical": 1.0,
        "finance": 2.0,
        "science": 3.0,
        "other": 4.0,
    }
    return mapping.get(domain, 0.0)


def load_jsonl(path: Path) -> List[Dict[str, any]]:
    """Load a JSONL file into a list of dicts."""
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                examples.append(obj)
            except json.JSONDecodeError:
                continue
    return examples


def prepare_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare a dataset for training from a JSONL labels file.

    Parameters
    ----------
    path: Path
        Path to the JSONL file containing labelled examples.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple `(X, y)` where `X` is a 2‑D array of shape `(n_samples, n_features)`
        and `y` is a 1‑D array of integer class labels.
    """
    examples = load_jsonl(path)
    features = []
    labels = []

    for ex in examples:
        q = ex.get("query", "")
        signals = compute_signals(q)
        # Feature vector: entity_count, has_time_expression, ambiguity_score,
        # breaking_news_score, freshness_days_max, risk, domain
        feat = [
            signals.entity_count,
            1.0 if signals.has_time_expression else 0.0,
            signals.ambiguity_score,
            signals.breaking_news_score,
            float(ex.get("freshness_days_max", 0)),
            encode_risk(ex.get("risk", "low")),
            encode_domain(ex.get("domain", "general")),
        ]
        features.append(feat)

        action = ex.get("action", "parametric").lower()
        labels.append(ACTION_TO_ID.get(action, 0))

    X = np.array(features, dtype=float)
    y = np.array(labels, dtype=int)
    return X, y