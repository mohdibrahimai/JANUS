"""Controller for the JANUS gating policy.

This module implements a small inference wrapper around the trained softmax
classifier produced by `policy/train_dpo.py`.  Given a query and optional
metadata such as volatility class, it returns the recommended action and
the freshness target (days) predicted by the freshness head.

To use this controller you must first load a model JSON file containing
`weights`, `bias` and `id_to_action`.  See `policy/train_dpo.py` for
details on the format.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .dataset import ID_TO_ACTION, encode_risk, encode_domain
from ..features.query_signals import compute_signals
from .freshness_head import predict_staleness_days


class JanusController:
    """A simple gating policy controller."""

    def __init__(self, model: Dict[str, Any]):
        # Load weights and bias into numpy arrays
        self.W = np.array(model["weights"], dtype=float)
        self.b = np.array(model["bias"], dtype=float)
        # id_to_action mapping; fallback to global mapping if absent
        self.id_to_action = model.get("id_to_action", ID_TO_ACTION)

    @classmethod
    def from_file(cls, path: Path) -> "JanusController":
        with path.open("r", encoding="utf-8") as f:
            model = json.load(f)
        return cls(model)

    def compute_feature_vector(self, query: str, metadata: Dict[str, Any]) -> np.ndarray:
        """Compute the feature vector for a single query.

        Parameters
        ----------
        query: str
            The user query.
        metadata: dict
            Additional metadata containing at least `freshness_days_max`, `risk` and
            `domain`.

        Returns
        -------
        np.ndarray
            A 1‑D feature vector consistent with training.
        """
        signals = compute_signals(query)
        freshness_days_max = float(metadata.get("freshness_days_max", 0))
        risk_code = encode_risk(metadata.get("risk", "low"))
        domain_code = encode_domain(metadata.get("domain", "general"))
        feats = np.array([
            signals.entity_count,
            1.0 if signals.has_time_expression else 0.0,
            signals.ambiguity_score,
            signals.breaking_news_score,
            freshness_days_max,
            risk_code,
            domain_code,
        ], dtype=float)
        return feats

    def decide(self, query: str, metadata: Dict[str, Any]) -> Tuple[str, float]:
        """Return the recommended action and freshness target.

        Parameters
        ----------
        query: str
            The user query.
        metadata: dict
            Additional fields such as volatility.

        Returns
        -------
        (str, float)
            The predicted action name and the freshness target in days.
        """
        x = self.compute_feature_vector(query, metadata)
        logits = x @ self.W + self.b
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        action_id = int(probs.argmax())
        action = self.id_to_action[str(action_id)] if isinstance(self.id_to_action, dict) else self.id_to_action.get(action_id, "parametric")
        # Use metadata volatility if provided; default to 'fast'
        volatility = metadata.get("volatility", "fast")
        freshness_target = predict_staleness_days(query, volatility)
        return action, freshness_target