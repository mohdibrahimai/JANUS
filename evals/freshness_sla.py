"""Evaluate Freshness SLA hit‑rate.

This module provides a function to compute the proportion of queries for
which the predicted staleness target meets or exceeds the required
freshness specified by the user.
"""
from __future__ import annotations

from typing import List, Tuple


def compute_freshness_sla(predictions: List[float], requirements: List[int]) -> float:
    """Compute the fraction of cases where staleness target >= requirement.

    Parameters
    ----------
    predictions: List[float]
        Predicted staleness horizons in days.
    requirements: List[int]
        User‑specified maximum acceptable staleness in days.

    Returns
    -------
    float
        The hit rate: fraction of predictions satisfying the requirement.
    """
    assert len(predictions) == len(requirements)
    hits = sum(1 for p, r in zip(predictions, requirements) if p >= r)
    return hits / len(predictions) if predictions else 0.0