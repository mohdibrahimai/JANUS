"""Freshness decay head for JANUS.

The freshness head predicts how quickly the model's parametric knowledge
about a given query will become stale.  It can be trained from human
annotations specifying acceptable staleness windows.  Here we implement a
simple heuristic: high breaking news score and high volatility imply short
freshness; timeless queries imply long freshness.
"""
from __future__ import annotations

from typing import Dict

from ..features.query_signals import compute_signals


def predict_staleness_days(query: str, volatility: str) -> float:
    """Predict a staleness time horizon for a query.

    Parameters
    ----------
    query: str
        The user query.
    volatility: str
        One of {"timeless", "slow", "fast", "breaking"}.

    Returns
    -------
    float
        The predicted number of days after which the parametric answer may be
        stale.  This is a heuristic and should be replaced by a trained
        regression model.
    """
    signals = compute_signals(query)
    # Base horizon based on volatility class
    base = {
        "timeless": 3650.0,  # 10 years
        "slow": 180.0,
        "fast": 30.0,
        "breaking": 1.0,
    }.get(volatility, 30.0)
    # Adjust based on breaking news score; more breaking -> shorter horizon
    adjustment = (1.0 - signals.breaking_news_score) * base
    # Ensure at least one day
    return max(1.0, adjustment)