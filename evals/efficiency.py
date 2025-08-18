"""Efficiency evaluation utilities for JANUS.

This module contains functions to compute latency and token cost metrics,
such as median (p50) and 95th percentile (p95) of latency and cost per
supported claim.
"""
from __future__ import annotations

import numpy as np
from typing import List, Dict


def latency_percentiles(latencies: List[float]) -> Dict[str, float]:
    """Compute median and 95th percentile latency.

    Parameters
    ----------
    latencies: List[float]
        A list of observed latencies (e.g. in seconds).

    Returns
    -------
    Dict[str, float]
        A dict with keys 'p50' and 'p95'.
    """
    if not latencies:
        return {"p50": 0.0, "p95": 0.0}
    arr = np.array(latencies)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def cost_per_claim(costs: List[float], supports: List[bool]) -> float:
    """Compute the cost per supported claim.

    Parameters
    ----------
    costs: List[float]
        Token or latency cost for each answer.
    supports: List[bool]
        Flags indicating whether the corresponding answer is supported.

    Returns
    -------
    float
        Total cost divided by number of supported answers.  Returns 0 if
        no supported answers.
    """
    total_cost = sum(costs)
    supported = sum(1 for s in supports if s)
    return total_cost / supported if supported > 0 else 0.0