"""Uncertainty estimation utilities for JANUS.

The gating policy benefits from signals about the language model's confidence
in its own answer.  This module provides lightweight functions to compute
proxy measures of uncertainty without requiring access to the full model
internals.

Currently these functions are stubs and return simple heuristics.  To make
them useful, integrate them with your language model inference pipeline
and compute quantities such as entropy over next‑token distributions or
variation between sampled answers (self‑consistency).
"""
from __future__ import annotations

import math
from typing import List


def entropy_from_probs(probs: List[float]) -> float:
    """Compute the Shannon entropy of a probability distribution.

    Parameters
    ----------
    probs: List[float]
        A list of probabilities that sum to one.

    Returns
    -------
    float
        The entropy in nats.
    """
    return -sum(p * math.log(p + 1e-12) for p in probs)


def self_consistency_variance(samples: List[str]) -> float:
    """Compute a simple variance metric across multiple sampled answers.

    We measure how often the most common answer differs from the rest.  In
    practice you might use BLEU or semantic similarity; here we count the
    frequency of the top answer and derive a variance score in [0,1].
    """
    if not samples:
        return 0.0
    from collections import Counter

    counts = Counter(samples)
    most_common, freq = counts.most_common(1)[0]
    return 1.0 - (freq / len(samples))