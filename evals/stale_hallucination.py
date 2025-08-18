"""Evaluate stale hallucination rate.

This script provides a function to compute the fraction of queries for which
the parametric answer is stale or hallucinated.  In this prototype
implementation we use dummy evaluation and return zero.  In your
experiments you should implement logic to detect when an answer contains
claims contradicted by or missing from retrieved evidence.
"""
from __future__ import annotations

from typing import List


def compute_stale_hallucination_rate(answers: List[str], references: List[str]) -> float:
    """Return the stale hallucination rate.

    Parameters
    ----------
    answers: List[str]
        A list of answers produced by the parametric model.
    references: List[str]
        Ground truth answers or reference documents.

    Returns
    -------
    float
        The fraction of answers judged stale or hallucinated.  Currently
        returns 0.0 as a placeholder.
    """
    # TODO: implement evaluation using NLI or fuzzy matching
    return 0.0