"""Evaluate truthfulness and citation precision for JANUS.

This module provides functions to compute truth and citation metrics on
a set of answered queries.  It uses the `truth/truthlens_adapter` module
to score individual answers.  The current implementation simply
aggregates dummy scores.
"""
from __future__ import annotations

from typing import List, Tuple, Dict

from ..truth.truthlens_adapter import evaluate_answer


def evaluate_answers(answers: List[str], sources_list: List[Dict[str, str]]) -> Dict[str, float]:
    """Evaluate multiple answers for truthfulness and citation precision.

    Parameters
    ----------
    answers: List[str]
        The generated answers.
    sources_list: List[Dict[str, str]]
        A list of dicts mapping source IDs to document content used in the
        answers.

    Returns
    -------
    Dict[str, float]
        Averaged metrics across all answers.
    """
    metrics = {"support": 0.0, "contradict": 0.0, "unverifiable": 0.0, "citation_precision": 0.0}
    n = len(answers)
    if n == 0:
        return metrics
    for ans, sources in zip(answers, sources_list):
        scores = evaluate_answer(ans, sources)
        for k in metrics:
            metrics[k] += scores.get(k, 0.0)
    for k in metrics:
        metrics[k] /= n
    return metrics