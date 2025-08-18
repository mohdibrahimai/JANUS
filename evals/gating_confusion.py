"""Compute gating confusion metrics.

Retrieve‑When‑Needed Recall and Parametric‑When‑Safe Precision quantify how
well the policy chooses between retrieval and parametric actions.  These
metrics require labels indicating whether retrieval was necessary for each
query (i.e. the parametric model would hallucinate or be stale) and
whether a parametric answer would be safe.
"""
from __future__ import annotations

from typing import List, Tuple


def compute_gating_metrics(predicted: List[str], needed_retrieval: List[bool], safe_parametric: List[bool]) -> Tuple[float, float]:
    """Compute recall and precision for the gating policy.

    Parameters
    ----------
    predicted: List[str]
        Predicted actions ('parametric', 'retrieve', etc.).
    needed_retrieval: List[bool]
        Flags indicating whether retrieval was actually needed for each query.
    safe_parametric: List[bool]
        Flags indicating whether a parametric answer would have been safe.

    Returns
    -------
    (float, float)
        (retrieve_when_needed_recall, parametric_when_safe_precision).
    """
    assert len(predicted) == len(needed_retrieval) == len(safe_parametric)

    tp_retrieve = 0  # predicted retrieve when needed
    fn_retrieve = 0  # failed to retrieve when needed
    tp_parametric = 0  # predicted parametric when safe
    fp_parametric = 0  # predicted parametric when not safe

    for p, need, safe in zip(predicted, needed_retrieval, safe_parametric):
        if need:
            if p == "retrieve":
                tp_retrieve += 1
            else:
                fn_retrieve += 1
        if p == "parametric":
            if safe:
                tp_parametric += 1
            else:
                fp_parametric += 1

    recall = tp_retrieve / (tp_retrieve + fn_retrieve) if (tp_retrieve + fn_retrieve) > 0 else 0.0
    precision = tp_parametric / (tp_parametric + fp_parametric) if (tp_parametric + fp_parametric) > 0 else 0.0
    return recall, precision