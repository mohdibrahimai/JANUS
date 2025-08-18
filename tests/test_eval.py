"""Tests for evaluation metrics functions."""
from __future__ import annotations

from janus.evals.gating_confusion import compute_gating_metrics
from janus.evals.freshness_sla import compute_freshness_sla
from janus.evals.multilingual import cross_lingual_gap


def test_gating_metrics() -> None:
    predicted = ["retrieve", "parametric", "parametric", "retrieve"]
    needed = [True, False, True, True]
    safe = [False, True, False, True]
    recall, precision = compute_gating_metrics(predicted, needed, safe)
    # In this synthetic example: retrieval is needed at positions 0, 2 and 3; the policy retrieves at 0 and 3, hence recall = 2/3.
    assert abs(recall - 2/3) < 1e-6
    # Parametric is predicted at positions 1 and 2, but only position 1 is safe.  So precision = 1/2 = 0.5.
    assert abs(precision - 0.5) < 1e-6


def test_freshness_sla() -> None:
    preds = [5.0, 2.0, 10.0]
    reqs = [3, 2, 15]
    # hits at indices 0 and 2: 5>=3 and 10>=15? false => 1 out of 3
    hit_rate = compute_freshness_sla(preds, reqs)
    assert abs(hit_rate - (2/3)) < 1e-6


def test_cross_lingual_gap() -> None:
    metrics = {"en": 0.8, "hi": 0.6, "ur": 0.85}
    gaps = cross_lingual_gap(metrics, reference_lang="en")
    assert abs(gaps["hi"] - (-0.2)) < 1e-6
    assert abs(gaps["ur"] - 0.05) < 1e-6