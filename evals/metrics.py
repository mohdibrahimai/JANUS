"""Aggregate evaluation metrics for JANUS.

This module exposes a top‑level function `evaluate` that runs a set of
evaluation functions across a dataset of answers and associated metadata.
Because this repository provides only stubs, the aggregate metrics are
illustrative and return zeros or dummy values.  Replace the stub calls
with actual computations when integrating with your own models and
retrievers.
"""
from __future__ import annotations

from typing import List, Dict, Any

from .stale_hallucination import compute_stale_hallucination_rate
from .truth_citation import evaluate_answers
from .gating_confusion import compute_gating_metrics
from .freshness_sla import compute_freshness_sla
from .multilingual import cross_lingual_gap
from .efficiency import latency_percentiles, cost_per_claim
from .pareto import pareto_front


def evaluate(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute a collection of metrics on a dataset of query/answer pairs.

    Parameters
    ----------
    dataset: List[Dict[str, Any]]
        Each element should contain the fields: 'answer', 'sources',
        'predicted_action', 'needed_retrieval', 'safe_parametric',
        'freshness_prediction', 'freshness_requirement', 'latency', 'cost',
        'language'.

    Returns
    -------
    Dict[str, Any]
        A dict of aggregated metrics.
    """
    answers = [ex["answer"] for ex in dataset]
    sources_list = [ex.get("sources", {}) for ex in dataset]
    truth_metrics = evaluate_answers(answers, sources_list)

    predicted_actions = [ex.get("predicted_action", "parametric") for ex in dataset]
    needed_retrieval = [ex.get("needed_retrieval", False) for ex in dataset]
    safe_parametric = [ex.get("safe_parametric", True) for ex in dataset]
    recall, precision = compute_gating_metrics(predicted_actions, needed_retrieval, safe_parametric)

    freshness_preds = [ex.get("freshness_prediction", 0.0) for ex in dataset]
    freshness_reqs = [ex.get("freshness_requirement", 0) for ex in dataset]
    sla_hit_rate = compute_freshness_sla(freshness_preds, freshness_reqs)

    # Multilingual gaps: compute average support per language
    lang_scores: Dict[str, float] = {}
    lang_counts: Dict[str, int] = {}
    for ex, answer in zip(dataset, answers):
        lang = ex.get("language", "en")
        lang_scores[lang] = lang_scores.get(lang, 0.0) + truth_metrics.get("support", 0.0)
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    avg_lang_scores = {lang: score / lang_counts[lang] for lang, score in lang_scores.items()}
    gaps = cross_lingual_gap(avg_lang_scores, reference_lang="en")

    latencies = [ex.get("latency", 0.0) for ex in dataset]
    costs = [ex.get("cost", 0.0) for ex in dataset]
    supports = [ex.get("supported", True) for ex in dataset]
    latency_stats = latency_percentiles(latencies)
    cost_per = cost_per_claim(costs, supports)

    # Pareto frontier: we use support score and latency as example; assume higher support and lower latency is better
    points = [(truth_metrics.get("support", 0.0), -lat) for lat in latencies]
    pareto = pareto_front(points, maximize=True)

    return {
        "truth": truth_metrics,
        "gating_recall": recall,
        "gating_precision": precision,
        "freshness_sla_hit_rate": sla_hit_rate,
        "cross_lingual_gap": gaps,
        "latency": latency_stats,
        "cost_per_supported": cost_per,
        "pareto_frontier": pareto,
    }