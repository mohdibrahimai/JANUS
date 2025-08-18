"""Multilingual evaluation utilities for JANUS.

This module contains helper functions to compute cross‑lingual gaps in
performance metrics.  A cross‑lingual gap is defined as the difference
between the metric values for English and another language (Hindi, Urdu or
Spanish).  The functions expect dictionaries keyed by language codes.
"""
from __future__ import annotations

from typing import Dict, List


def cross_lingual_gap(metric_by_lang: Dict[str, float], reference_lang: str = "en") -> Dict[str, float]:
    """Compute cross‑lingual gaps relative to the reference language.

    Parameters
    ----------
    metric_by_lang: Dict[str, float]
        A mapping from language code to metric value.
    reference_lang: str
        The language to use as the baseline (default "en").

    Returns
    -------
    Dict[str, float]
        A dictionary mapping each non‑reference language to the gap value
        (metric - reference_metric).
    """
    gaps = {}
    ref_val = metric_by_lang.get(reference_lang)
    if ref_val is None:
        return gaps
    for lang, val in metric_by_lang.items():
        if lang == reference_lang:
            continue
        gaps[lang] = val - ref_val
    return gaps