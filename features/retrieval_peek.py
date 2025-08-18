"""Lightweight retrieval preview for JANUS.

In order to decide whether to retrieve or trust parametric memory, the gating
policy can peek at the results of a cheap retrieval.  This module defines
functions to compute quick features from candidate documents, such as the
distribution of publication timestamps.

These previews are not meant to retrieve full content – they only access
metadata such as timestamps and scores.  A full retrieval happens later if
the policy decides to take the `retrieve` action.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Dict, Tuple


def timestamp_histogram(docs: List[Dict[str, str]]) -> Tuple[int, int, int]:
    """Compute a coarse histogram of document ages.

    Given a list of documents, each represented as a dict with a `timestamp`
    field (ISO 8601 string), this function counts how many docs are newer
    than 7 days, between 7 and 30 days, and older than 30 days.  If no
    timestamps are available, returns zeros.

    Parameters
    ----------
    docs: List[Dict[str, str]]
        A list of document metadata dictionaries.  Each should have a
        `timestamp` key with an ISO formatted date.

    Returns
    -------
    Tuple[int, int, int]
        A tuple with counts for (0–7 days, 7–30 days, >30 days).
    """
    now = datetime.now(timezone.utc)
    bins = [0, 0, 0]
    for doc in docs:
        ts_str = doc.get("timestamp")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str)
        except ValueError:
            continue
        age_days = (now - ts).days
        if age_days <= 7:
            bins[0] += 1
        elif age_days <= 30:
            bins[1] += 1
        else:
            bins[2] += 1
    return tuple(bins)