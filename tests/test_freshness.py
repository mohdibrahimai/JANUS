"""Tests for freshness head predictions."""
from __future__ import annotations

from janus.policy.freshness_head import predict_staleness_days


def test_freshness_monotonicity() -> None:
    query = "Breaking news about AI"
    # Breaking queries should have shorter staleness horizon
    breaking = predict_staleness_days(query, "breaking")
    fast = predict_staleness_days(query, "fast")
    slow = predict_staleness_days(query, "slow")
    timeless = predict_staleness_days(query, "timeless")
    assert breaking < fast < slow < timeless