"""Hybrid retriever for JANUS.

This module wraps retrieval functionality.  The `quick_peek` function
performs a lightweight retrieval using only metadata to produce features
for the gating policy.  The `retrieve_full` function performs a full
retrieval and returns passages for answer generation.

The current implementation does not interact with any external search
engine.  You should connect this to a BM25 + embedding retriever (e.g.
Elasticsearch + FAISS) and a reranker.
"""
from __future__ import annotations

from typing import List, Dict

from ..features.retrieval_peek import timestamp_histogram


def quick_peek(query: str) -> List[int]:
    """Return a histogram of document ages from a dummy retrieval.

    In a real system this would query an index and fetch just the timestamps.
    Here we return a placeholder histogram.
    """
    # Placeholder: pretend we found documents evenly distributed across bins
    dummy_docs = [
        {"timestamp": "2025-08-15T00:00:00+00:00"},
        {"timestamp": "2025-08-01T00:00:00+00:00"},
        {"timestamp": "2024-01-01T00:00:00+00:00"},
    ]
    return list(timestamp_histogram(dummy_docs))


def retrieve_full(query: str, top_k: int = 3) -> List[Dict[str, str]]:
    """Perform a full retrieval and return documents.

    This stub returns placeholder documents.  Integrate your retriever here.
    """
    return [
        {"content": "[Document 1 placeholder for query: " + query + "]", "source": "doc1"},
        {"content": "[Document 2 placeholder for query: " + query + "]", "source": "doc2"},
    ][:top_k]