"""Feature extraction for query signals.

This module contains functions to derive low‑level features from user queries.
Features include named entity counts, the presence of time expressions, a
simple ambiguity heuristic and detection of breaking‑news lexemes.  These
signals are used by the gating policy to decide whether to trust parametric
memory or to retrieve fresh documents.

The current implementation uses simple heuristics because heavy NLP
dependencies such as spaCy are not installed by default.  You can plug in
your favourite NLP library by modifying the functions below.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict


# A small set of lexemes that might signal breaking news queries.
BREAKING_NEWS_LEXEMES = {
    "breaking", "news", "update", "latest", "just in", "breaking news",
    "live", "速報", "últimas", "خبر", "अभी अभी",
}


@dataclass
class QuerySignals:
    entity_count: int
    has_time_expression: bool
    ambiguity_score: float
    breaking_news_score: float


def extract_entities(text: str) -> List[str]:
    """Naive named entity extractor.

    This function uses a simple regex to find capitalised words as a
    placeholder for named entities.  In a production system you should
    integrate a proper NER model (e.g. spaCy) here.

    Parameters
    ----------
    text: str
        The query text.

    Returns
    -------
    List[str]
        A list of detected entity tokens.
    """
    return re.findall(r"\b[A-Z][a-z]+\b", text)


def detect_time_expression(text: str) -> bool:
    """Detect whether the query contains time expressions.

    We look for simple patterns such as years (four digits), months or
    words like 'today', 'yesterday', 'tomorrow'.  This is a very crude
    heuristic and can be replaced with a date parser.
    """
    patterns = [r"\b\d{4}\b", r"\b(today|yesterday|tomorrow)\b", r"\b\d{1,2}\/[\d{1,2}]\/(\d{2}|\d{4})\b"]
    for pat in patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False


def compute_ambiguity(text: str) -> float:
    """A toy ambiguity heuristic.

    We define ambiguity as the ratio of stop‑words to content words.  A higher
    ratio suggests a less specific query.  This is only a placeholder; you
    should refine this measure using semantic similarity models.
    """
    # A small set of common English stop words; extend for other languages as needed.
    stop_words = set("the of and a an to in for on with at by from".split())
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not tokens:
        return 0.0
    stop_count = sum(1 for t in tokens if t in stop_words)
    return stop_count / len(tokens)


def breaking_news_score(text: str) -> float:
    """Return a score indicating whether the query sounds like breaking news.

    We compute the fraction of tokens that match known breaking news lexemes.
    """
    tokens = set(re.findall(r"\b\w+\b", text.lower()))
    hits = tokens.intersection({lex.lower() for lex in BREAKING_NEWS_LEXEMES})
    return len(hits) / (len(tokens) or 1)


def compute_signals(text: str) -> QuerySignals:
    """Compute query signal features for a single query.

    Parameters
    ----------
    text: str
        The query text.

    Returns
    -------
    QuerySignals
        A dataclass containing the extracted features.
    """
    entities = extract_entities(text)
    return QuerySignals(
        entity_count=len(entities),
        has_time_expression=detect_time_expression(text),
        ambiguity_score=compute_ambiguity(text),
        breaking_news_score=breaking_news_score(text),
    )