"""Streamlit data collection app for JANUS.

This module defines a simple Streamlit interface for collecting human labels
for the JANUS gating policy.  Annotators are asked to assign an action
(`parametric`, `retrieve`, `compute`, `clarify`, `escalate` or `abstain`),
specify a maximum acceptable staleness in days, classify the volatility of
the underlying knowledge, assess risk and domain, and optionally provide
minimum evidence and citation quality.  Multilingual variants of the same
query can be labelled side‑by‑side.

The collected labels are written to JSONL files in the `data/` directory.

Note: Streamlit is not installed by default in this environment.  To run this
app locally you must install `streamlit` (e.g. via `pip install streamlit`).
"""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

import pandas as pd

# Only import streamlit inside functions so that unit tests can run without it.

@dataclass
class LabelEntry:
    """Represents a single labelled query example."""
    id: str
    lang: str
    query: str
    draft_answer: str
    action: str
    freshness_days_max: int
    volatility: str
    risk: str
    domain: str
    min_evidence: Optional[List[str]] = None
    citation_quality: Optional[int] = None
    gold_check: bool = False
    rater_id: Optional[str] = None
    notes: Optional[str] = None


def save_labels(labels: List[LabelEntry], file_path: Path) -> None:
    """Append labelled entries to a JSONL file.

    Parameters
    ----------
    labels: List[LabelEntry]
        The list of labels to append.
    file_path: Path
        The JSONL file to append to.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as f:
        for entry in labels:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")


def run_app() -> None:
    """Run the Streamlit annotation UI.

    This function is the entrypoint used when launching the app via
    `streamlit run app/data_studio.py`.  It builds a small user interface for
    entering queries, selecting labels and writing them to disk.
    """
    import streamlit as st  # type: ignore

    st.title("JANUS Data Labeling Studio")

    st.markdown("""
    Provide human labels for training the JANUS gating policy.  For each query
    (and optional draft answer), select the appropriate action and metadata.
    You can add multiple language variants for the same underlying question.
    """)

    # Input fields for metadata
    id_value = st.text_input("Unique ID", value="")
    query = st.text_area("Query", value="")
    draft_answer = st.text_area("Draft Answer", value="")
    lang = st.selectbox("Language", ["en", "hi", "ur", "es"])
    action = st.selectbox(
        "Action", ["parametric", "retrieve", "compute", "clarify", "escalate", "abstain"]
    )
    freshness_days_max = st.number_input(
        "Max Acceptable Staleness (days)", min_value=0, max_value=365, value=7
    )
    volatility = st.selectbox(
        "Volatility", ["timeless", "slow", "fast", "breaking"]
    )
    risk = st.selectbox("Risk", ["low", "med", "high"])
    domain = st.selectbox("Domain", ["general", "medical", "finance", "science", "other"])
    min_evidence = st.text_area(
        "Minimal Evidence Set (optional)", value="", help="Comma separated URLs or doc IDs"
    )
    citation_quality = st.slider(
        "Citation Quality (1–5)", min_value=1, max_value=5, value=3, step=1
    )
    gold_check = st.checkbox("Gold Check", value=False)
    rater_id = st.text_input("Rater ID", value="")
    notes = st.text_area("Notes", value="")

    if st.button("Add Label"):
        labels: List[LabelEntry] = []
        entry = LabelEntry(
            id=id_value or st.session_state.get("temp_id", ""),
            lang=lang,
            query=query,
            draft_answer=draft_answer,
            action=action,
            freshness_days_max=int(freshness_days_max),
            volatility=volatility,
            risk=risk,
            domain=domain,
            min_evidence=[x.strip() for x in min_evidence.split(",") if x.strip()] or None,
            citation_quality=citation_quality,
            gold_check=gold_check,
            rater_id=rater_id or None,
            notes=notes or None,
        )
        labels.append(entry)
        save_path = Path("data/curated/labels.jsonl")
        save_labels(labels, save_path)
        st.success(f"Saved label {entry.id} to {save_path}")


if __name__ == "__main__":
    run_app()