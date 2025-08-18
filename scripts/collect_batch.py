"""Collect parametric and retrieve outputs for counterfactuals.

This script iterates over a set of labelled queries and generates both
parametric and retrieval answers.  Annotators can then compare the two and
select which one they prefer (for DPO/ORPO training).  Currently this
script simply copies the query and produces placeholder answers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..policy.dataset import load_jsonl
from ..orchestrator import tools
from ..orchestrator import retriever


def collect(input_jsonl: Path, output_jsonl: Path) -> None:
    examples = load_jsonl(input_jsonl)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for ex in examples:
            query = ex.get("query", "")
            # Generate parametric and retrieval answers
            parametric_ans = tools.answer_parametric(query)
            retrieve_ans = tools.answer_with_rag(query)
            out = {
                "id": ex.get("id"),
                "query": query,
                "parametric_answer": parametric_ans,
                "retrieve_answer": retrieve_ans,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect counterfactual pairs")
    parser.add_argument("--input", type=str, required=True, help="Input labelled JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file with pairs")
    args = parser.parse_args()
    collect(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()