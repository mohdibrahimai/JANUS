"""Run JANUS evaluation on a dataset of answered queries.

This script loads a dataset file (JSONL) where each line contains
evaluation information such as the answer, predicted action, freshness
prediction, requirements, latency, cost, etc.  It then computes
aggregated metrics using `evals/metrics.py` and prints the results as
JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from ..evals.metrics import evaluate


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JANUS evaluation")
    parser.add_argument("--data", type=str, required=True, help="Path to evaluation dataset JSONL")
    args = parser.parse_args()
    dataset = load_dataset(Path(args.data))
    metrics = evaluate(dataset)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()