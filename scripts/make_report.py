"""Generate a Freshness & Truth report for JANUS.

This script reads metrics JSON (from `scripts/run_eval.py`) and renders a
human‑readable report as an HTML file.  It uses basic string formatting
instead of a templating engine to avoid dependencies.  You may replace
this with Jinja2 or Markdown rendering for richer reports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def make_html_report(metrics: dict, output_path: Path) -> None:
    """Write a simple HTML report summarising JANUS metrics."""
    html = ["<html><head><title>JANUS Freshness & Truth Report</title></head><body>"]
    html.append("<h1>JANUS Freshness & Truth Report</h1>")
    html.append("<h2>Summary</h2>")
    html.append("<pre>")
    html.append(json.dumps(metrics, indent=2))
    html.append("</pre>")
    html.append("</body></html>")
    output_path.write_text("\n".join(html), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a JANUS report")
    parser.add_argument("--metrics", type=str, required=True, help="Metrics JSON file from evaluation")
    parser.add_argument("--out", type=str, required=True, help="Output HTML report path")
    args = parser.parse_args()
    metrics_path = Path(args.metrics)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    make_html_report(metrics, Path(args.out))
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()