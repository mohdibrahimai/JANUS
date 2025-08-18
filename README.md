# JANUS: Human-Guided Parametric-vs‑Retrieval Gating

JANUS is a proof‑of‑concept system for controlling how a language model answers user queries.  It tries to decide, per query, whether it should:

- **answer from the model’s parametric memory** (fast but potentially stale),
- **retrieve and cite supporting documents**,
- **run computations**,
- **clarify the question**, or
- **escalate / abstain** when it cannot safely answer.

The goal is to maximise truthfulness and citation quality while keeping latency and token costs within budget.  JANUS is designed to be multilingual (English, Hindi, Urdu and Spanish) and to train the decision policy from human‑labelled data.

This repository contains a runnable prototype with a modular architecture.  Each component lives in its own directory:

* `app/` – user‑facing applications (Streamlit data labelling app and an evaluation dashboard).
* `data/` – datasets and data cards.
* `features/` – feature extraction modules used by the gating policy.
* `policy/` – data loading and training code for the gating controller.
* `orchestrator/` – code that ties the policy, retrieval and tools together to answer queries.
* `truth/` – adapters for truthfulness and citation checking metrics.
* `evals/` – evaluation scripts to measure stale hallucination rate, citation precision, freshness SLA compliance and more.
* `ci/` – continuous integration configuration (gates and workflow).
* `scripts/` – helper scripts for data collection, training and evaluation.
* `tests/` – unit tests.

## Quickstart

1. **Install dependencies**.  JANUS is built with Python 3.  You need a few packages such as `numpy`, `pandas`, and `fastapi`.  To train or run the model you will also need `torch` and `transformers`.  Because this repository contains minimal stubs, you may install additional packages as needed:

   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the data studio**.  The `app/data_studio.py` script exposes a Streamlit app for collecting labels.  Run:

   ```bash
   streamlit run app/data_studio.py
   ```

3. **Train the gating policy**.  After collecting some labelled queries, create a dataset using `policy/dataset.py` and train a policy:

   ```bash
   python scripts/train_policy.py --data=data/curated/labels.jsonl --out=policy/model
   ```

4. **Run the orchestrator API**.  The orchestrator is a FastAPI service that exposes a `/answer` endpoint.  Start it with:

   ```bash
   uvicorn orchestrator.pipeline:app --reload
   ```

5. **Evaluate**.  Use the evaluation scripts in `evals/` to measure performance of the gating policy against baseline strategies and generate reports.

This project is under development and not production ready.  It provides a framework to explore research ideas around freshness‑aware retrieval gating.  Many components contain placeholders that need to be fleshed out for a full deployment.