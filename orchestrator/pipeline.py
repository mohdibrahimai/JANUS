"""Orchestrator API for JANUS.

This module defines a FastAPI application that exposes an `/answer` endpoint.
It uses the JanusController to determine how to answer incoming queries
based on user‑provided metadata (volatility, risk, domain and freshness
requirement).  Depending on the action, it will either answer directly
from the parametric model, perform retrieval, run a computation, ask for
clarification or refuse.

The API returns a JSON object containing the chosen action, the answer
text, the predicted freshness target and any additional metadata.

This is a prototype implementation; integrate your own language model and
retrieval system to obtain meaningful answers.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..policy.controller import JanusController
from ..policy.freshness_head import predict_staleness_days
from . import tools
from . import retriever


class QueryRequest(BaseModel):
    query: str = Field(..., description="The user query")
    freshness_days_max: int = Field(7, description="Maximum acceptable staleness in days")
    volatility: str = Field("fast", description="Volatility class: timeless/slow/fast/breaking")
    risk: str = Field("low", description="Risk level: low/med/high")
    domain: str = Field("general", description="Domain: general/medical/finance/science/other")


class AnswerResponse(BaseModel):
    action: str
    answer: str
    freshness_target: float
    staleness_met: bool
    metadata: Dict[str, Any] = {}


def load_controller() -> JanusController:
    """Load the gating model from disk.

    The environment variable `JANUS_MODEL_PATH` can be set to point at a
    model file; otherwise the default `policy/model.json` is used.
    """
    model_path = os.environ.get("JANUS_MODEL_PATH", "policy/model.json")
    path = Path(model_path)
    if not path.exists():
        raise RuntimeError(f"Model file not found at {path}.  Train a model first.")
    return JanusController.from_file(path)


app = FastAPI(title="JANUS Orchestrator")

controller = None  # Will be initialised at startup


@app.on_event("startup")
def on_startup() -> None:
    global controller
    controller = load_controller()


@app.post("/answer", response_model=AnswerResponse)
def answer(request: QueryRequest) -> AnswerResponse:
    if controller is None:
        raise HTTPException(status_code=500, detail="Controller not initialised")
    query = request.query.strip()
    metadata = {
        "freshness_days_max": request.freshness_days_max,
        "risk": request.risk,
        "domain": request.domain,
        "volatility": request.volatility,
    }
    action, freshness_target = controller.decide(query, metadata)
    # Determine if user’s required freshness is satisfied
    staleness_met = request.freshness_days_max <= freshness_target
    # Route to corresponding tool
    if action == "parametric":
        ans = tools.answer_parametric(query)
    elif action == "retrieve":
        ans = tools.answer_with_rag(query)
    elif action == "compute":
        ans = tools.answer_with_tools(query)
    elif action == "clarify":
        ans = tools.ask_one_question(query)
    elif action in ("escalate", "abstain"):
        ans = tools.safe_refusal(query)
    else:
        ans = "[Unknown action]"
    return AnswerResponse(
        action=action,
        answer=ans,
        freshness_target=freshness_target,
        staleness_met=staleness_met,
        metadata={"quick_peek_hist": retriever.quick_peek(query)},
    )