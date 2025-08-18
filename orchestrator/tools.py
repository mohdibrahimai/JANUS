"""General purpose tools used by the JANUS orchestrator.

This module defines helper functions for answering queries using different
mechanisms such as parametric memory (LLM), retrieval with RAG, simple
computations (calculator / code execution) and clarifications.  The
implementations here are minimal and primarily for demonstration.  You
should replace them with calls to your language model, retriever and
toolchain.
"""
from __future__ import annotations

import ast
import operator
from typing import Any, Tuple


def answer_parametric(query: str) -> str:
    """Return a parametric (LLM) answer for the query.

    Currently returns a placeholder response.  Integrate your LLM here.
    """
    return "[Parametric answer placeholder]"


def answer_with_rag(query: str) -> str:
    """Return an answer by performing retrieval and citing sources.

    This is a stub function.  You should integrate a proper retriever and
    summariser.  For now, it returns a placeholder string.
    """
    return "[Retrieved answer placeholder]"


def safe_eval(expression: str) -> Any:
    """Safely evaluate a simple arithmetic expression.

    Only basic arithmetic operations are allowed.  This function parses
    the expression into an AST and evaluates it using a restricted set of
    operators.  Any unsupported node will raise an exception.
    """
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in allowed_operators:
                raise ValueError(f"Unsupported operator: {op_type}")
            return allowed_operators[op_type](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in allowed_operators:
                raise ValueError(f"Unsupported unary operator: {op_type}")
            return allowed_operators[op_type](_eval(node.operand))
        else:
            raise ValueError(f"Unsupported expression: {expression}")

    parsed = ast.parse(expression, mode="eval")
    return _eval(parsed.body)


def answer_with_tools(query: str) -> str:
    """Attempt to answer a query using calculator and code tools.

    If the query contains a simple arithmetic expression, evaluate it.  For
    anything else, return a placeholder.  In a full implementation you
    could integrate a safe code execution sandbox here.
    """
    # Attempt to extract a math expression
    expr = query.strip().rstrip("?")
    try:
        result = safe_eval(expr)
        return str(result)
    except Exception:
        return "[Computation not supported]"


def ask_one_question(query: str) -> str:
    """Return a clarifying question for the user."""
    return "Could you please clarify your question?"


def safe_refusal(query: str) -> str:
    """Return a safe refusal when the system cannot answer."""
    return "I’m sorry, but I’m unable to help with that request."