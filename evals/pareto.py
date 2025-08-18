"""Compute Pareto frontiers for JANUS evaluation.

This module provides a function to compute the Pareto frontier of two
metrics, such as truthfulness and latency.  A point is on the Pareto
frontier if no other point dominates it in all dimensions.
"""
from __future__ import annotations

from typing import List, Tuple


def pareto_front(points: List[Tuple[float, float]], maximize: bool = True) -> List[Tuple[float, float]]:
    """Compute the Pareto frontier of a set of 2D points.

    Parameters
    ----------
    points: List[Tuple[float, float]]
        A list of (x, y) tuples representing two metrics.  If maximize is
        True, larger values are considered better; otherwise smaller values
        are better.
    maximize: bool
        Whether to consider larger values better for both dimensions.

    Returns
    -------
    List[Tuple[float, float]]
        The subset of points on the Pareto frontier.
    """
    if not points:
        return []
    # Sort points by x (descending if maximize else ascending)
    sorted_pts = sorted(points, key=lambda p: p[0], reverse=maximize)
    frontier = []
    best_y = None
    for x, y in sorted_pts:
        if best_y is None:
            frontier.append((x, y))
            best_y = y
        else:
            if maximize:
                if y >= best_y:
                    frontier.append((x, y))
                    best_y = y
            else:
                if y <= best_y:
                    frontier.append((x, y))
                    best_y = y
    return frontier