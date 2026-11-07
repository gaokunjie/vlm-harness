"""Aggregate statistics for the DOE report."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

from scipy.stats import binomtest

from .ledger import TrialRecord


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def mcnemar(b: int, c: int) -> dict[str, float]:
    """Exact two-sided McNemar test on the discordant pairs."""
    n = b + c
    if n == 0:
        return {"b": float(b), "c": float(c), "p_value": 1.0}
    p_value = binomtest(min(b, c), n, 0.5).pvalue
    return {"b": float(b), "c": float(c), "p_value": float(p_value)}


def failure_pareto(records: Iterable[TrialRecord]) -> list[tuple[str, int]]:
    counts = Counter(
        record.primary_failure_type
        for record in records
        if record.primary_failure_type and not record.success
    )
    return counts.most_common()


def aggregate_by_dimension(records: Iterable[TrialRecord]) -> dict[str, float]:
    """Unweighted mean success per paper-style capability dimension."""
    records = list(records)
    dimensions = {
        "spatial_understanding": {3, 6, 7},
        "object_generalization": {2},
        "instruction_following": {1},
        "embodied_reasoning": {8},
        "execution": {4, 5},
    }
    result: dict[str, float] = {}
    for name, stages in dimensions.items():
        relevant = [
            record
            for record in records
            if record.primary_failure_stage is None or record.primary_failure_stage in stages
        ]
        result[name] = (
            sum(1 for record in relevant if record.success) / len(relevant)
            if relevant
            else 0.0
        )
    return result
