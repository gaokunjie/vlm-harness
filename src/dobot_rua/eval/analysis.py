"""Aggregate statistics for the DOE report."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

from scipy.stats import binomtest

from .ledger import TrialRecord
from .rubric import RubricConfig, stage_passes


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


DIMENSION_STAGES: dict[str, set[int]] = {
    "spatial_understanding": {3, 6, 7},
    "object_generalization": {2},
    "instruction_following": {1},
    "embodied_reasoning": {8},
    "execution": {4, 5},
}


def aggregate_by_dimension(
    records: Iterable[TrialRecord], config: RubricConfig | None = None
) -> dict[str, float | None]:
    """Paper-style 0–20 score per dimension, then an unweighted mean.

    Each record contributes ``20 * passed_weight / dimension_weight``. The
    two paper dimensions with no dedicated stages in this one-task rubric are
    returned as ``None`` rather than a fabricated zero.
    """
    config = config or RubricConfig()
    records = list(records)
    result: dict[str, float | None] = {}
    for name, stages in DIMENSION_STAGES.items():
        total_weight = sum(float(config.weights[stage]) for stage in stages)
        if not records:
            result[name] = 0.0
            continue
        scores = []
        for record in records:
            passed = stage_passes(record, config)
            earned = sum(
                float(config.weights[stage]) for stage in stages if passed[stage]
            )
            scores.append(20.0 * earned / total_weight)
        result[name] = sum(scores) / len(scores)
    result["compositional_generalization"] = None
    result["unseen_tasks"] = None
    return result
