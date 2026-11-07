"""Milestone rubric: score is the sum of weights before the first failure."""

from __future__ import annotations

from dataclasses import dataclass, field

from .ledger import TrialRecord

DEFAULT_WEIGHTS: dict[int, float] = {
    1: 5.0,
    2: 10.0,
    3: 15.0,
    4: 20.0,
    5: 10.0,
    6: 15.0,
    7: 20.0,
    8: 5.0,
}

STAGE_NAMES: dict[int, str] = {
    1: "instruction_parse",
    2: "block_identification",
    3: "block_localization",
    4: "approach_and_grasp",
    5: "transport",
    6: "intersection_localization",
    7: "placement",
    8: "completion",
}

DEFAULT_FAILURES: dict[int, str] = {
    1: "color_parse_error",
    2: "block_identification_error",
    3: "block_localization_error",
    4: "grasp_failure",
    5: "transport_drop",
    6: "intersection_localization_error",
    7: "place_offset",
    8: "false_completion",
}


@dataclass(frozen=True)
class RubricConfig:
    weights: dict[int, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    pick_tolerance_mm: float = 6.0
    target_tolerance_mm: float = 8.0
    place_radius_mm: float = 17.5


@dataclass(frozen=True)
class StageResult:
    stage: int
    name: str
    passed: bool
    failure_type: str | None = None
    attribution: str | None = None


@dataclass(frozen=True)
class RubricResult:
    score: float
    success: bool
    stages: tuple[StageResult, ...]
    primary_failure_stage: int | None
    primary_failure_type: str | None
    failure_attribution: str | None


def _stage_checks(record: TrialRecord, config: RubricConfig) -> list[tuple[int, bool]]:
    return [
        (1, bool(record.instruction_parse_correct)),
        (
            2,
            bool(
                record.color_correct
                and not record.target_missed
                and not record.block_line_confusion
            ),
        ),
        (
            3,
            record.block_error_mm is not None
            and record.block_error_mm <= config.pick_tolerance_mm,
        ),
        (4, bool(record.grasp_success)),
        (5, not bool(record.transport_drop)),
        (
            6,
            bool(record.target_selection_correct)
            and record.intersection_error_mm is not None
            and record.intersection_error_mm <= config.target_tolerance_mm,
        ),
        (
            7,
            bool(record.release_ok)
            and record.place_error_mm is not None
            and record.place_error_mm <= config.place_radius_mm,
        ),
        (8, bool(record.completion_correct)),
    ]


def score_trial(record: TrialRecord, config: RubricConfig | None = None) -> RubricResult:
    config = config or RubricConfig()
    stages: list[StageResult] = []
    score = 0.0
    primary_stage: int | None = None
    primary_type: str | None = None

    for stage, passed in _stage_checks(record, config):
        if primary_stage is not None:
            stages.append(StageResult(stage, STAGE_NAMES[stage], False))
            continue
        if passed:
            stages.append(StageResult(stage, STAGE_NAMES[stage], True))
            score += float(config.weights[stage])
            continue
        primary_stage = stage
        primary_type = record.failure_types.get(stage, DEFAULT_FAILURES[stage])
        stages.append(
            StageResult(
                stage,
                STAGE_NAMES[stage],
                False,
                failure_type=primary_type,
                attribution=record.failure_attribution,
            )
        )

    return RubricResult(
        score=score,
        success=primary_stage is None,
        stages=tuple(stages),
        primary_failure_stage=primary_stage,
        primary_failure_type=primary_type,
        failure_attribution=record.failure_attribution if primary_stage is not None else None,
    )
