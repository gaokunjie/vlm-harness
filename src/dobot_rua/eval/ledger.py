"""Append-only trial ledger (JSONL; Parquet export belongs to the analysis phase)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrialRecord:
    trial_id: str
    plan_id: str
    experiment_id: str
    split: str
    success: bool
    score: float = 0.0

    # Task / configuration
    instruction_raw: str = ""
    target_block_color: str = ""
    target_line_color: str = ""
    target_id: str = ""
    distractor_present: bool = False
    distractor_color: str | None = None

    # Ground truth and estimates (millimetres)
    block_xy_gt_mm: tuple[float, float] | None = None
    est_block_xy_mm: tuple[float, float] | None = None
    block_error_mm: float | None = None
    intersection_gt_mm: tuple[float, float] | None = None
    est_intersection_mm: tuple[float, float] | None = None
    intersection_error_mm: float | None = None
    place_error_mm: float | None = None

    # Stage outcomes
    instruction_parse_correct: bool = False
    color_correct: bool = False
    target_missed: bool = False
    block_line_confusion: bool = False
    grasp_success: bool = False
    transport_drop: bool = False
    target_selection_correct: bool = False
    release_ok: bool = False
    completion_correct: bool = False

    # Diagnosis
    failure_types: dict[int, str] = field(default_factory=dict)
    primary_failure_stage: int | None = None
    primary_failure_type: str | None = None
    failure_attribution: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.failure_types:
            self.failure_types = {
                int(key): str(value) for key, value in self.failure_types.items()
            }


def write_trial(path: str | Path, record: TrialRecord) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def read_trials(path: str | Path) -> list[TrialRecord]:
    target = Path(path)
    if not target.exists():
        return []
    records: list[TrialRecord] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        data["failure_types"] = {
            int(key): value for key, value in (data.get("failure_types") or {}).items()
        }
        records.append(TrialRecord(**data))
    return records
