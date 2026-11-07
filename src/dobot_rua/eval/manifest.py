"""Immutable planned-trial manifests and split disjointness checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class PlannedTrial:
    plan_id: str
    split: str
    seed: int
    target_block_color: str
    target_line_color: str
    instruction: str
    block_xy_planned_mm: tuple[float, float] | None = None
    distractor_present: bool = False
    distractor_color: str | None = None
    block_id: int = 0
    run_order: int = 0
    notes: str = ""


@dataclass(frozen=True)
class Manifest:
    experiment_id: str
    split: str
    seed: int
    trials: tuple[PlannedTrial, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.split not in {"dev", "val", "test"}:
            raise ValueError(f"unknown split: {self.split}")
        trials = tuple(self.trials)
        plan_ids = [trial.plan_id for trial in trials]
        if len(plan_ids) != len(set(plan_ids)):
            raise ValueError("duplicate plan_id in manifest")
        object.__setattr__(self, "trials", trials)

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "split": self.split,
            "seed": self.seed,
            "trials": [asdict(trial) for trial in self.trials],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Manifest":
        return cls(
            experiment_id=str(data["experiment_id"]),
            split=str(data["split"]),
            seed=int(data["seed"]),
            trials=tuple(PlannedTrial(**trial) for trial in data["trials"]),
        )

    def stable_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assert_disjoint(*manifests: Manifest) -> None:
    seen: set[tuple[str, str]] = set()
    for manifest in manifests:
        for trial in manifest.trials:
            key = (trial.split, trial.plan_id)
            if key in seen:
                raise ValueError(f"duplicate planned trial across manifests: {key}")
            seen.add(key)
