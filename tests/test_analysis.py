import pytest

from dobot_rua.eval.analysis import failure_pareto, mcnemar, wilson_ci
from dobot_rua.eval.ledger import TrialRecord


def test_wilson_ci_known_value():
    low, high = wilson_ci(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-3)
    assert high == pytest.approx(0.5962, abs=1e-3)


def test_mcnemar_exact_known_value():
    result = mcnemar(b=10, c=2)
    assert result["p_value"] == pytest.approx(0.03857, abs=5e-4)


def test_failure_pareto_counts_primary_failures():
    records = [
        TrialRecord(trial_id="t1", plan_id="p1", experiment_id="e", split="dev", success=False,
                    primary_failure_type="wrong_line_color"),
        TrialRecord(trial_id="t2", plan_id="p2", experiment_id="e", split="dev", success=False,
                    primary_failure_type="wrong_line_color"),
        TrialRecord(trial_id="t3", plan_id="p3", experiment_id="e", split="dev", success=False,
                    primary_failure_type="suction_failure"),
        TrialRecord(trial_id="t4", plan_id="p4", experiment_id="e", split="dev", success=True),
    ]
    assert failure_pareto(records) == [("wrong_line_color", 2), ("suction_failure", 1)]
