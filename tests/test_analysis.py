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


def test_aggregate_by_dimension_uses_stage_weights():
    from dobot_rua.eval.analysis import aggregate_by_dimension

    record = TrialRecord(
        trial_id="t1",
        plan_id="p1",
        experiment_id="e",
        split="dev",
        success=False,
        instruction_parse_correct=True,
        color_correct=True,
        block_error_mm=2.0,
        grasp_success=True,
        transport_drop=False,
        target_selection_correct=False,
        intersection_error_mm=30.0,
        release_ok=True,
        place_error_mm=5.0,
        completion_correct=True,
    )
    dims = aggregate_by_dimension([record])
    assert dims["spatial_understanding"] == pytest.approx(14.0)
    assert dims["object_generalization"] == pytest.approx(20.0)
    assert dims["instruction_following"] == pytest.approx(20.0)
    assert dims["execution"] == pytest.approx(20.0)
    assert dims["embodied_reasoning"] == pytest.approx(20.0)
    assert dims["compositional_generalization"] is None
    assert dims["unseen_tasks"] is None
