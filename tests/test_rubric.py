import pytest

from dobot_rua.eval.ledger import TrialRecord
from dobot_rua.eval.rubric import RubricConfig, score_trial


def _record(**kwargs):
    base = dict(
        trial_id="t1",
        plan_id="p1",
        experiment_id="e1",
        split="dev",
        success=True,
        instruction_parse_correct=True,
        color_correct=True,
        target_missed=False,
        block_line_confusion=False,
        block_error_mm=2.0,
        grasp_success=True,
        transport_drop=False,
        target_selection_correct=True,
        intersection_error_mm=3.0,
        release_ok=True,
        place_error_mm=5.0,
        completion_correct=True,
    )
    base.update(kwargs)
    return TrialRecord(**base)


def test_all_stages_pass_scores_100_and_success():
    result = score_trial(_record(), RubricConfig())
    assert result.success is True
    assert result.score == 100.0
    assert result.primary_failure_stage is None


def test_failure_at_stage_6_truncates_score():
    result = score_trial(
        _record(
            target_selection_correct=False,
            failure_types={6: "wrong_line_color"},
            failure_attribution="model",
        ),
        RubricConfig(),
    )
    assert result.success is False
    assert result.primary_failure_stage == 6
    assert result.primary_failure_type == "wrong_line_color"
    assert result.failure_attribution == "model"
    assert result.score == pytest.approx(60.0)


def test_failure_at_stage_3_scores_only_first_two():
    result = score_trial(_record(block_error_mm=12.0), RubricConfig())
    assert result.primary_failure_stage == 3
    assert result.primary_failure_type == "block_localization_error"
    assert result.score == pytest.approx(15.0)
