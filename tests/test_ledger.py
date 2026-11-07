from dobot_rua.eval.ledger import TrialRecord, read_trials, write_trial


def _record(trial_id="t1", **kwargs):
    base = dict(
        trial_id=trial_id,
        plan_id="p1",
        experiment_id="e1",
        split="dev",
        success=True,
        score=100.0,
    )
    base.update(kwargs)
    return TrialRecord(**base)


def test_trial_ledger_roundtrip(tmp_path):
    path = tmp_path / "trials.jsonl"
    write_trial(path, _record("t1"))
    write_trial(path, _record("t2", success=False, score=60.0))
    rows = read_trials(path)
    assert [row.trial_id for row in rows] == ["t1", "t2"]
    assert rows[0].success is True
    assert rows[1].score == 60.0


def test_ledger_preserves_failure_diagnostics(tmp_path):
    path = tmp_path / "trials.jsonl"
    write_trial(
        path,
        _record(
            "t3",
            success=False,
            score=60.0,
            primary_failure_stage=6,
            primary_failure_type="wrong_line_color",
            failure_attribution="model",
        ),
    )
    row = read_trials(path)[0]
    assert row.primary_failure_stage == 6
    assert row.primary_failure_type == "wrong_line_color"
    assert row.failure_attribution == "model"
