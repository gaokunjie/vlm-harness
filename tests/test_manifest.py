import pytest

from dobot_rua.eval.manifest import Manifest, PlannedTrial, assert_disjoint


def _trial(plan_id="p1", split="dev", seed=1, instruction="Place the yellow block at the black lines intersection."):
    return PlannedTrial(
        plan_id=plan_id,
        split=split,
        seed=seed,
        target_block_color="yellow",
        target_line_color="black",
        instruction=instruction,
        block_xy_planned_mm=(100.0, 50.0),
        distractor_present=True,
        distractor_color="blue",
        block_id=0,
        run_order=1,
    )


def test_manifest_roundtrip_and_stable_hash():
    manifest = Manifest(experiment_id="e1", split="dev", seed=1, trials=[_trial()])
    restored = Manifest.from_dict(manifest.to_dict())
    assert restored == manifest
    assert restored.stable_hash() == manifest.stable_hash()


def test_manifest_hash_changes_with_content():
    a = Manifest(experiment_id="e1", split="dev", seed=1, trials=[_trial()])
    b = Manifest(
        experiment_id="e1",
        split="dev",
        seed=1,
        trials=[_trial(instruction="Put the yellow cube on the black cross.")],
    )
    assert a.stable_hash() != b.stable_hash()


def test_manifest_rejects_duplicate_plan_ids():
    with pytest.raises(ValueError):
        Manifest(
            experiment_id="e1",
            split="dev",
            seed=1,
            trials=[_trial("p1"), _trial("p1")],
        )


def test_manifests_are_disjoint():
    dev = Manifest(experiment_id="e1", split="dev", seed=1, trials=[_trial("p1", "dev", seed=1)])
    test = Manifest(experiment_id="e1", split="test", seed=2, trials=[_trial("p2", "test", seed=2)])
    assert_disjoint(dev, test)
    with pytest.raises(ValueError):
        assert_disjoint(dev, dev)


def test_assert_disjoint_rejects_same_layout_across_splits():
    dev = Manifest(experiment_id="e1", split="dev", seed=1, trials=[_trial("p1", "dev")])
    test = Manifest(experiment_id="e1", split="test", seed=1, trials=[_trial("p2", "test")])
    with pytest.raises(ValueError):
        assert_disjoint(dev, test)
