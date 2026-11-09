def test_calibration_plan_builds_two_planes():
    from dobot_rua.calibration.front_planar import build_pose_plan

    plan = build_pose_plan(
        center_xy_mm=(150.0, 0.0),
        spacing_mm=20.0,
        cols=3,
        rows=3,
        table_z_mm=-21.0,
        block_height_mm=25.0,
        board_thickness_mm=2.0,
    )
    assert len(plan) == 18  # 9 per plane
    assert {p.plane for p in plan} == {"table", "top"}
