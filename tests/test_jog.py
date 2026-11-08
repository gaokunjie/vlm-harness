from dobot_rua.hardware.jog import JogBox, next_target


BOX = JogBox(x=(100.0, 220.0), y=(-80.0, 80.0), z=(0.0, 120.0), r=(-45.0, 45.0))


def test_wasd_maps_to_base_axes():
    start = (150.0, 0.0, 20.0, 0.0)
    assert next_target(start, "w", 2.0, 5.0, BOX).target == (152.0, 0.0, 20.0, 0.0)
    assert next_target(start, "s", 2.0, 5.0, BOX).target == (148.0, 0.0, 20.0, 0.0)
    assert next_target(start, "a", 2.0, 5.0, BOX).target == (150.0, 2.0, 20.0, 0.0)
    assert next_target(start, "d", 2.0, 5.0, BOX).target == (150.0, -2.0, 20.0, 0.0)
    assert next_target(start, "r", 2.0, 5.0, BOX).target == (150.0, 0.0, 22.0, 0.0)
    assert next_target(start, "f", 2.0, 5.0, BOX).target == (150.0, 0.0, 18.0, 0.0)
    assert next_target(start, "e", 2.0, 5.0, BOX).target == (150.0, 0.0, 20.0, 5.0)
    assert next_target(start, "q", 2.0, 5.0, BOX).target == (150.0, 0.0, 20.0, -5.0)


def test_unknown_key_returns_none():
    assert next_target((150.0, 0.0, 20.0, 0.0), "z", 2.0, 5.0, BOX) is None


def test_target_is_clamped_to_the_safe_box():
    result = next_target((219.0, 0.0, 20.0, 0.0), "w", 5.0, 5.0, BOX)
    assert result.target == (220.0, 0.0, 20.0, 0.0)
    assert result.clamped is True
    low = next_target((150.0, 0.0, 1.0, 0.0), "f", 5.0, 5.0, BOX)
    assert low.target == (150.0, 0.0, 0.0, 0.0)
    assert low.clamped is True
