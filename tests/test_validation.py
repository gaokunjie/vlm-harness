import numpy as np

from dobot_rua.calibration.validation import summarize_validation


def test_validation_passes_small_errors():
    errors = np.array([[0.5, -0.3], [0.2, 0.4], [-0.1, 0.2]])
    report = summarize_validation(errors)
    assert report.passed is True
    assert report.bias_norm_mm < 2.0
    assert report.random_mm < 3.0


def test_validation_fails_large_bias():
    errors = np.array([[3.0, 0.0], [3.1, 0.1], [2.9, -0.1]])
    assert summarize_validation(errors).passed is False


def test_validation_fails_large_random():
    errors = np.array([[-4.0, 0.0], [4.0, 0.0], [0.0, 0.0]])
    assert summarize_validation(errors).passed is False
