import numpy as np
import pytest

from dobot_rua.calibration.fk_fit import FKSample, fit_fk
from dobot_rua.kinematics import FKParams, forward_kinematics


def _samples(params: FKParams, rng, n: int) -> list[FKSample]:
    samples = []
    for _ in range(n):
        j1 = float(rng.uniform(-70.0, 70.0))
        j2 = float(rng.uniform(-10.0, 75.0))
        j3 = float(rng.uniform(0.0, 85.0))
        x, y, z = forward_kinematics(params, j1, j2, j3)
        samples.append(FKSample(j1, j2, j3, float(x), float(y), float(z)))
    return samples


def test_fit_recovers_a_known_chain():
    truth = FKParams(a=120.0, h=180.0, l1=140.0, l2=210.0, c2_deg=40.0, c3_deg=-25.0)
    rng = np.random.default_rng(3)
    fit = fit_fk(_samples(truth, rng, 30))

    assert fit.max_mm < 0.05
    for _ in range(40):
        j1 = float(rng.uniform(-70.0, 70.0))
        j2 = float(rng.uniform(-10.0, 75.0))
        j3 = float(rng.uniform(0.0, 85.0))
        predicted = forward_kinematics(fit.params, j1, j2, j3)
        assert np.allclose(predicted, forward_kinematics(truth, j1, j2, j3), atol=0.05)


def test_fit_reports_residuals_and_azimuth_error():
    truth = FKParams(a=120.0, h=180.0, l1=140.0, l2=210.0, c2_deg=40.0, c3_deg=-25.0)
    fit = fit_fk(_samples(truth, np.random.default_rng(7), 12))

    assert len(fit.residuals_mm) == 12
    assert fit.rms_mm <= fit.max_mm
    assert fit.azimuth_max_deg < 1e-6


def test_fit_needs_enough_samples():
    with pytest.raises(ValueError, match="at least 4"):
        fit_fk([FKSample(0.0, 45.0, 45.0, 199.4, 0.0, -8.4)])
