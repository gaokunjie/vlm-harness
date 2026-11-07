import pytest

from dobot_rua.probe import ProbeError, probe


def test_probe_requires_api_key():
    with pytest.raises(ProbeError):
        probe(api_key="", n=1)


def test_probe_rejects_nonpositive_calls():
    with pytest.raises(ValueError):
        probe(api_key="not-a-real-key", n=0)
