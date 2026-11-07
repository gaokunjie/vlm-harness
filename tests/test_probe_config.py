import pytest

from dobot_rua.probe import ProbeError, probe


def test_probe_requires_api_key():
    with pytest.raises(ProbeError):
        probe(api_key="", n=1)


def test_probe_rejects_nonpositive_calls():
    with pytest.raises(ValueError):
        probe(api_key="not-a-real-key", n=0)


def test_probe_normalizes_unsupported_all_proxy(monkeypatch):
    import os

    from dobot_rua.probe import _normalize_proxy_env

    monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:7890")
    _normalize_proxy_env()
    value = os.environ.get("ALL_PROXY")
    assert value is None or value.startswith("socks5://")
