"""Dobot Magician integration shim; implementation lives in the dobot_rua package."""


def get_integration_spec():
    from dobot_rua.integration.spec import get_integration_spec as _get

    return _get()
