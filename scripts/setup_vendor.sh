#!/usr/bin/env bash
# Recreate the pinned uni-robot-agent checkout and the integration shim.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
COMMIT="$(cat "$ROOT/vendor/UNI_ROBOT_AGENT_COMMIT")"
VENDOR="$ROOT/vendor/uni-robot-agent"

if [[ ! -d "$VENDOR/uni_robot_agent" ]]; then
  git clone https://github.com/Open-RUA/uni-robot-agent.git "$VENDOR"
  git -C "$VENDOR" checkout "$COMMIT"
else
  echo "vendor already present; expected commit $COMMIT"
fi

mkdir -p "$VENDOR/uni_robot_agent/integrations/dobot_magician"
cat > "$VENDOR/uni_robot_agent/integrations/dobot_magician/__init__.py" <<'PY'
"""Dobot Magician integration shim; implementation lives in the dobot_rua package."""


def get_integration_spec():
    from dobot_rua.integration.spec import get_integration_spec as _get

    return _get()
PY
echo "vendor ready: $VENDOR @ $COMMIT"
