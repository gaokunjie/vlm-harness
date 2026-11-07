"""IntegrationSpec for the Dobot Magician line-intersection task."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from uni_robot_agent.execution.scheduler import ToolExecutionPolicy
from uni_robot_agent.integrations.gripper import (
    build_gripper_spec,
    configure_tools as configure_gripper_tools,
    validate_gripper_cli,
)
from uni_robot_agent.integrations.prepared_episode import PreparedEpisodeSession

from .tools import NAMES as PERCEPTION_NAMES
from .tools import PerceptionTools


def configure_tools(session: Any):
    """Compose the official gripper/scene/skill/episode tools plus perception."""
    return (*configure_gripper_tools(session), PerceptionTools(session.episode))


def create_session(context: Any):
    return PreparedEpisodeSession(context, configure_tools=configure_tools)


def validate_cli(args: Any) -> None:
    validate_gripper_cli(args)


def get_integration_spec():
    spec = build_gripper_spec(
        name="dobot_magician",
        create_session=create_session,
        validate_cli=validate_cli,
    )
    policies = dict(spec.tool_policies)
    for name in PERCEPTION_NAMES:
        policies[name] = ToolExecutionPolicy(
            resource_key="episode",
            return_mode="immediate",
            concurrency="parallel",
            mutates_world=False,
        )
    return replace(
        spec,
        eval_tool_names=(*spec.eval_tool_names, *PERCEPTION_NAMES),
        tool_policies=policies,
    )
