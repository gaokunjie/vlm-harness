"""Model-facing perception tools for the Dobot line-intersection task.

These tools are read-only over the latest frozen snapshot. The handlers are
wired to the perception modules in the environment-service phase; keeping the
schemas here lets the integration spec be loaded and tested without hardware.
"""

from __future__ import annotations

from typing import Any

from uni_robot_agent.tools.toolkit import (
    ToolOutput,
    ToolProvider,
    ToolSpec,
    bind_tools,
    schema,
)

NAMES = (
    "list_objects",
    "find_line_intersection",
    "wrist_contact_gap",
    "wrist_verify_grasp",
    "wrist_lateral_offset",
    "verify_place",
)

SNAPSHOT_ID = {
    "type": "string",
    "description": "Optional snapshot id from capture_scene/view_env_state; defaults to the latest snapshot.",
}


def tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            "list_objects",
            "Detect colored blocks in the latest frozen snapshot. Returns block ids, colors, "
            "world XY positions and confidence. Use after capture_scene.",
            schema({"snapshot_id": SNAPSHOT_ID}, ()),
            readonly=True,
        ),
        ToolSpec(
            "find_line_intersection",
            "Detect the two dominant lines of the requested color and return their intersection "
            "in world coordinates. The instruction names the color (black or red).",
            schema(
                {
                    "line_color": {
                        "type": "string",
                        "enum": ["black", "red", "blue", "green"],
                    },
                    "snapshot_id": SNAPSHOT_ID,
                },
                ("line_color",),
            ),
            readonly=True,
        ),
        ToolSpec(
            "wrist_contact_gap",
            "Measure the vertical gap between the suction-cup edge and the block top in the "
            "wrist view. Returns gap_mm and contact.",
            schema({"snapshot_id": SNAPSHOT_ID}, ()),
            readonly=True,
        ),
        ToolSpec(
            "wrist_verify_grasp",
            "Check whether the block is still attached to the suction cup after lifting.",
            schema({"snapshot_id": SNAPSHOT_ID}, ()),
            readonly=True,
        ),
        ToolSpec(
            "wrist_lateral_offset",
            "Return a local lateral correction between the suction tip and the target. "
            "The measurable axis depends on the forward-looking wrist camera.",
            schema(
                {
                    "target": {"type": "string", "enum": ["block", "intersection"]},
                    "snapshot_id": SNAPSHOT_ID,
                },
                ("target",),
            ),
            readonly=True,
        ),
        ToolSpec(
            "verify_place",
            "Compare the final block position with the requested intersection in the latest "
            "snapshot. Returns placed, error_mm and evidence.",
            schema({"snapshot_id": SNAPSHOT_ID}, ()),
            readonly=True,
        ),
    ]


class PerceptionTools(ToolProvider):
    provider_id = "perception"

    def __init__(self, episode: Any) -> None:
        self.episode = episode

    def get_tools(self):
        return bind_tools(
            tool_specs(),
            provider_id=self.provider_id,
            handlers={name: getattr(self, name) for name in NAMES},
        )

    def _not_wired(self, name: str) -> ToolOutput:
        return ToolOutput(
            success=False,
            error=f"{name} is not wired to the environment snapshot yet",
            extra={"no_execution": True},
        )

    def list_objects(self, snapshot_id: str | None = None) -> ToolOutput:
        return self._not_wired("list_objects")

    def find_line_intersection(self, line_color: str, snapshot_id: str | None = None) -> ToolOutput:
        return self._not_wired("find_line_intersection")

    def wrist_contact_gap(self, snapshot_id: str | None = None) -> ToolOutput:
        return self._not_wired("wrist_contact_gap")

    def wrist_verify_grasp(self, snapshot_id: str | None = None) -> ToolOutput:
        return self._not_wired("wrist_verify_grasp")

    def wrist_lateral_offset(
        self, target: str, snapshot_id: str | None = None
    ) -> ToolOutput:
        return self._not_wired("wrist_lateral_offset")

    def verify_place(self, snapshot_id: str | None = None) -> ToolOutput:
        return self._not_wired("verify_place")
