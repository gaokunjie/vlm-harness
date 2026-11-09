"""Model-facing perception tools for the Dobot line-intersection task.

The tools read one immutable snapshot, run the perception modules, and return
structured JSON plus a debug image. They never expose GT or task verdicts.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import cv2
import numpy as np

from uni_robot_agent.tools.toolkit import (
    ToolOutput,
    ToolProvider,
    ToolSpec,
    bind_tools,
    schema,
)

from ..perception.blocks import detect_blocks
from ..perception.lines import detect_line_intersection

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


class FrontCameraUncalibrated(RuntimeError):
    """The front camera has no usable world extrinsic; fail closed."""


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

    # ---- snapshot plumbing ----
    def _snapshot(self, snapshot_id: str | None) -> dict:
        if snapshot_id:
            return self.episode.env.snapshot(snapshot_id)
        return self.episode.latest_snapshot

    def _front_frame(self, snapshot: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        camera = snapshot["cameras"]["front"]
        extrinsic_value = camera.get("T_world_from_optical")
        if extrinsic_value is None:
            raise FrontCameraUncalibrated(
                "front camera is not calibrated: T_world_from_optical is missing; "
                "run scripts/finish_front_extrinsic.py solve before using world coordinates"
            )
        extrinsic = np.asarray(extrinsic_value, dtype=float)
        if extrinsic.shape != (4, 4):
            raise FrontCameraUncalibrated(
                "front camera extrinsic must be a 4x4 homogeneous transform"
            )
        rotation = extrinsic[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or float(
            np.linalg.det(rotation)
        ) <= 0.0:
            raise FrontCameraUncalibrated(
                "front camera extrinsic is not a proper rigid transform; recalibrate "
                "before using world coordinates"
            )
        path = self.episode.artifact_mirror.local_path(camera["rgb"])
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"could not read front camera image at {path}")
        intrinsic = np.asarray(camera["intrinsic_K"], dtype=float)
        return image, intrinsic, extrinsic

    def _front_or_error(self, snapshot: dict):
        try:
            return self._front_frame(snapshot), None
        except FrontCameraUncalibrated as error:
            return None, ToolOutput(
                success=False,
                error=str(error),
                extra={
                    "snapshot_id": snapshot.get("snapshot_id"),
                    "mock": self._is_mock(),
                },
            )

    def _metadata(self) -> dict:
        return self.episode.description.environment.metadata

    def _plane_z(self) -> float:
        return float(self._metadata().get("table_z_m", 0.0))

    def _is_mock(self) -> bool:
        return bool(self._metadata().get("mock", False))

    # ---- handlers ----
    def list_objects(self, snapshot_id: str | None = None) -> ToolOutput:
        snapshot = self._snapshot(snapshot_id)
        frame, error = self._front_or_error(snapshot)
        if error is not None:
            return error
        image, intrinsic, extrinsic = frame
        blocks = detect_blocks(image, intrinsic, extrinsic, self._plane_z())
        return ToolOutput(
            extra={
                "snapshot_id": snapshot["snapshot_id"],
                "blocks": [asdict(block) for block in blocks],
                "mock": self._is_mock(),
            }
        )

    def find_line_intersection(
        self, line_color: str, snapshot_id: str | None = None
    ) -> ToolOutput:
        snapshot = self._snapshot(snapshot_id)
        frame, error = self._front_or_error(snapshot)
        if error is not None:
            return error
        image, intrinsic, extrinsic = frame
        result = detect_line_intersection(
            image, intrinsic, extrinsic, self._plane_z(), line_color
        )
        if result is None:
            return ToolOutput(
                success=False,
                error=f"no {line_color} line intersection found",
                extra={"snapshot_id": snapshot["snapshot_id"], "mock": self._is_mock()},
            )
        return ToolOutput(
            extra={
                "snapshot_id": snapshot["snapshot_id"],
                **asdict(result),
                "mock": self._is_mock(),
            }
        )

    def wrist_contact_gap(self, snapshot_id: str | None = None) -> ToolOutput:
        arm = self.episode.state.arms["arm"]
        z_m = float(arm.control_pose.position[2])
        block_top_m = float(self._metadata().get("block_top_z_m", 0.025))
        gap_mm = max(0.0, (z_m - block_top_m) * 1000.0)
        return ToolOutput(
            extra={
                "gap_mm": gap_mm,
                "contact": gap_mm <= 2.0,
                "mock": self._is_mock(),
            }
        )

    def wrist_verify_grasp(self, snapshot_id: str | None = None) -> ToolOutput:
        arm = self.episode.state.arms["arm"]
        z_m = float(arm.control_pose.position[2])
        block_top_m = float(self._metadata().get("block_top_z_m", 0.025))
        attached = float(arm.gripper) <= 0.5 and z_m > block_top_m + 0.005
        return ToolOutput(extra={"attached": attached, "mock": self._is_mock()})

    def wrist_lateral_offset(
        self, target: str, snapshot_id: str | None = None
    ) -> ToolOutput:
        # The forward-looking wrist camera's geometric servo is a hardware-phase
        # task. The mock returns a neutral correction and is explicit about it.
        return ToolOutput(
            extra={
                "offset_mm": 0.0,
                "axis": "x",
                "target": target,
                "confidence": 0.5 if self._is_mock() else 0.0,
                "mock": self._is_mock(),
            }
        )

    def verify_place(self, snapshot_id: str | None = None) -> ToolOutput:
        if self._is_mock():
            return ToolOutput(
                extra={
                    "placed": None,
                    "error": "mock environment does not simulate object transport",
                    "mock": True,
                }
            )
        snapshot = self._snapshot(snapshot_id)
        frame, error = self._front_or_error(snapshot)
        if error is not None:
            return error
        image, intrinsic, extrinsic = frame
        blocks = detect_blocks(image, intrinsic, extrinsic, self._plane_z())
        return ToolOutput(
            extra={
                "snapshot_id": snapshot["snapshot_id"],
                "blocks": [asdict(block) for block in blocks],
                "placed": None,
                "note": "compare block_xy with the requested intersection",
                "mock": False,
            }
        )
