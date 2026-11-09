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

from ..geometry import point_in_polygon
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

    def _front_frame(
        self, snapshot: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        camera = snapshot["cameras"]["front"]
        calibration = camera.get("calibration")
        if calibration is not None:
            if calibration.get("status") not in {"VALID", "PROVISIONAL"}:
                raise FrontCameraUncalibrated(
                    "front camera planar calibration is not VALID or PROVISIONAL; "
                    "recalibrate before using world coordinates"
                )
        elif not self._is_mock():
            raise FrontCameraUncalibrated(
                "front camera planar calibration is missing; run "
                "scripts/calibrate_front_planar.py before using world coordinates"
            )
        extrinsic_value = camera.get("T_world_from_optical")
        if extrinsic_value is None:
            raise FrontCameraUncalibrated(
                "front camera is not calibrated: T_world_from_optical is missing"
            )
        intrinsic_value = camera.get("intrinsic_K")
        if intrinsic_value is None:
            raise FrontCameraUncalibrated(
                "front camera is not calibrated: intrinsic_K is missing"
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
        intrinsic = np.asarray(intrinsic_value, dtype=float)
        distortion_value = (calibration or {}).get("distortion")
        distortion = (
            None
            if distortion_value is None
            else np.asarray(distortion_value, dtype=float).reshape(-1)
        )
        return image, intrinsic, extrinsic, distortion

    def _calibrated_region(self, snapshot: dict) -> np.ndarray | None:
        calibration = snapshot["cameras"]["front"].get("calibration") or {}
        region = calibration.get("calibrated_region")
        if region is None:
            if self._is_mock():
                return None
            raise FrontCameraUncalibrated(
                "front camera calibrated_region is missing; recalibrate"
            )
        return np.asarray(region, dtype=float).reshape(-1, 2)

    def _calibration_status(self, snapshot: dict) -> str | None:
        calibration = snapshot["cameras"]["front"].get("calibration") or {}
        return calibration.get("status")

    def _front_or_error(self, snapshot: dict):
        try:
            frame = self._front_frame(snapshot)
            # Validate the region here too so a missing region is returned as a
            # structured tool error rather than escaping the handler.
            self._calibrated_region(snapshot)
            return frame, None
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

    def _table_z(self) -> float:
        return float(self._metadata().get("table_z_m", 0.0))

    def _plane_z(self) -> float:
        return self._table_z()

    def _block_top_z(self) -> float:
        return float(self._metadata().get("block_top_z_m", 0.025))

    def _place_radius_m(self) -> float:
        return float(self._metadata().get("place_radius_m", 0.0175))

    def _is_mock(self) -> bool:
        return bool(self._metadata().get("mock", False))

    # ---- handlers ----
    def list_objects(self, snapshot_id: str | None = None) -> ToolOutput:
        snapshot = self._snapshot(snapshot_id)
        frame, error = self._front_or_error(snapshot)
        if error is not None:
            return error
        image, intrinsic, extrinsic, distortion = frame
        region = self._calibrated_region(snapshot)
        blocks = detect_blocks(
            image,
            intrinsic,
            extrinsic,
            self._block_top_z(),
            colors=tuple(self._metadata().get("block_colors", ("blue", "yellow"))),
            distortion=distortion,
        )
        if region is not None:
            blocks = [
                block
                for block in blocks
                if point_in_polygon(
                    np.asarray(block.center_xyz_m[:2], dtype=float), region
                )
            ]
        return ToolOutput(
            extra={
                "snapshot_id": snapshot["snapshot_id"],
                "blocks": [asdict(block) for block in blocks],
                "calibration_status": self._calibration_status(snapshot),
                "provisional": self._calibration_status(snapshot) == "PROVISIONAL",
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
        image, intrinsic, extrinsic, distortion = frame
        region = self._calibrated_region(snapshot)
        result = detect_line_intersection(
            image,
            intrinsic,
            extrinsic,
            self._table_z(),
            line_color,
            distortion=distortion,
            region_world=region,
        )
        if result is None:
            return ToolOutput(
                success=False,
                error=f"no {line_color} line intersection found",
                extra={"snapshot_id": snapshot["snapshot_id"], "mock": self._is_mock()},
            )
        if region is not None and not point_in_polygon(
            np.asarray(result.intersection_xyz_m[:2], dtype=float), region
        ):
            return ToolOutput(
                success=False,
                error="intersection outside calibrated_region",
                extra={
                    "snapshot_id": snapshot["snapshot_id"],
                    "intersection_xyz_m": result.intersection_xyz_m,
                    "calibration_status": self._calibration_status(snapshot),
                    "provisional": self._calibration_status(snapshot) == "PROVISIONAL",
                    "mock": self._is_mock(),
                },
            )
        return ToolOutput(
            extra={
                "snapshot_id": snapshot["snapshot_id"],
                **asdict(result),
                "calibration_status": self._calibration_status(snapshot),
                "provisional": self._calibration_status(snapshot) == "PROVISIONAL",
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
        image, intrinsic, extrinsic, distortion = frame
        region = self._calibrated_region(snapshot)
        blocks = detect_blocks(
            image,
            intrinsic,
            extrinsic,
            self._block_top_z(),
            colors=tuple(self._metadata().get("block_colors", ("blue", "yellow"))),
            distortion=distortion,
        )
        if region is not None:
            blocks = [
                block
                for block in blocks
                if point_in_polygon(
                    np.asarray(block.center_xyz_m[:2], dtype=float), region
                )
            ]
        if not blocks:
            return ToolOutput(
                success=False,
                error="no block found for verify_place",
                extra={"snapshot_id": snapshot["snapshot_id"], "mock": False},
            )
        intersections = {}
        for color in ("black", "red"):
            result = detect_line_intersection(
                image,
                intrinsic,
                extrinsic,
                self._table_z(),
                color,
                distortion=distortion,
                region_world=region,
            )
            if result is None:
                continue
            if region is not None and not point_in_polygon(
                np.asarray(result.intersection_xyz_m[:2], dtype=float), region
            ):
                continue
            intersections[color] = result
        if not intersections:
            return ToolOutput(
                success=False,
                error="no line intersection found for verify_place",
                extra={"snapshot_id": snapshot["snapshot_id"], "mock": False},
            )
        best_error = None
        best_block = None
        best_color = None
        for block in blocks:
            for color, result in intersections.items():
                error_mm = float(
                    np.hypot(
                        block.center_xyz_m[0] - result.intersection_xyz_m[0],
                        block.center_xyz_m[1] - result.intersection_xyz_m[1],
                    )
                    * 1000.0
                )
                if best_error is None or error_mm < best_error:
                    best_error = error_mm
                    best_block = block
                    best_color = color
        placed = bool(best_error is not None and best_error <= self._place_radius_m() * 1000.0)
        return ToolOutput(
            extra={
                "snapshot_id": snapshot["snapshot_id"],
                "placed": placed,
                "error_mm": best_error,
                "matched_line_color": best_color,
                "block": asdict(best_block) if best_block is not None else None,
                "intersections": {
                    color: asdict(result) for color, result in intersections.items()
                },
                "calibration_status": self._calibration_status(snapshot),
                "provisional": self._calibration_status(snapshot) == "PROVISIONAL",
                "mock": False,
            }
        )
