"""Mock environment service for the official uni-robot-agent harness smoke.

No hardware is touched: a FakeRobot and a synthetic front/side/wrist image are
served through the same EnvironmentBackend protocol as the real adapter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from uni_robot_agent.integrations.simulation import serve_episode
from uni_robot_agent.services.environment.snapshots import CameraFrame

from ..hardware.interface import FakeRobot
from .adapter import AdapterConfig, DobotAdapter

CAMERAS = ("front", "side", "wrist")
BLACK_INTERSECTION_MM = (160.0, -120.0)


def _mock_success(robot: FakeRobot) -> bool:
    """Mock-only success: final tip near the black intersection, vacuum off."""
    state = robot.read_state()
    distance = ((state.x_mm - BLACK_INTERSECTION_MM[0]) ** 2 + (state.y_mm - BLACK_INTERSECTION_MM[1]) ** 2) ** 0.5
    return distance <= 20.0 and not state.suction_on


def _scene() -> np.ndarray:
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (300, 220), (340, 260), (0, 255, 255), -1)  # yellow block
    cv2.rectangle(image, (180, 180), (220, 220), (255, 0, 0), -1)  # blue block
    cv2.line(image, (200, 300), (600, 300), (0, 0, 0), 3)
    cv2.line(image, (400, 100), (400, 500), (0, 0, 0), 3)
    cv2.line(image, (150, 350), (350, 350), (0, 0, 255), 3)
    cv2.line(image, (250, 250), (250, 450), (0, 0, 255), 3)
    return image


class MockCameraSource:
    def __init__(self) -> None:
        self._intrinsic = np.array(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        )
        self._extrinsic = np.eye(4)
        self._extrinsic[:3, :3] = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
        )
        self._extrinsic[:3, 3] = [0.0, 0.0, 1.0]
        self._image = _scene()

    def capture(self, cameras: list[str] | None = None) -> dict[str, CameraFrame]:
        names = list(cameras) if cameras else list(CAMERAS)
        # CameraFrame.rgb is RGB; the scene is drawn with OpenCV's BGR order.
        rgb = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
        return {
            name: CameraFrame(
                rgb=rgb.copy(),
                depth_m=None,
                intrinsic_K=self._intrinsic.copy(),
                T_world_from_optical=self._extrinsic.copy(),
            )
            for name in names
        }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mock Dobot environment service")
    p.add_argument("--run-id", required=True)
    p.add_argument("--instruction", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18760)
    return p


def prepare(args: argparse.Namespace) -> DobotAdapter:
    robot = FakeRobot()
    robot.connect()
    adapter = DobotAdapter(
        robot=robot,
        camera_source=MockCameraSource(),
        config=AdapterConfig(),
        episode_id=args.run_id,
        instruction=args.instruction,
        metadata={
            "task_id": "line_intersection",
            "mock": True,
            "table_z_m": 0.0,
            "block_top_z_m": 0.025,
        },
    )
    # Mock-only proxy for "the tip finished at the requested intersection with
    # the vacuum released". It does not simulate object transport.
    adapter.set_success_fn(lambda: _mock_success(robot))
    return adapter


def main() -> None:
    args = parser().parse_args()
    serve_episode(lambda: prepare(args), args, "dobot_magician")


if __name__ == "__main__":
    main()
