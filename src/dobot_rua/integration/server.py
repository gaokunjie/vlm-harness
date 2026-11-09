"""Real Dobot Magician environment service for the official harness."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from uni_robot_agent.integrations.simulation import serve_episode

from ..hardware.cameras import RealCameraSource, load_camera_specs, load_tip_length_mm
from ..hardware.dobot import DobotRobot
from ..hardware.safety import SafetyLimits
from ..kinematics import load_fk
from .adapter import AdapterConfig, DobotAdapter


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dobot Magician environment service")
    p.add_argument("--run-id", required=True)
    p.add_argument("--instruction", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18760)
    p.add_argument("--robot-port", default="/dev/dobot_follower")
    p.add_argument("--camera-config", type=Path, default=Path("configs/cameras.yaml"))
    p.add_argument("--task-config", type=Path, default=Path("configs/task.yaml"))
    p.add_argument("--mock", action="store_true", help="Serve the synthetic mock instead of hardware")
    return p


def _task_metadata(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    table_z = float(data.get("table_z_m", 0.0))
    block_height = float(
        data.get("block_height_m", data.get("block_size_m", 0.025))
    )
    place_radius = float(data.get("place_radius_m", 0.0175))
    return {
        "task_id": "line_intersection",
        "table_z_m": table_z,
        "block_top_z_m": table_z + block_height,
        "block_height_m": block_height,
        "place_radius_m": place_radius,
        "block_colors": [
            str(color) for color in data.get("block_colors", ["blue", "yellow"])
        ],
    }


def prepare(args: argparse.Namespace) -> DobotAdapter:
    if args.mock:
        from .mock_server import prepare as mock_prepare

        return mock_prepare(args)

    robot = DobotRobot(port=args.robot_port)
    robot.connect()
    camera_source = RealCameraSource(load_camera_specs(args.camera_config))
    metadata = _task_metadata(args.task_config)
    calibration_status = camera_source.specs["front"].calibration_status
    metadata["calibration_status"] = calibration_status
    if calibration_status == "PROVISIONAL":
        print(
            "WARNING: front calibration is PROVISIONAL; this episode is for "
            "pipeline/demo only and must not enter the formal DOE"
        )
    adapter = DobotAdapter(
        robot=robot,
        camera_source=camera_source,
        config=AdapterConfig(
            has_calibration=camera_source.has_calibration,
            tip_length_mm=load_tip_length_mm(args.camera_config),
        ),
        episode_id=args.run_id,
        instruction=args.instruction,
        metadata=metadata,
        safety_limits=SafetyLimits.from_yaml("configs/robot_limits.yaml"),
        fk_params=load_fk("configs/fk_magician.yaml"),
    )
    # The real success criterion is measured offline from the final snapshot;
    # the harness's finish() must not receive a self-reported verdict.
    adapter.set_success_fn(lambda: None)
    return adapter


def main() -> None:
    args = parser().parse_args()
    serve_episode(lambda: prepare(args), args, "dobot_magician")


if __name__ == "__main__":
    main()
