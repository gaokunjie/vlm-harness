#!/usr/bin/env python3
"""Build a PROVISIONAL front calibration from a saved calibration audit.

This does not touch the robot.  It is for pipeline/demo work only: the board
pose spread in the saved samples is far above the formal gates, so the result
is written with ``calibration_status: PROVISIONAL`` and must not be used for
the formal DOE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from dobot_rua.calibration.front_planar import (
    CalibrationRejected,
    PlanarSample,
    camera_controls_from_config,
    fit_two_plane,
    save_planar_calibration,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, default=Path("audit/front_planar_last.json"))
    p.add_argument("--camera-config", type=Path, default=Path("configs/cameras.yaml"))
    p.add_argument("--task-config", type=Path, default=Path("configs/task.yaml"))
    p.add_argument("--max-fit-px", type=float, default=20.0)
    p.add_argument("--max-holdout-px", type=float, default=35.0)
    p.add_argument("--min-markers", type=int, default=10)
    p.add_argument("--no-save", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    data = json.loads(args.audit.read_text(encoding="utf-8"))
    samples = [
        PlanarSample(
            tip_xyz_m=tuple(sample["tip_xyz_m"]),
            cup_pixel_uv=tuple(sample["cup_pixel_uv"]),
            plane=sample["plane"],
            marker_count=int(sample["marker_count"]),
            board_reprojection_px=float(sample["board_reprojection_px"]),
        )
        for sample in data["samples"]
    ]
    camera_config = yaml.safe_load(
        args.camera_config.read_text(encoding="utf-8")
    ) or {}
    front = (camera_config.get("cameras") or {}).get("front") or {}
    task = yaml.safe_load(args.task_config.read_text(encoding="utf-8")) or {}
    intrinsic = np.asarray(front["intrinsic_K"], dtype=float)
    distortion = np.asarray(front["distortion"], dtype=float).reshape(-1)
    controls = camera_controls_from_config(front)
    p_cup = tuple(
        float(value) for value in front.get("cup_center_in_board_mm", [25.0, 75.0, 0.0])
    )
    table_z_m = float(task.get("table_z_m", 0.0))
    block_height_m = float(
        task.get("block_height_m", task.get("block_size_m", 0.025))
    )
    board_thickness_m = float(task.get("board_thickness_m", 0.002))
    print(
        f"audit samples={len(samples)} normal_spread="
        f"{data.get('board_normal_spread_deg', 'unknown')} deg"
    )
    print(
        "WARNING: this is a PROVISIONAL calibration from a flexing paper board; "
        "it must not be used for formal DOE."
    )
    try:
        calibration, diagnostics = fit_two_plane(
            samples,
            intrinsic,
            distortion,
            table_z_m,
            table_z_m + block_height_m,
            p_cup,
            block_height_m,
            board_thickness_m,
            controls,
            holdout=2,
            min_marker_count=args.min_markers,
            max_fit_rmse_px=args.max_fit_px,
            max_holdout_px=args.max_holdout_px,
            calibration_status="PROVISIONAL",
        )
    except CalibrationRejected as error:
        print(f"provisional calibration rejected: {error}")
        return 1
    print(
        f"fit_rmse={diagnostics['fit_rmse_px']:.3f}px "
        f"holdout_max={diagnostics['holdout_max_px']:.3f}px "
        f"samples={diagnostics['usable_samples']} "
        f"hash={calibration.calibration_hash}"
    )
    if not args.no_save:
        save_planar_calibration(args.camera_config, "front", calibration)
        print(f"wrote {args.camera_config} with calibration_status=PROVISIONAL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
