#!/usr/bin/env python3
"""Fit ``configs/fk_magician.yaml`` to the firmware's own pose readings.

Always home the arm first: the firmware's joint zeros are established by
homing, and a fit collected on an unhomed arm bakes the wrong zeros into
``c2``/``c3``.

    python scripts/fit_fk.py --collect          # sweep poses, write samples
    python scripts/fit_fk.py                    # fit, print residuals
    python scripts/fit_fk.py --save             # fit and update the config
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from dobot_rua.calibration.fk_fit import FKSample, fit_fk
from dobot_rua.hardware.dobot import DobotRobot
from dobot_rua.kinematics import load_limits, within_limits

# Cartesian sweep: x and z drive the shoulder/elbow angles, y drives the base
# yaw. Poses stay away from the far/up corner, where the USB cable is strained.
POSES: list[tuple[float, float, float]] = [
    (199.0, 0.0, -8.0),
    (199.0, 0.0, 30.0),
    (199.0, 0.0, 70.0),
    (199.0, 0.0, 105.0),
    (150.0, 0.0, 0.0),
    (150.0, 0.0, 40.0),
    (150.0, 0.0, 90.0),
    (150.0, 0.0, 125.0),
    (240.0, 0.0, 0.0),
    (240.0, 0.0, 40.0),
    (240.0, 0.0, 85.0),
    (199.0, 70.0, 20.0),
    (199.0, -70.0, 20.0),
    (199.0, 70.0, 80.0),
    (199.0, -70.0, 80.0),
    (150.0, 60.0, 40.0),
    (150.0, -60.0, 40.0),
    (230.0, 60.0, 40.0),
    (230.0, -60.0, 40.0),
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=Path, default=Path("audit/fk_samples.json"))
    p.add_argument("--config", type=Path, default=Path("configs/fk_magician.yaml"))
    p.add_argument("--limits", type=Path, default=Path("configs/robot_limits.yaml"))
    p.add_argument(
        "--limit-margin",
        type=float,
        default=2.0,
        help="drop samples closer than this many degrees to a joint limit",
    )
    p.add_argument("--collect", action="store_true", help="move the arm and record samples")
    p.add_argument("--save", action="store_true", help="write the fitted parameters")
    p.add_argument("--robot-port", default="/dev/dobot_follower")
    p.add_argument("--speed", type=int, default=40)
    return p


def collect(args) -> int:
    robot = DobotRobot(port=args.robot_port)
    robot.connect()
    robot.set_speed(args.speed, args.speed)
    samples = []
    misses = 0
    start = robot.read_state()
    try:
        for x, y, z in POSES:
            robot.move_pose(x, y, z, 0.0, timeout_s=25.0)
            time.sleep(0.3)
            state = robot.read_state()
            error = float(
                np.linalg.norm(
                    [state.x_mm - x, state.y_mm - y, state.z_mm - z]
                )
            )
            if error > 4.0:
                misses += 1
                print(
                    f"({x:6.1f},{y:6.1f},{z:6.1f}) MISSED by {error:.1f} mm "
                    f"(reported {state.x_mm:.1f},{state.y_mm:.1f},{state.z_mm:.1f})"
                )
                if misses >= 3:
                    print("three poses in a row missed; stopping the sweep")
                    break
                continue
            misses = 0
            samples.append(
                {
                    "commanded": [x, y, z],
                    "j1": state.j1_deg,
                    "j2": state.j2_deg,
                    "j3": state.j3_deg,
                    "j4": state.j4_deg,
                    "x": state.x_mm,
                    "y": state.y_mm,
                    "z": state.z_mm,
                }
            )
            print(
                f"({x:6.1f},{y:6.1f},{z:6.1f}) -> j=({state.j1_deg:7.2f},"
                f"{state.j2_deg:7.2f},{state.j3_deg:7.2f})  "
                f"pose=({state.x_mm:7.1f},{state.y_mm:7.1f},{state.z_mm:7.1f})"
            )
    except Exception as exc:  # serial drop: keep the samples already taken
        print(f"sweep aborted: {exc}")
    finally:
        try:
            robot.move_pose(start.x_mm, start.y_mm, start.z_mm, start.r_deg, timeout_s=25.0)
        finally:
            robot.close(release=False)
    args.samples.parent.mkdir(parents=True, exist_ok=True)
    args.samples.write_text(json.dumps({"samples": samples}, indent=1), encoding="utf-8")
    print(f"wrote {len(samples)} samples to {args.samples}")
    return 0


def load_samples(path: Path) -> list[FKSample]:
    data = json.loads(path.read_text(encoding="utf-8"))["samples"]
    return [
        FKSample(
            j1_deg=float(s["j1"]),
            j2_deg=float(s["j2"]),
            j3_deg=float(s["j3"]),
            x_mm=float(s["x"]),
            y_mm=float(s["y"]),
            z_mm=float(s["z"]),
        )
        for s in data
    ]


def main() -> int:
    args = parser().parse_args()
    if args.collect:
        return collect(args)

    samples = load_samples(args.samples)
    limits = load_limits(args.limits)
    kept = [
        s
        for s in samples
        if within_limits(
            (s.j1_deg, s.j2_deg, s.j3_deg), limits, margin_deg=args.limit_margin
        )
    ]
    dropped = len(samples) - len(kept)
    if dropped:
        print(
            f"dropped {dropped} sample(s) within {args.limit_margin} deg of a joint limit"
        )
    samples = kept
    fit = fit_fk(samples)
    p = fit.params
    print(f"samples {len(samples)}")
    print(f"a={p.a:.3f} h={p.h:.3f} L1={p.l1:.3f} L2={p.l2:.3f} "
          f"c2={p.c2_deg:.3f} c3={p.c3_deg:.3f}")
    print(f"residual RMS {fit.rms_mm:.3f} mm  max {fit.max_mm:.3f} mm  "
          f"azimuth max {fit.azimuth_max_deg:.4f} deg")
    worst = int(np.argmax(fit.residuals_mm))
    print(f"worst sample #{worst + 1}: {fit.residuals_mm[worst]:.2f} mm")

    if not args.save:
        return 0
    config = {
        "a": round(p.a, 6),
        "h": round(p.h, 6),
        "L1": round(p.l1, 6),
        "L2": round(p.l2, 6),
        "c2_deg": round(p.c2_deg, 6),
        "c3_deg": round(p.c3_deg, 6),
        "samples": len(samples),
        "holdout_median_mm": round(float(np.median(fit.residuals_mm)), 3),
        "holdout_max_mm": round(fit.max_mm, 3),
    }
    args.config.write_text(
        "# 由 scripts/fit_fk.py 生成，勿手改；回零后的固件位姿为参考\n"
        + yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"wrote {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
