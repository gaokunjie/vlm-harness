"""Environment service entry point for the Dobot Magician episode.

The real camera source and hardware configuration are filled in during the
hardware bring-up phase. Until then this module fails loudly instead of
pretending to own a robot.
"""

from __future__ import annotations

import argparse


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dobot Magician environment service")
    p.add_argument("--run-id", required=True)
    p.add_argument("--instruction", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18760)
    return p


def prepare(args: argparse.Namespace):
    raise RuntimeError(
        "Hardware camera/robot wiring is pending; run the mock adapter tests now "
        "and finish Task 1 of the hardware bring-up plan before serving."
    )


def main() -> None:
    args = parser().parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
