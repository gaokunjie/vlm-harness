#!/usr/bin/env python3
"""Probe DeepSeek-V4.1-Flash for image input + function calling."""

from __future__ import annotations

import argparse
import json
import os

from dobot_rua.probe import probe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com/v1")
    args = parser.parse_args()

    key = (
        os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("UNI_ROBOT_AGENT_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    )
    if not key:
        raise SystemExit(
            "Set DEEPSEEK_API_KEY, UNI_ROBOT_AGENT_API_KEY, or ANTHROPIC_AUTH_TOKEN."
        )
    report = probe(key, base_url=args.base_url, model=args.model, n=args.n)
    print(json.dumps(report.__dict__, indent=2))
    return 0 if report.ok_calls else 1


if __name__ == "__main__":
    raise SystemExit(main())
