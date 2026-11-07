"""DeepSeek image + function-calling capability probe (no harness involved)."""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass, field

import httpx
from PIL import Image, ImageDraw


class ProbeError(RuntimeError):
    pass


@dataclass
class ProbeReport:
    total: int
    ok_calls: int
    tool_call_ok: int
    image_ok: int
    latencies_s: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _synthetic_image() -> str:
    image = Image.new("RGB", (96, 96), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((28, 28, 68, 68), fill="red")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _tool_schema() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_scene",
                "description": "Return the dominant color of the square in the image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "color": {"type": "string", "enum": ["red", "blue", "green"]}
                    },
                    "required": ["color"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def probe(
    api_key: str,
    *,
    base_url: str = "https://api.deepseek.com/v1",
    model: str = "deepseek-flash",
    n: int = 10,
    timeout_s: float = 120.0,
) -> ProbeReport:
    if not api_key:
        raise ProbeError("no API key supplied")
    if n <= 0:
        raise ValueError("n must be positive")

    image_b64 = _synthetic_image()
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a vision tool-use probe. Call get_scene when asked about colors.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Call get_scene with color='red' for the visible square.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            },
        ],
        "tools": _tool_schema(),
        "tool_choice": "auto",
        "max_tokens": 256,
    }
    report = ProbeReport(total=n, ok_calls=0, tool_call_ok=0, image_ok=0)
    with httpx.Client(timeout=timeout_s) as client:
        for _ in range(n):
            started = time.perf_counter()
            try:
                response = client.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
            except Exception as error:  # noqa: BLE001 - probe reports all failures
                report.errors.append(f"{type(error).__name__}: {error}")
                continue
            finally:
                report.latencies_s.append(time.perf_counter() - started)

            message = data["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                continue
            report.ok_calls += 1
            report.tool_call_ok += 1
            try:
                arguments = tool_calls[0]["function"]["arguments"]
                if "red" in arguments:
                    report.image_ok += 1
            except Exception:  # noqa: BLE001 - malformed tool arguments count as a failure
                report.errors.append("malformed tool-call arguments")
    return report
