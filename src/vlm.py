"""Local Ollama VLM client and strict response validation."""

from __future__ import annotations

import base64
import json
import math
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.agent_tools import VLMFrame


class VLMUnavailableError(RuntimeError):
    """The configured local VLM could not complete a request."""


class OllamaVLMClient:
    """Send ordered image frames to a local Ollama vision model."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5vl:7b",
        timeout_sec: float = 300,
        temperature: float = 0,
        context_length: int = 32768,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("Ollama base_url must use HTTP or HTTPS")
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        if temperature != 0:
            raise ValueError("The constrained VLM client requires temperature=0")
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        self.base_url = base_url.rstrip("/")
        self._model = model
        self.timeout_sec = float(timeout_sec)
        self.context_length = int(context_length)

    @property
    def model(self) -> str:
        return (
            f"ollama:{self.base_url}:{self._model}:temperature=0:"
            f"num_ctx={self.context_length}"
        )

    def describe(self, frames: list[VLMFrame], prompt: str) -> str:
        payload = {
            "model": self._model,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "num_ctx": self.context_length,
            },
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [
                        base64.b64encode(frame.jpeg_bytes).decode("ascii")
                        for frame in frames
                    ],
                }
            ],
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_sec) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace").strip()
            if len(details) > 1000:
                details = f"{details[:1000]}…"
            message = f"Ollama returned HTTP {error.code}"
            if details:
                message = f"{message}: {details}"
            raise VLMUnavailableError(message) from error
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
            raise VLMUnavailableError(
                f"Ollama request failed: {error}"
            ) from error

        try:
            content = response_data["message"]["content"]
        except (KeyError, TypeError) as error:
            raise VLMUnavailableError(
                "Ollama response did not contain message.content"
            ) from error
        if not isinstance(content, str):
            raise VLMUnavailableError("Ollama message.content was not text")
        return content


def parse_and_validate_vlm_json(raw: str) -> dict[str, Any]:
    """Parse and validate the constrained patient description response."""
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"Response was not valid JSON: {error.msg}") from error
    if not isinstance(result, dict):
        raise ValueError("Response must be a JSON object")

    required = {
        "patient_visible",
        "location",
        "posture",
        "other_person_present",
        "confidence",
        "evidence",
    }
    if set(result) != required:
        raise ValueError(f"Response keys must be exactly {sorted(required)}")
    if type(result["patient_visible"]) is not bool:
        raise ValueError("patient_visible must be a boolean")
    if type(result["other_person_present"]) is not bool:
        raise ValueError("other_person_present must be a boolean")
    if not isinstance(result["location"], str) or result["location"] not in {
        "on_bed",
        "beside_bed",
        "floor",
        "chair",
        "elsewhere",
        "not_visible",
    }:
        raise ValueError("location is not an allowed value")
    if not isinstance(result["posture"], str) or result["posture"] not in {
        "lying",
        "sitting",
        "standing",
        "walking",
        "unknown",
    }:
        raise ValueError("posture is not an allowed value")
    confidence = result["confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0 <= float(confidence) <= 1
    ):
        raise ValueError("confidence must be a finite number from 0 to 1")
    if not isinstance(result["evidence"], str) or not result["evidence"].strip():
        raise ValueError("evidence must be a non-empty string")
    return {
        **result,
        "confidence": float(confidence),
        "evidence": result["evidence"].strip(),
    }


def unknown_vlm_result() -> dict[str, Any]:
    """Return a safe, schema-shaped answer when the VLM cannot be used."""
    return {
        "patient_visible": False,
        "location": "not_visible",
        "posture": "unknown",
        "other_person_present": False,
        "confidence": 0.0,
        "evidence": "No validated VLM description is available.",
    }
