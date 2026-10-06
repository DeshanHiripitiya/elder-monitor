"""Gemini VLM client and strict response validation."""

from __future__ import annotations

import base64
import json
import math
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from src.agent_tools import VLMFrame


class VLMUnavailableError(RuntimeError):
    """The configured VLM could not complete a request."""


class GeminiVLMClient:
    """Send ordered image frames to the Gemini API."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key_env: str = "GEMINI_API_KEY",
        timeout_sec: float = 120,
        temperature: float = 0,
        allow_cloud_upload: bool = False,
    ) -> None:
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        if temperature != 0:
            raise ValueError("The constrained VLM client requires temperature=0")
        if not api_key_env:
            raise ValueError("api_key_env must not be empty")
        self._model = model
        self.api_key_env = api_key_env
        self.timeout_sec = float(timeout_sec)
        self.allow_cloud_upload = allow_cloud_upload

    @property
    def model(self) -> str:
        return f"gemini:{self._model}:temperature=0"

    def describe(self, frames: list[VLMFrame], prompt: str) -> str:
        if not self.allow_cloud_upload:
            raise VLMUnavailableError(
                "Gemini image upload is disabled; explicitly allow cloud upload "
                "before sending patient frames."
            )
        if not frames:
            raise ValueError("At least one image frame is required")
        env_path = Path(__file__).resolve().parents[1] / ".env"
        load_dotenv(dotenv_path=env_path, override=False)
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key or api_key.lower() in {
            "your_gemini_api_key_here",
            "paste_your_gemini_api_key_here",
            "replace_with_your_gemini_api_key",
        }:
            raise VLMUnavailableError(
                f"Set a real Gemini API key in {env_path} or the "
                f"{self.api_key_env} environment variable."
            )

        properties = {
            "patient_visible": {"type": "BOOLEAN"},
            "location": {
                "type": "STRING",
                "enum": [
                    "on_bed",
                    "beside_bed",
                    "floor",
                    "chair",
                    "elsewhere",
                    "not_visible",
                ],
            },
            "posture": {
                "type": "STRING",
                "enum": ["lying", "sitting", "standing", "walking", "unknown"],
            },
            "other_person_present": {"type": "BOOLEAN"},
            "confidence": {"type": "NUMBER"},
            "evidence": {"type": "STRING"},
        }
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        *[
                            {
                                "inlineData": {
                                    "mimeType": "image/jpeg",
                                    "data": base64.b64encode(frame.jpeg_bytes).decode(
                                        "ascii"
                                    ),
                                }
                            }
                            for frame in frames
                        ],
                        {"text": prompt},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": properties,
                    "required": list(properties),
                    "propertyOrdering": list(properties),
                },
            },
        }
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:generateContent"
        )
        request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_sec) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace").strip()
            if len(details) > 1000:
                details = f"{details[:1000]}…"
            message = f"Gemini returned HTTP {error.code}"
            if details:
                message = f"{message}: {details}"
            raise VLMUnavailableError(message) from error
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
            raise VLMUnavailableError(
                f"Gemini request failed: {error}"
            ) from error

        try:
            content = response_data["candidates"][0]["content"]["parts"]
        except (IndexError, KeyError, TypeError) as error:
            raise VLMUnavailableError(
                "Gemini response did not contain candidate content"
            ) from error
        if not isinstance(content, list):
            raise VLMUnavailableError("Gemini candidate content was not a list")
        text = "".join(
            part["text"]
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
        if not text:
            raise VLMUnavailableError("Gemini candidate did not contain text")
        return text


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
