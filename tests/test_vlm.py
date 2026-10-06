import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.agent_tools import VLMFrame, vlm_describe_clip
from src.vlm import (
    GeminiVLMClient,
    VLMUnavailableError,
    parse_and_validate_vlm_json,
)


VALID_RESULT = {
    "patient_visible": True,
    "location": "on_bed",
    "posture": "lying",
    "other_person_present": True,
    "confidence": 0.87,
    "evidence": "The patient is lying beneath the blanket on the bed.",
}


def make_test_video(path: Path):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        2,
        (160, 120),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MP4 encoder is unavailable")
    for index in range(8):
        writer.write(np.full((120, 160, 3), index * 20, dtype=np.uint8))
    writer.release()


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        "[]",
        json.dumps({**VALID_RESULT, "posture": "running"}),
        json.dumps({**VALID_RESULT, "confidence": 2}),
        json.dumps({**VALID_RESULT, "patient_visible": "yes"}),
        json.dumps({key: value for key, value in VALID_RESULT.items() if key != "evidence"}),
    ],
)
def test_response_validator_rejects_invalid_content(payload):
    with pytest.raises(ValueError):
        parse_and_validate_vlm_json(payload)


def test_response_validator_returns_normalized_confidence():
    response = parse_and_validate_vlm_json(json.dumps(VALID_RESULT))

    assert response["confidence"] == 0.87
    assert response["posture"] == "lying"


def test_gemini_request_sends_ordered_images_and_structured_json():
    body = {
        "candidates": [
            {"content": {"parts": [{"text": json.dumps(VALID_RESULT)}]}}
        ]
    }
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(body).encode()
    frames = [VLMFrame(1.0, b"first"), VLMFrame(2.0, b"second")]
    client = GeminiVLMClient(
        api_key_env="TEST_GEMINI_API_KEY",
        allow_cloud_upload=True,
    )

    with (
        patch.dict("os.environ", {"TEST_GEMINI_API_KEY": "test-key"}),
        patch("src.vlm.urlopen", return_value=response) as request_call,
    ):
        raw = client.describe(frames, "Narrow question")

    request = request_call.call_args.args[0]
    request_payload = json.loads(request.data)
    parts = request_payload["contents"][0]["parts"]
    assert request.full_url == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.5-flash:generateContent"
    )
    assert request.get_header("X-goog-api-key") == "test-key"
    assert request_payload["generationConfig"]["temperature"] == 0
    assert request_payload["generationConfig"]["responseMimeType"] == "application/json"
    assert request_payload["generationConfig"]["responseSchema"]["required"] == [
        "patient_visible",
        "location",
        "posture",
        "other_person_present",
        "confidence",
        "evidence",
    ]
    assert [part["inlineData"]["data"] for part in parts[:-1]] == [
        "Zmlyc3Q=",
        "c2Vjb25k",
    ]
    assert all(part["inlineData"]["mimeType"] == "image/jpeg" for part in parts[:-1])
    assert parts[-1] == {"text": "Narrow question"}
    assert raw == json.dumps(VALID_RESULT)


def test_gemini_upload_requires_explicit_opt_in():
    client = GeminiVLMClient(api_key_env="TEST_GEMINI_API_KEY")

    with (
        patch.dict("os.environ", {"TEST_GEMINI_API_KEY": "test-key"}),
        patch("src.vlm.urlopen") as request_call,
        pytest.raises(VLMUnavailableError, match="upload is disabled"),
    ):
        client.describe([VLMFrame(0, b"image")], "Question")

    request_call.assert_not_called()


def test_gemini_missing_api_key_is_explicitly_unavailable():
    client = GeminiVLMClient(
        api_key_env="MISSING_TEST_GEMINI_API_KEY",
        allow_cloud_upload=True,
    )

    with (
        patch.dict("os.environ", {}, clear=True),
        patch("src.vlm.urlopen") as request_call,
        pytest.raises(VLMUnavailableError, match="Set a real Gemini API key"),
    ):
        client.describe([VLMFrame(0, b"image")], "Question")

    request_call.assert_not_called()


def test_gemini_rejects_placeholder_api_key():
    client = GeminiVLMClient(allow_cloud_upload=True)

    with (
        patch.dict(
            "os.environ",
            {"GEMINI_API_KEY": "your_gemini_api_key_here"},
            clear=True,
        ),
        patch("src.vlm.urlopen") as request_call,
        pytest.raises(VLMUnavailableError, match="Set a real Gemini API key"),
    ):
        client.describe([VLMFrame(0, b"image")], "Question")

    request_call.assert_not_called()


def test_gemini_http_errors_include_response_details():
    from io import BytesIO
    from urllib.error import HTTPError

    client = GeminiVLMClient(
        api_key_env="TEST_GEMINI_API_KEY",
        allow_cloud_upload=True,
    )
    response = HTTPError(
        "https://generativelanguage.googleapis.com/",
        429,
        "Too Many Requests",
        {},
        BytesIO(b'{"error":{"message":"quota exceeded"}}'),
    )

    with (
        patch.dict("os.environ", {"TEST_GEMINI_API_KEY": "test-key"}),
        patch("src.vlm.urlopen", side_effect=response),
        pytest.raises(VLMUnavailableError, match="quota exceeded"),
    ):
        client.describe([VLMFrame(0, b"image")], "Question")


def test_gemini_connection_errors_are_explicitly_unavailable():
    from urllib.error import URLError

    client = GeminiVLMClient(
        api_key_env="TEST_GEMINI_API_KEY",
        timeout_sec=0.1,
        allow_cloud_upload=True,
    )
    with (
        patch.dict("os.environ", {"TEST_GEMINI_API_KEY": "test-key"}),
        patch("src.vlm.urlopen", side_effect=URLError("connection refused")),
    ):
        with pytest.raises(VLMUnavailableError, match="request failed"):
            client.describe([VLMFrame(0, b"image")], "Question")


class SequenceVLM:
    model = "sequence-vlm"

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def describe(self, frames, prompt):
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_invalid_json_retries_once_then_returns_unknown(tmp_path):
    video = tmp_path / "clip.mp4"
    make_test_video(video)
    client = SequenceVLM(["invalid", "{still invalid"])

    result = vlm_describe_clip(
        0,
        1,
        "Describe posture.",
        client=client,
        config={"agent": {"vlm_frames": 2}, "bed_polygon": []},
        video_path=video,
        cache_dir=tmp_path / "cache",
    )

    assert len(client.prompts) == 2
    assert "failed JSON/schema validation" in client.prompts[1]
    assert "fixed camera, in time order, with timestamps" in client.prompts[0]
    assert "The red polygon is the bed" in client.prompts[0]
    assert "Describe ONLY the elderly patient" in client.prompts[0]
    assert "Ignore any other person" in client.prompts[0]
    assert "other_person_present" in client.prompts[0]
    assert "Do not guess" in client.prompts[0]
    assert result["status"] == "invalid_response"
    assert result["result"]["posture"] == "unknown"
    assert result["result"]["confidence"] == 0


def test_invalid_json_retry_can_recover(tmp_path):
    video = tmp_path / "clip.mp4"
    make_test_video(video)
    client = SequenceVLM(["bad response", json.dumps(VALID_RESULT)])

    result = vlm_describe_clip(
        0,
        1,
        "Describe posture.",
        client=client,
        config={"agent": {"vlm_frames": 2}, "bed_polygon": []},
        video_path=video,
        cache_dir=tmp_path / "cache",
    )

    assert result["status"] == "ok"
    assert result["result"]["posture"] == "lying"
    assert len(client.prompts) == 2
