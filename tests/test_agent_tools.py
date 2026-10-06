import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.agent_tools import (
    MockVLM,
    extend_window,
    get_pose_features,
    get_state_history,
    vlm_describe_clip,
)


def test_get_state_history_clips_segments_and_summarizes_bed_distance():
    timeline = pd.DataFrame(
        [
            {"start": 0, "end": 5, "state": "LYING_IN_BED", "mean_confidence": 0.9},
            {"start": 5, "end": 10, "state": "STANDING", "mean_confidence": 0.8},
        ]
    )
    features = pd.DataFrame({"t": [2, 4, 6, 8], "dist_to_bed": [0.1, 0.2, 1.0, 1.4]})

    result = get_state_history(
        3,
        7,
        timeline=timeline,
        features=features,
    )

    assert result == {
        "segments": [
            {"start": 3.0, "end": 5.0, "state": "LYING_IN_BED", "conf": 0.9},
            {"start": 5.0, "end": 7.0, "state": "STANDING", "conf": 0.8},
        ],
        "bed_dist": [0.2, 1.0],
    }


def test_get_pose_features_returns_small_summaries_only():
    features = pd.DataFrame(
        {
            "t": [0, 1, 2],
            "torso_angle": [10.0, 20.0, 30.0],
            "kp_in_bed_frac": [0.2, 0.4, 0.8],
            "dist_to_bed": [0.0, 0.3, 0.9],
            "speed": [0.1, 0.2, 0.3],
            "present": [True, True, False],
            "vis": [0.8, 0.6, np.nan],
        }
    )

    result = get_pose_features(0, 2, features=features)

    assert result == {
        "torso_angle": {"mean": 20.0, "min": 10.0, "max": 30.0},
        "kp_in_bed_frac": pytest.approx(0.4666666667),
        "dist_to_bed": {"start": 0.0, "end": 0.9},
        "speed_mean": pytest.approx(0.2),
        "present_frac": pytest.approx(2 / 3),
        "vis_mean": 0.7,
    }


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        ("backward", (20, 50)),
        ("forward", (30, 60)),
        ("both", (20, 60)),
    ],
)
def test_extend_window_respects_direction_and_video_bounds(direction, expected):
    assert extend_window(
        30,
        50,
        direction,
        seconds=10,
        video_end=60,
    ) == expected


def test_extend_window_caps_total_span_at_sixty_seconds():
    assert extend_window(
        50,
        100,
        "both",
        seconds=20,
        video_end=200,
    ) == (45.0, 105.0)


def test_extend_window_rejects_invalid_window_and_direction():
    with pytest.raises(ValueError, match="direction"):
        extend_window(0, 10, "sideways", video_end=100)
    with pytest.raises(ValueError, match="exceeds"):
        extend_window(0, 61, "both", video_end=100)


class CountingMockVLM(MockVLM):
    def __init__(self):
        self.calls = 0
        self.received_frames = []

    def describe(self, frames, prompt):
        self.calls += 1
        self.received_frames = frames
        return super().describe(frames, prompt)


def make_test_video(path: Path):
    width, height, fps, count = 160, 120, 2, 8
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MP4 encoder is unavailable")
    for index in range(count):
        frame = np.full((height, width, 3), index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return fps


def test_vlm_uses_annotated_boundary_frames_and_caches_json(tmp_path):
    video_path = tmp_path / "clip.mp4"
    fps = make_test_video(video_path)
    config = {
        "agent": {"vlm_frames": 3},
        "bed_polygon": [[10, 10], [100, 10], [100, 90], [10, 90]],
    }
    client = CountingMockVLM()
    cache = tmp_path / "cache"

    first = vlm_describe_clip(
        0,
        2,
        "Where is the person?",
        client=client,
        config=config,
        video_path=video_path,
        cache_dir=cache,
    )
    second = vlm_describe_clip(
        0,
        2,
        "Where is the person?",
        client=client,
        config=config,
        video_path=video_path,
        cache_dir=cache,
    )

    assert first == second
    assert client.calls == 1
    assert first["status"] == "ok"
    assert first["result"]["posture"] == "unknown"
    assert len(client.received_frames) == 5
    assert client.received_frames[0].timestamp_sec == 0
    assert client.received_frames[-1].timestamp_sec == 2
    assert all(frame.jpeg_bytes.startswith(b"\xff\xd8") for frame in client.received_frames)
    assert json.loads(next(cache.glob("*.json")).read_text()) == first
    assert fps == 2


def test_vlm_returns_unavailable_tool_result_without_raising(tmp_path):
    video_path = tmp_path / "clip.mp4"
    make_test_video(video_path)

    class OfflineVLM:
        model = "offline-test"

        def describe(self, frames, prompt):
            raise RuntimeError("Not used")

    class UnavailableVLM(OfflineVLM):
        def describe(self, frames, prompt):
            from src.vlm import VLMUnavailableError

            raise VLMUnavailableError("configured VLM service is unavailable")

    result = vlm_describe_clip(
        0,
        1,
        "Describe posture.",
        client=UnavailableVLM(),
        config={"agent": {"vlm_frames": 2}, "bed_polygon": []},
        video_path=video_path,
        cache_dir=tmp_path / "cache",
    )

    assert result["status"] == "vlm_unavailable"
    assert result["result"]["posture"] == "unknown"
    assert result["result"]["confidence"] == 0
