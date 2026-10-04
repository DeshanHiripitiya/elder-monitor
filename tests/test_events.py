import pandas as pd

from src.events import detect_exits
from tools.build_timeline import make_contiguous


CONFIG = {
    "in_bed_states": ["LYING_IN_BED", "SITTING_ON_BED"],
    "events": {
        "exit_away_sec": 4,
        "exit_away_dist": 0.8,
        "return_approach_dist": 0.8,
        "unknown_hold_sec": 4,
        "return_sit_sec": 2,
        "return_lie_sec": 5,
        "out_of_view_exit_sec": 10,
    },
}


def make_segments(states, confidence=0.9):
    """Create one-second hand-made segments; no video is needed."""
    return [
        {
            "start": float(index),
            "end": float(index + 1),
            "state": state,
            "mean_confidence": confidence,
        }
        for index, state in enumerate(states)
    ]


def make_features(states, distances=None, present=None):
    distances = distances if distances is not None else [0.0] * len(states)
    present = present if present is not None else [True] * len(states)
    return pd.DataFrame(
        {
            "t": [float(index) for index in range(len(states))],
            "dist_to_bed": distances,
            "present": present,
        }
    )


def event_types(states, distances=None, present=None):
    result = detect_exits(
        make_segments(states),
        make_features(states, distances, present),
        CONFIG,
    )
    return result, [event["type"] for event in result]


def test_sit_up_only_does_not_create_exit():
    states = ["LYING_IN_BED"] * 5 + ["SITTING_ON_BED"] * 60 + ["LYING_IN_BED"] * 5
    result, types = event_types(states)

    assert types == []
    assert result == []


def test_edge_sitting_for_sixty_seconds_does_not_create_exit():
    states = ["LYING_IN_BED"] * 5 + ["SITTING_ON_BED"] * 60

    result, types = event_types(states)

    assert types == []
    assert result == []


def test_brief_three_second_stand_is_cancelled():
    states = ["LYING_IN_BED"] * 5 + ["STANDING"] * 3 + ["SITTING_ON_BED"] * 5

    result, types = event_types(states)

    assert types == []
    assert result == []


def test_clean_exit_creates_one_exit():
    states = (
        ["LYING_IN_BED"] * 5
        + ["SITTING_ON_BED"] * 2
        + ["STANDING"] * 4
        + ["WALKING"] * 6
    )
    distances = [0.0] * 7 + [1.0] * 10

    result, types = event_types(states, distances)

    assert types == ["bed_exit"]
    assert result[0]["previous_state"] == "SITTING_ON_BED"
    assert 0.0 < result[0]["confidence"] <= 0.9


def test_exit_and_return_creates_one_exit_and_one_return():
    states = (
        ["LYING_IN_BED"] * 5
        + ["SITTING_ON_BED"] * 2
        + ["STANDING"] * 4
        + ["WALKING"] * 8
        + ["SITTING_ON_BED"] * 2
        + ["LYING_IN_BED"] * 5
    )
    distances = [0.0] * 7 + [1.0] * 12 + [0.5] * 7

    result, types = event_types(states, distances)

    assert types == ["bed_exit", "bed_return"]
    assert result[1]["previous_state"] == "OUT"
    assert result[1]["current_state"] == "LYING_IN_BED"


def test_out_of_view_for_thirty_seconds_creates_low_confidence_exit():
    states = ["LYING_IN_BED"] * 5 + ["OUT_OF_BED"] * 30
    present = [True] * 5 + [False] * 30

    result, types = event_types(states, present=present)

    assert types == ["bed_exit"]
    assert result[0]["note"] == "via_out_of_view"
    assert result[0]["confidence"] < 0.9


def test_unknown_two_second_blip_does_not_create_exit():
    states = ["LYING_IN_BED"] * 5 + ["UNKNOWN"] * 2 + ["LYING_IN_BED"] * 5

    result, types = event_types(states)

    assert types == []
    assert result == []


def test_fake_return_followed_by_walking_does_not_create_return():
    states = (
        ["LYING_IN_BED"] * 5
        + ["STANDING"] * 5
        + ["WALKING"] * 6
        + ["SITTING_ON_BED"]
        + ["WALKING"] * 5
    )
    distances = [0.0] * 5 + [1.0] * 11 + [0.5] * 6

    result, types = event_types(states, distances)

    assert types == ["bed_exit"]


def test_contiguous_timeline_durations_sum_to_total():
    segments = [
        {"start": 1.0, "end": 3.0, "state": "LYING_IN_BED", "mean_confidence": 0.9},
        {"start": 4.0, "end": 7.0, "state": "WALKING", "mean_confidence": 0.8},
    ]

    normalized = make_contiguous(segments, video_duration=10.0)
    duration_sum = sum(item["end"] - item["start"] for item in normalized)

    assert normalized[0]["start"] == 0.0
    assert normalized[-1]["end"] == 10.0
    assert abs(duration_sum - 10.0) < 1.0
    assert normalized[0]["state"] == "UNKNOWN"
    assert normalized[2]["state"] == "UNKNOWN"
