import pandas as pd

from src.agent import build_cases


CONFIG = {
    "agent": {
        "trigger_conf": 0.6,
        "trigger_short_sec": 3,
        "bed_floor_conflict": [0.3, 0.7],
    },
    "alerts": {"exit_low_conf": 0.6},
    "in_bed_states": ["LYING_IN_BED", "SITTING_ON_BED"],
}


def segment(start, end, state, confidence=0.9):
    return {
        "start": float(start),
        "end": float(end),
        "state": state,
        "mean_confidence": confidence,
    }


def test_overlapping_segment_triggers_are_combined_into_one_case():
    segments = [
        segment(0, 2, "WALKING", confidence=0.4),
        segment(2, 6, "UNKNOWN"),
    ]

    cases = build_cases(segments, pd.DataFrame(), [], CONFIG)

    assert len(cases) == 1
    assert cases[0]["start"] == 0
    assert cases[0]["end"] == 6
    assert {
        trigger["kind"] for trigger in cases[0]["triggers"]
    } == {"low_confidence", "short_segment", "candidate_bed_exit", "unknown_state"}


def test_mixed_bed_floor_signal_uses_segment_median():
    segments = [segment(0, 4, "LYING_IN_BED")]
    features = pd.DataFrame(
        {"t": [0, 1, 2, 3], "kp_in_bed_frac": [0.2, 0.4, 0.6, 0.9]}
    )

    cases = build_cases(segments, features, [], CONFIG)

    assert [item["kind"] for item in cases[0]["triggers"]] == ["bed_floor_conflict"]


def test_low_confidence_exit_adds_trigger_to_candidate_case():
    segments = [
        segment(0, 5, "LYING_IN_BED"),
        segment(5, 9, "STANDING"),
        segment(9, 12, "WALKING"),
    ]
    events = [
        {
            "type": "bed_exit",
            "start_time": 5,
            "confirmed_time": 10,
            "confidence": 0.4,
        }
    ]

    cases = build_cases(
        segments,
        pd.DataFrame(columns=["t", "kp_in_bed_frac"]),
        events,
        CONFIG,
    )

    assert len(cases) == 1
    assert {
        trigger["kind"] for trigger in cases[0]["triggers"]
    } == {"candidate_bed_exit", "low_confidence_exit"}


def test_stable_segments_without_uncertainty_do_not_create_cases():
    segments = [segment(0, 5, "LYING_IN_BED")]
    features = pd.DataFrame({"t": [0, 1, 2, 3, 4], "kp_in_bed_frac": [0.9] * 5})

    assert build_cases(segments, features, [], CONFIG) == []
