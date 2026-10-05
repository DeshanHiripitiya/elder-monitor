import pandas as pd
import yaml
from pathlib import Path

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
        segment(0, 5, "LYING_IN_BED"),
        segment(5, 7, "WALKING", confidence=0.4),
        segment(7, 11, "UNKNOWN"),
    ]

    cases = build_cases(
        segments,
        pd.DataFrame(columns=["t", "kp_in_bed_frac"]),
        [],
        CONFIG,
    )

    assert len(cases) == 1
    assert cases[0]["start"] == 5
    assert cases[0]["end"] == 11
    assert {
        trigger["kind"] for trigger in cases[0]["triggers"]
    } == {
        "low_confidence",
        "short_segment",
        "candidate_bed_exit",
        "unknown_state",
    }


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


def test_starting_outside_bed_does_not_create_a_false_exit_candidate():
    segments = [
        segment(0, 2, "SITTING_OUTSIDE_BED"),
        segment(2, 8, "WALKING"),
    ]

    cases = build_cases(
        segments,
        pd.DataFrame(columns=["t", "kp_in_bed_frac"]),
        [],
        CONFIG,
    )

    assert all(
        trigger["kind"] != "candidate_bed_exit"
        for case in cases
        for trigger in case["triggers"]
    )


def test_each_configured_segment_trigger_is_reported_with_evidence():
    segments = [
        segment(0, 4, "LYING_IN_BED"),
        segment(4, 6, "WALKING", confidence=0.5),
        segment(6, 8, "UNKNOWN"),
    ]
    features = pd.DataFrame(
        {
            "t": [0, 1, 2, 3],
            "kp_in_bed_frac": [0.4, 0.5, 0.5, 0.6],
        }
    )

    cases = build_cases(segments, features, [], CONFIG)
    triggers = [
        trigger
        for case in cases
        for trigger in case["triggers"]
    ]

    assert {
        "low_confidence",
        "short_segment",
        "unknown_state",
        "bed_floor_conflict",
        "candidate_bed_exit",
    } <= {trigger["kind"] for trigger in triggers}
    assert all(trigger["evidence"] for trigger in triggers)


def test_low_confidence_exit_rule_uses_alert_config_threshold():
    segments = [
        segment(0, 5, "LYING_IN_BED"),
        segment(5, 10, "WALKING"),
    ]
    events = [
        {
            "type": "bed_exit",
            "start_time": 5,
            "confirmed_time": 9,
            "confidence": 0.59,
        }
    ]
    features = pd.DataFrame(columns=["t", "kp_in_bed_frac"])

    cases = build_cases(segments, features, events, CONFIG)

    assert any(
        trigger["kind"] == "low_confidence_exit"
        for case in cases
        for trigger in case["triggers"]
    )


def test_exit_at_confidence_threshold_does_not_trigger_uncertainty_case():
    segments = [
        segment(0, 5, "LYING_IN_BED"),
        segment(5, 10, "WALKING"),
    ]
    events = [
        {
            "type": "bed_exit",
            "start_time": 5,
            "confirmed_time": 9,
            "confidence": 0.6,
        }
    ]
    features = pd.DataFrame(columns=["t", "kp_in_bed_frac"])

    cases = build_cases(segments, features, events, CONFIG)

    assert not any(
        trigger["kind"] == "low_confidence_exit"
        for case in cases
        for trigger in case["triggers"]
    )


def test_production_and_demo_configs_define_all_trigger_and_budget_settings():
    project_root = Path(__file__).resolve().parents[1]
    expected = {
        "trigger_conf",
        "trigger_short_sec",
        "bed_floor_conflict",
        "max_steps_per_case",
        "max_vlm_calls_per_case",
        "max_vlm_calls_total",
        "vlm_frames",
        "vlm_provider",
        "vlm_base_url",
        "vlm_model",
        "vlm_timeout_sec",
        "vlm_temperature",
        "vlm_context_length",
    }

    for config_name in ("config.yaml", "config_demo.yaml"):
        config = yaml.safe_load((project_root / config_name).read_text(encoding="utf-8"))

        assert set(config["agent"]) == expected
