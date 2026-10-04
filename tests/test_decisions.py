import pandas as pd

from src.decisions import decide_bed_exit, evaluate_decision, run_alerts
from tools.score_states import STATES, score_row


ALERTS = {
    "edge_sit_monitor_sec": 300,
    "unknown_monitor_sec": 30,
    "exit_low_conf": 0.6,
    "out_of_bed_alert_sec": {"day": 1800, "night": 900},
    "out_of_view_alert_sec": {"day": 900, "night": 600},
    "floor_lying_alert_sec": 10,
    "night_window": ["22:00", "06:00"],
    "video_start_clock": "02:00",
}


def segment(state, start, end, confidence=0.9):
    return {
        "state": state,
        "start": float(start),
        "end": float(end),
        "mean_confidence": confidence,
    }


def config():
    return {
        "thresholds": {
            "lying_angle_deg": 60,
            "walk_speed": 0.25,
            "unknown_score": 0.3,
        },
        "state_scoring": {
            "upright_angle_deg": 45,
            "low_hip_knee_ratio": 0.16,
            "high_hip_knee_ratio": 0.24,
            "high_bed_fraction": 0.5,
            "floor_bed_fraction_max": 0.25,
            "floor_torso_angle_min_deg": 60,
            "low_speed": 0.25,
            "visibility_unknown": 0.45,
        },
    }


def test_floor_state_requires_horizontal_torso_and_low_bed_fraction():
    row = pd.Series(
        {
            "present": True,
            "torso_angle": 85.0,
            "bbox_aspect": 1.5,
            "kp_in_bed_frac": 0.1,
            "hip_in_bed": 0.0,
            "hip_knee_ratio": 0.2,
            "speed": 0.0,
            "vis": 0.9,
        }
    )

    scores = score_row(row, config())

    assert STATES[scores.argmax()] == "LYING_ON_FLOOR"


def test_floor_state_rejects_non_horizontal_torso():
    row = pd.Series(
        {
            "present": True,
            "torso_angle": 45.0,
            "bbox_aspect": 1.5,
            "kp_in_bed_frac": 0.1,
            "hip_in_bed": 0.0,
            "hip_knee_ratio": 0.2,
            "speed": 0.0,
            "vis": 0.9,
        }
    )

    scores = score_row(row, config())

    assert STATES[scores.argmax()] != "LYING_ON_FLOOR"


def test_out_of_bed_duration_alert_has_highest_priority():
    events = [
        {
            "type": "bed_exit",
            "start_time": 0.0,
            "confirmed_time": 0.0,
            "confidence": 0.9,
        }
    ]

    decision = evaluate_decision(
        [segment("WALKING", 0, 2000)],
        events,
        ALERTS,
        video_duration=2000,
    )

    assert decision["decision"] == "ALERT"
    assert decision["rule"] == "out_of_bed_duration"


def test_out_of_view_threshold_alerts_before_out_of_bed_threshold():
    events = [
        {
            "type": "bed_exit",
            "start_time": 0.0,
            "confirmed_time": 0.0,
            "confidence": 0.9,
            "note": "via_out_of_view",
        }
    ]

    decision = evaluate_decision(
        [segment("OUT_OF_BED", 0, 601)],
        events,
        ALERTS,
        video_duration=601,
    )

    assert decision["decision"] == "ALERT"
    assert decision["rule"] == "out_of_view_duration"


def test_floor_lying_alerts_after_threshold():
    decision = evaluate_decision(
        [segment("LYING_ON_FLOOR", 0, 11)],
        [],
        ALERTS,
        video_duration=11,
    )

    assert decision["decision"] == "ALERT"
    assert decision["rule"] == "floor_lying"


def test_confirmed_night_exit_is_monitor_with_night_note():
    event = {
        "type": "bed_exit",
        "start_time": 5.0,
        "confirmed_time": 9.0,
        "confidence": 0.8,
    }

    decision = evaluate_decision(
        [segment("WALKING", 0, 20)],
        [event],
        ALERTS,
        video_duration=20,
    )

    assert decision["decision"] == "MONITOR"
    assert decision["note"] == "night_exit"


def test_returned_night_exit_keeps_its_event_monitor_decision():
    event = {
        "type": "bed_exit",
        "start_time": 5.0,
        "confirmed_time": 9.0,
        "confidence": 0.8,
    }

    decision = decide_bed_exit(event, ALERTS)

    assert decision["decision"] == "MONITOR"
    assert decision["note"] == "night_exit"


def test_low_confidence_exit_is_monitor():
    event = {
        "type": "bed_exit",
        "start_time": 5.0,
        "confirmed_time": 9.0,
        "confidence": 0.4,
    }

    decision = evaluate_decision(
        [segment("WALKING", 0, 20)],
        [event],
        ALERTS,
        video_duration=20,
    )

    assert decision["decision"] == "MONITOR"
    assert decision["rule"] == "low_confidence_exit"


def test_prolonged_bed_sitting_is_monitor_with_limitation():
    decision = evaluate_decision(
        [segment("SITTING_ON_BED", 0, 301)],
        [],
        ALERTS,
        video_duration=301,
    )

    assert decision["decision"] == "MONITOR"
    assert "not distinguished" in decision["limitation"]


def test_continuous_unknown_is_monitor():
    decision = evaluate_decision(
        [segment("UNKNOWN", 0, 31)],
        [],
        ALERTS,
        video_duration=31,
    )

    assert decision["decision"] == "MONITOR"
    assert decision["rule"] == "continuous_unknown"


def test_no_matching_condition_is_normal():
    decision = evaluate_decision(
        [segment("LYING_IN_BED", 0, 100)],
        [],
        ALERTS,
        video_duration=100,
    )

    assert decision["decision"] == "NORMAL"


def test_returned_exit_is_no_longer_an_active_monitor_condition():
    events = [
        {
            "type": "bed_exit",
            "start_time": 10.0,
            "confirmed_time": 15.0,
            "confidence": 0.9,
        },
        {
            "type": "bed_return",
            "start_time": 30.0,
            "confirmed_time": 40.0,
            "confidence": 0.9,
        },
    ]

    decision = evaluate_decision(
        [segment("LYING_IN_BED", 40, 50)],
        events,
        ALERTS,
        video_duration=50,
    )

    assert decision["decision"] == "NORMAL"


def test_out_of_bed_alert_is_timestamped_at_threshold_not_return():
    alerts = {**ALERTS, "video_start_clock": "22:00"}
    events = [
        {
            "type": "bed_exit",
            "start_time": 0.0,
            "confirmed_time": 100.0,
            "confidence": 0.9,
        },
        {
            "type": "bed_return",
            "start_time": 1901.0,
            "confirmed_time": 1905.0,
            "confidence": 0.9,
        },
    ]
    timeline = run_alerts(
        [segment("WALKING", 100, 1905)],
        events,
        alerts,
        video_duration=2000,
    )

    alert = next(item for item in timeline if item["rule"] == "out_of_bed_duration")
    assert alert["decision"] == "ALERT"
    assert alert["t"] == 1000.0
    assert alert["t"] < events[1]["confirmed_time"]


def test_out_of_view_alert_is_timestamped_at_its_shorter_threshold():
    alerts = {**ALERTS, "video_start_clock": "22:00"}
    events = [
        {
            "type": "bed_exit",
            "start_time": 0.0,
            "confirmed_time": 0.0,
            "confidence": 0.9,
            "note": "via_out_of_view",
        }
    ]
    timeline = run_alerts(
        [segment("OUT_OF_BED", 0, 700)],
        events,
        alerts,
        video_duration=700,
    )

    alert = next(item for item in timeline if item["rule"] == "out_of_view_duration")
    assert alert["t"] == 600.0


def test_threshold_uses_night_clock_when_it_changes_during_exit():
    alerts = {**ALERTS, "video_start_clock": "21:50"}
    events = [
        {
            "type": "bed_exit",
            "start_time": 0.0,
            "confirmed_time": 0.0,
            "confidence": 0.9,
        }
    ]

    timeline = run_alerts(
        [segment("WALKING", 0, 1200)],
        events,
        alerts,
        video_duration=1200,
    )

    alert = next(item for item in timeline if item["rule"] == "out_of_bed_duration")
    assert alert["t"] == 900.0
    assert "night threshold" in alert["reason"]


def test_floor_alert_appears_at_start_plus_threshold():
    timeline = run_alerts(
        [segment("LYING_ON_FLOOR", 20, 50)],
        [],
        ALERTS,
        video_duration=50,
    )

    alert = next(item for item in timeline if item["rule"] == "floor_lying")
    assert alert["t"] == 30.0


def test_bed_sitting_monitor_is_timestamped_when_threshold_is_reached():
    timeline = run_alerts(
        [segment("SITTING_ON_BED", 5, 400)],
        [],
        ALERTS,
        video_duration=400,
    )

    monitor = next(
        item for item in timeline if item["rule"] == "prolonged_bed_sitting"
    )
    assert monitor["t"] == 305.0


def test_alert_timeline_is_chronological():
    timeline = run_alerts(
        [
            segment("UNKNOWN", 0, 100),
            segment("LYING_ON_FLOOR", 110, 140),
        ],
        [],
        ALERTS,
        video_duration=140,
    )

    assert [item["t"] for item in timeline] == sorted(
        item["t"] for item in timeline
    )


def test_highest_priority_rule_wins_at_the_same_time():
    timeline = run_alerts(
        [
            segment("LYING_ON_FLOOR", 0, 20),
            segment("UNKNOWN", 0, 40),
        ],
        [],
        ALERTS,
        video_duration=40,
    )

    same_time = [item for item in timeline if item["t"] == 10.0]
    assert len(same_time) == 1
    assert same_time[0]["rule"] == "floor_lying"
