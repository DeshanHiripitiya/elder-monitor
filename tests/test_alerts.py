from src.decisions import decision_level, run_alerts


ALERTS = {
    "edge_sit_monitor_sec": 300,
    "unknown_monitor_sec": 30,
    "exit_low_conf": 0.6,
    "out_of_bed_alert_sec": {"day": 1800, "night": 900},
    "out_of_view_alert_sec": {"day": 900, "night": 600},
    "floor_lying_alert_sec": 10,
    "night_window": ["22:00", "06:00"],
    "video_start_clock": "22:00",
}


def segment(state, start, end):
    return {"state": state, "start": float(start), "end": float(end)}


def event(kind, confirmed, confidence=0.9, note=None):
    item = {
        "type": kind,
        "start_time": float(confirmed - 5),
        "confirmed_time": float(confirmed),
        "confidence": confidence,
    }
    if note:
        item["note"] = note
    return item


def evaluate(segments, events, *, duration, video_start_clock="22:00"):
    alerts = {**ALERTS, "video_start_clock": video_start_clock}
    timeline = run_alerts(segments, events, alerts, video_duration=duration)
    return timeline


def test_exit_and_return_before_threshold_is_monitor_without_alert():
    events = [
        event("bed_exit", 10),
        {
            "type": "bed_return",
            "start_time": 100.0,
            "confirmed_time": 110.0,
            "confidence": 0.9,
        },
    ]

    timeline = evaluate(
        [segment("LYING_IN_BED", 0, 10), segment("WALKING", 10, 100)],
        events,
        duration=120,
    )

    assert [item["decision"] for item in timeline] == ["MONITOR"]
    assert not any(item["decision"] == "ALERT" for item in timeline)
    assert decision_level(timeline) == "MONITOR"


def test_exit_without_return_longer_than_night_threshold_alerts():
    timeline = evaluate(
        [segment("WALKING", 0, 950)],
        [event("bed_exit", 0)],
        duration=950,
    )

    alert = next(item for item in timeline if item["decision"] == "ALERT")
    assert alert["rule"] == "out_of_bed_duration"
    assert alert["t"] == 900.0
    assert decision_level(timeline) == "ALERT"


def test_same_duration_during_day_is_monitor_not_alert():
    timeline = evaluate(
        [segment("WALKING", 0, 950)],
        [event("bed_exit", 0)],
        duration=950,
        video_start_clock="10:00",
    )

    assert not any(item["decision"] == "ALERT" for item in timeline)
    assert any(item["decision"] == "MONITOR" for item in timeline)
    assert decision_level(timeline) == "MONITOR"


def test_exit_then_out_of_view_longer_than_night_threshold_alerts():
    timeline = evaluate(
        [segment("OUT_OF_BED", 0, 650)],
        [event("bed_exit", 0, note="via_out_of_view")],
        duration=650,
    )

    alert = next(item for item in timeline if item["decision"] == "ALERT")
    assert alert["rule"] == "out_of_view_duration"
    assert alert["t"] == 600.0


def test_edge_sitting_ten_minutes_is_monitor():
    timeline = evaluate(
        [segment("SITTING_ON_BED", 0, 600)],
        [],
        duration=600,
    )

    monitor = next(item for item in timeline if item["rule"] == "prolonged_bed_sitting")
    assert monitor["decision"] == "MONITOR"
    assert monitor["t"] == 300.0


def test_unknown_for_forty_five_seconds_is_monitor():
    timeline = evaluate(
        [segment("UNKNOWN", 0, 45)],
        [],
        duration=45,
    )

    monitor = next(item for item in timeline if item["rule"] == "continuous_unknown")
    assert monitor["decision"] == "MONITOR"
    assert monitor["t"] == 30.0


def test_low_confidence_exit_is_monitor():
    timeline = evaluate(
        [segment("WALKING", 0, 100)],
        [event("bed_exit", 10, confidence=0.4)],
        duration=100,
    )

    monitor = next(item for item in timeline if item["rule"] == "low_confidence_exit")
    assert monitor["decision"] == "MONITOR"


def test_horizontal_off_bed_for_fifteen_seconds_alerts():
    timeline = evaluate(
        [segment("LYING_ON_FLOOR", 0, 15)],
        [],
        duration=15,
    )

    alert = next(item for item in timeline if item["rule"] == "floor_lying")
    assert alert["decision"] == "ALERT"
    assert alert["t"] == 10.0


def test_horizontal_on_bed_for_one_hour_is_normal():
    timeline = evaluate(
        [segment("LYING_IN_BED", 0, 3600)],
        [],
        duration=3600,
    )

    assert timeline == []
    assert decision_level(timeline) == "NORMAL"
