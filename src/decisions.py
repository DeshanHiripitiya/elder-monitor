"""Priority-ordered, explainable caregiver decision rules."""

from __future__ import annotations

from typing import Any


def _clock_minutes(value: str) -> int:
    hour_text, minute_text = value.split(":", maxsplit=1)
    hour, minute = int(hour_text), int(minute_text)
    if not 0 <= hour < 24 or not 0 <= minute < 60:
        raise ValueError(f"Invalid local clock time: {value}")
    return hour * 60 + minute


def is_night(timestamp: float, alerts: dict[str, Any]) -> bool:
    start_clock = _clock_minutes(str(alerts["video_start_clock"]))
    current_clock = (start_clock + int(timestamp // 60)) % (24 * 60)
    night_start, night_end = (
        _clock_minutes(str(value)) for value in alerts["night_window"]
    )
    if night_start == night_end:
        return True
    if night_start < night_end:
        return night_start <= current_clock < night_end
    return current_clock >= night_start or current_clock < night_end


def _decision(level: str, rule: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {"decision": level, "rule": rule, "reason": reason, **extra}


def decision_level(decisions: list[dict[str, Any]]) -> str:
    """Return the highest severity represented by decision records."""
    rank = {"NORMAL": 0, "MONITOR": 1, "ALERT": 2}
    return max(
        (str(item.get("decision", "NORMAL")) for item in decisions),
        key=lambda level: rank.get(level, -1),
        default="NORMAL",
    )


def decide_bed_exit(event: dict[str, Any], alerts: dict[str, Any]) -> dict[str, Any]:
    """Assign the default MONITOR outcome to a confirmed exit event."""
    confidence = float(event.get("confidence", 0.0))
    if confidence < float(alerts["exit_low_conf"]):
        reason = (
            f"bed_exit confidence {confidence:.2f} < "
            f"{float(alerts['exit_low_conf']):.2f} threshold"
        )
        rule = "low_confidence_exit"
    else:
        reason = f"confirmed bed_exit at {float(event['confirmed_time']):.0f}s"
        rule = "confirmed_bed_exit"
    note = (
        "night_exit"
        if is_night(float(event["confirmed_time"]), alerts)
        else "confirmed_exit"
    )
    return _decision("MONITOR", rule, reason, note=note)


def run_alerts(
    segments: list[dict[str, Any]],
    events: list[dict[str, Any]],
    alerts: dict[str, Any],
    *,
    video_duration: float,
) -> list[dict[str, Any]]:
    """Evaluate event and sustained-state rules in timeline order.

    Threshold-based records are timestamped at the first second when the
    configured threshold is reached. Iterating at one-second resolution also
    correctly handles a day/night threshold change during an ongoing episode.
    """
    decisions: list[dict[str, Any]] = []
    exits = sorted(
        (event for event in events if event.get("type") == "bed_exit"),
        key=lambda event: float(event["confirmed_time"]),
    )
    returns = sorted(
        (event for event in events if event.get("type") == "bed_return"),
        key=lambda event: float(event["confirmed_time"]),
    )
    used_returns: set[int] = set()

    def add(
        timestamp: float,
        priority: int,
        level: str,
        rule: str,
        reason: str,
        **extra: Any,
    ) -> None:
        decisions.append(
            {
                "decision": level,
                "t": round(float(timestamp), 3),
                "rule": rule,
                "reason": reason,
                "_priority": priority,
                **extra,
            }
        )

    for event in exits:
        confirmed = float(event["confirmed_time"])
        matched_return = next(
            (
                (index, return_event)
                for index, return_event in enumerate(returns)
                if index not in used_returns
                and float(return_event["confirmed_time"]) >= confirmed
            ),
            None,
        )
        interval_end = (
            float(matched_return[1]["confirmed_time"])
            if matched_return is not None
            else float(video_duration)
        )
        if matched_return is not None:
            used_returns.add(matched_return[0])

        exit_decision = decide_bed_exit(event, alerts)
        add(
            confirmed,
            4 if exit_decision["rule"] == "confirmed_bed_exit" else 5,
            exit_decision["decision"],
            exit_decision["rule"],
            exit_decision["reason"],
            note=exit_decision["note"],
        )

        elapsed_second = int(confirmed) + 1
        while elapsed_second <= interval_end:
            elapsed = elapsed_second - confirmed
            period = "night" if is_night(elapsed_second, alerts) else "day"
            out_limit = float(alerts["out_of_bed_alert_sec"][period])
            if elapsed >= out_limit:
                add(
                    confirmed + out_limit,
                    1,
                    "ALERT",
                    "out_of_bed_duration",
                    f"out_of_bed duration reached {out_limit:.0f}s {period} threshold",
                )
                break

            if event.get("note") == "via_out_of_view":
                view_limit = float(alerts["out_of_view_alert_sec"][period])
                if elapsed >= view_limit:
                    add(
                        confirmed + view_limit,
                        2,
                        "ALERT",
                        "out_of_view_duration",
                        f"out_of_view duration reached {view_limit:.0f}s {period} threshold",
                    )
                    break
            elapsed_second += 1

    for segment in sorted(segments, key=lambda item: float(item["start"])):
        start = float(segment["start"])
        end = min(float(segment["end"]), float(video_duration))
        if end <= start:
            continue
        state = str(segment["state"])
        if state == "LYING_ON_FLOOR":
            threshold = float(alerts["floor_lying_alert_sec"])
            if end - start >= threshold:
                add(
                    start + threshold,
                    3,
                    "ALERT",
                    "floor_lying",
                    f"lying_on_floor duration reached {threshold:.0f}s threshold",
                )
        elif state == "SITTING_ON_BED":
            threshold = float(alerts["edge_sit_monitor_sec"])
            if end - start >= threshold:
                add(
                    start + threshold,
                    6,
                    "MONITOR",
                    "prolonged_bed_sitting",
                    f"sitting_on_bed duration reached {threshold:.0f}s monitor threshold",
                    limitation=(
                        "Edge sitting is approximated by all SITTING_ON_BED; "
                        "sitting up in bed is not distinguished."
                    ),
                )
        elif state == "UNKNOWN":
            threshold = float(alerts["unknown_monitor_sec"])
            if end - start >= threshold:
                add(
                    start + threshold,
                    7,
                    "MONITOR",
                    "continuous_unknown",
                    f"UNKNOWN duration reached {threshold:.0f}s monitor threshold",
                )

    decisions.sort(key=lambda decision: (float(decision["t"]), decision["_priority"]))
    timeline: list[dict[str, Any]] = []
    for decision in decisions:
        if timeline and float(timeline[-1]["t"]) == float(decision["t"]):
            continue
        decision.pop("_priority")
        timeline.append(decision)
    return timeline


def evaluate_decision(
    segments: list[dict[str, Any]],
    events: list[dict[str, Any]],
    alerts: dict[str, Any],
    *,
    video_duration: float,
) -> dict[str, Any]:
    """Return the first matching NORMAL/MONITOR/ALERT rule for video end."""
    if not segments:
        return _decision("MONITOR", "unknown", "No timeline segments are available.")

    ordered = sorted(segments, key=lambda segment: float(segment["start"]))
    end_time = float(video_duration)
    final_segment = next(
        (
            segment
            for segment in reversed(ordered)
            if float(segment["start"]) <= end_time
            and float(segment["end"]) >= end_time
        ),
        ordered[-1],
    )
    final_state = str(final_segment["state"])
    final_state_duration = max(
        0.0,
        min(end_time, float(final_segment["end"]))
        - max(float(final_segment["start"]), 0.0),
    )
    exits = sorted(
        (event for event in events if event.get("type") == "bed_exit"),
        key=lambda event: float(event["confirmed_time"]),
    )
    returns = sorted(
        (event for event in events if event.get("type") == "bed_return"),
        key=lambda event: float(event["confirmed_time"]),
    )
    active_exit = next(
        (
            event
            for event in reversed(exits)
            if not any(
                float(return_event["confirmed_time"])
                >= float(event["confirmed_time"])
                for return_event in returns
            )
        ),
        None,
    )
    night_now = is_night(end_time, alerts)
    period = "night" if night_now else "day"

    if active_exit is not None:
        elapsed = max(0.0, end_time - float(active_exit["confirmed_time"]))
        out_of_bed_limit = float(alerts["out_of_bed_alert_sec"][period])
        if elapsed > out_of_bed_limit:
            return _decision(
                "ALERT",
                "out_of_bed_duration",
                f"out_of_bed {elapsed:.0f}s > {out_of_bed_limit:.0f}s {period} threshold",
                event=active_exit,
            )

        if active_exit.get("note") == "via_out_of_view":
            out_of_view_limit = float(alerts["out_of_view_alert_sec"][period])
            if elapsed > out_of_view_limit:
                return _decision(
                    "ALERT",
                    "out_of_view_duration",
                    f"out_of_view {elapsed:.0f}s > {out_of_view_limit:.0f}s {period} threshold",
                    event=active_exit,
                )

    if (
        final_state == "LYING_ON_FLOOR"
        and final_state_duration > float(alerts["floor_lying_alert_sec"])
    ):
        limit = float(alerts["floor_lying_alert_sec"])
        return _decision(
            "ALERT",
            "floor_lying",
            f"lying_on_floor {final_state_duration:.0f}s > {limit:.0f}s threshold",
        )

    if active_exit is not None:
        return {
            **decide_bed_exit(active_exit, alerts),
            "event": active_exit,
        }

    if (
        final_state == "SITTING_ON_BED"
        and final_state_duration > float(alerts["edge_sit_monitor_sec"])
    ):
        limit = float(alerts["edge_sit_monitor_sec"])
        return _decision(
            "MONITOR",
            "prolonged_bed_sitting",
            f"sitting_on_bed {final_state_duration:.0f}s > {limit:.0f}s monitor threshold",
            limitation=(
                "Edge sitting is approximated by all SITTING_ON_BED; "
                "sitting up in bed is not distinguished."
            ),
        )

    if (
        final_state == "UNKNOWN"
        and final_state_duration > float(alerts["unknown_monitor_sec"])
    ):
        limit = float(alerts["unknown_monitor_sec"])
        return _decision(
            "MONITOR",
            "continuous_unknown",
            f"UNKNOWN {final_state_duration:.0f}s > {limit:.0f}s monitor threshold",
        )

    return _decision("NORMAL", "no_rule_matched", "No configured alert or monitor rule matched.")
