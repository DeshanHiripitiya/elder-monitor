"""Create bounded, explainable agent cases from timeline and pose evidence."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _segment_features(
    features: pd.DataFrame,
    start: float,
    end: float,
) -> pd.DataFrame:
    return features[(features["t"] >= start) & (features["t"] < end)]


def _candidate_exit_windows(
    segments: list[dict[str, Any]],
    events: list[dict[str, Any]],
    in_bed_states: set[str],
) -> list[dict[str, Any]]:
    exits = [event for event in events if event.get("type") == "bed_exit"]
    returns = [event for event in events if event.get("type") == "bed_return"]
    windows: list[dict[str, Any]] = []
    last_in_bed_end: float | None = None
    candidate_start: float | None = None
    phase = "IN_BED"
    return_after_exit: float | None = None
    ordered_segments = sorted(segments, key=lambda item: float(item["start"]))

    for segment in ordered_segments:
        start = float(segment["start"])
        end = float(segment["end"])
        state = str(segment["state"])

        if phase == "OUT":
            if return_after_exit is None or return_after_exit > start:
                continue
            phase = "IN_BED"
            last_in_bed_end = return_after_exit
            return_after_exit = None

        if state in in_bed_states:
            if candidate_start is not None:
                windows.append(
                    {
                        "start": candidate_start,
                        "end": start,
                        "resolution": "cancelled",
                    }
                )
            candidate_start = None
            last_in_bed_end = end
            continue

        if (
            candidate_start is None
            and last_in_bed_end is not None
            and state in {"STANDING", "WALKING", "OUT_OF_BED"}
        ):
            candidate_start = last_in_bed_end
            matching_exit = next(
                (
                    event
                    for event in exits
                    if float(event["start_time"]) <= end
                    and float(event["confirmed_time"]) >= candidate_start
                ),
                None,
            )
            if matching_exit is not None:
                windows.append(
                    {
                        "start": candidate_start,
                        "end": float(matching_exit["confirmed_time"]),
                        "resolution": "confirmed",
                    }
                )
                candidate_start = None
                phase = "OUT"
                next_return = next(
                    (
                        event
                        for event in returns
                        if float(event["confirmed_time"])
                        >= float(matching_exit["confirmed_time"])
                    ),
                    None,
                )
                return_after_exit = (
                    float(next_return["confirmed_time"])
                    if next_return is not None
                    else None
                )

    if candidate_start is not None:
        windows.append(
            {
                "start": candidate_start,
                "end": float(ordered_segments[-1]["end"]),
                "resolution": "pending",
            }
        )
    return windows


def build_cases(
    segments: list[dict[str, Any]],
    features: pd.DataFrame,
    events: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build cases by combining coincident trigger evidence into time windows."""
    agent_config = config["agent"]
    confidence_threshold = float(agent_config["trigger_conf"])
    short_seconds = float(agent_config["trigger_short_sec"])
    bed_floor_low, bed_floor_high = map(
        float, agent_config["bed_floor_conflict"]
    )
    triggers: list[dict[str, Any]] = []

    for index, segment in enumerate(
        sorted(segments, key=lambda item: float(item["start"]))
    ):
        start = float(segment["start"])
        end = float(segment["end"])
        state = str(segment["state"])
        confidence = float(segment["mean_confidence"])
        duration = end - start
        context = {"segment_index": index, "state": state}

        if confidence < confidence_threshold:
            triggers.append(
                {
                    **context,
                    "kind": "low_confidence",
                    "start": start,
                    "end": end,
                    "evidence": f"mean_confidence={confidence:.3f} < {confidence_threshold:.3f}",
                }
            )
        if duration < short_seconds:
            triggers.append(
                {
                    **context,
                    "kind": "short_segment",
                    "start": start,
                    "end": end,
                    "evidence": f"duration={duration:.3f}s < {short_seconds:.3f}s",
                }
            )
        if state == "UNKNOWN":
            triggers.append(
                {
                    **context,
                    "kind": "unknown_state",
                    "start": start,
                    "end": end,
                    "evidence": "timeline state is UNKNOWN",
                }
            )
        if state in {"LYING_IN_BED", "LYING_ON_FLOOR"}:
            frame_features = _segment_features(features, start, end)
            bed_fraction = frame_features["kp_in_bed_frac"].dropna()
            if not bed_fraction.empty:
                median_fraction = float(bed_fraction.median())
                if bed_floor_low <= median_fraction <= bed_floor_high:
                    triggers.append(
                        {
                            **context,
                            "kind": "bed_floor_conflict",
                            "start": start,
                            "end": end,
                            "evidence": (
                                f"median kp_in_bed_frac={median_fraction:.3f} "
                                f"in [{bed_floor_low:.3f}, {bed_floor_high:.3f}]"
                            ),
                        }
                    )

    for window in _candidate_exit_windows(
        segments,
        events,
        set(config["in_bed_states"]),
    ):
        triggers.append(
            {
                "kind": "candidate_bed_exit",
                "start": window["start"],
                "end": window["end"],
                "evidence": f"candidate exit {window['resolution']}",
            }
        )

    low_exit_threshold = float(config["alerts"]["exit_low_conf"])
    for event in events:
        if (
            event.get("type") == "bed_exit"
            and float(event["confidence"]) < low_exit_threshold
        ):
            triggers.append(
                {
                    "kind": "low_confidence_exit",
                    "start": float(event["start_time"]),
                    "end": float(event["confirmed_time"]),
                    "evidence": (
                        f"exit confidence={float(event['confidence']):.3f} "
                        f"< alerts.exit_low_conf={low_exit_threshold:.3f}"
                    ),
                }
            )

    if not triggers:
        return []

    triggers.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    grouped: list[dict[str, Any]] = []
    for trigger in triggers:
        start = float(trigger["start"])
        end = float(trigger["end"])
        if grouped and start <= grouped[-1]["end"]:
            grouped[-1]["end"] = max(grouped[-1]["end"], end)
            grouped[-1]["triggers"].append(trigger)
        else:
            grouped.append(
                {"start": start, "end": end, "triggers": [trigger]}
            )

    return [
        {
            "case_id": f"case-{index:03d}",
            "start": group["start"],
            "end": group["end"],
            "duration_sec": group["end"] - group["start"],
            "triggers": group["triggers"],
        }
        for index, group in enumerate(grouped, start=1)
    ]
