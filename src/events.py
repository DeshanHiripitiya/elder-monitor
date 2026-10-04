"""Segment-level event detection for the elder-monitor timeline."""

from __future__ import annotations

from typing import Any

import pandas as pd


def event_confidence(
    segments: list[dict[str, Any]],
    start_time: float,
    end_time: float,
) -> float:
    """Return an explainable minimum-segment confidence heuristic."""
    relevant = [
        segment
        for segment in segments
        if float(segment["end"]) > start_time
        and float(segment["start"]) < end_time
    ]
    if not relevant:
        return 0.0
    minimum = min(float(segment["mean_confidence"]) for segment in relevant)
    has_unknown = any(str(segment["state"]) == "UNKNOWN" for segment in relevant)
    return minimum * (0.8 if has_unknown else 1.0)


def detect_exits(
    segments: list[dict[str, Any]],
    features: pd.DataFrame,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Detect confirmed bed exits from smoothed segments and frame features."""
    event_config = config["events"]
    in_bed_states = set(config["in_bed_states"])
    away_seconds = float(event_config["exit_away_sec"])
    away_distance = float(event_config["exit_away_dist"])
    return_approach_distance = float(event_config["return_approach_dist"])
    unknown_hold = float(event_config["unknown_hold_sec"])
    out_of_view_seconds = float(event_config["out_of_view_exit_sec"])
    return_sit_seconds = float(event_config["return_sit_sec"])
    return_lie_seconds = float(event_config["return_lie_sec"])

    ordered = sorted(segments, key=lambda segment: float(segment["start"]))
    frames = features.sort_values("t").reset_index(drop=True)
    events: list[dict[str, Any]] = []
    phase = "IN_BED"
    last_in_bed_end: float | None = None
    previous_in_bed_state: str | None = None
    candidate_start: float | None = None

    if frames.empty or not ordered:
        return events
    step = (
        float(frames["t"].diff().dropna().median())
        if len(frames) > 1
        else 0.0
    )
    segment_index = 0
    unknown_duration = 0.0
    away_start: float | None = None
    out_of_view_start: float | None = None
    approach_start: float | None = None
    sit_start: float | None = None
    lie_start: float | None = None

    for row in frames.itertuples(index=False):
        row_time = float(row.t)
        while (
            segment_index + 1 < len(ordered)
            and row_time >= float(ordered[segment_index]["end"])
        ):
            segment_index += 1
        segment = ordered[segment_index]
        state = str(segment["state"])
        next_time = row_time + step
        distance = float(row.dist_to_bed) if pd.notna(row.dist_to_bed) else 0.0

        if phase == "OUT":
            if not bool(row.present):
                continue
            if state == "UNKNOWN":
                continue
            if distance > return_approach_distance:
                continue
            phase = "APPROACH"
            approach_start = row_time
            sit_start = None
            lie_start = None

        if phase in {"APPROACH", "SIT"}:
            if not bool(row.present):
                phase = "OUT"
                approach_start = None
                sit_start = None
                lie_start = None
                continue
            if state == "UNKNOWN":
                continue
            if distance > return_approach_distance:
                phase = "OUT"
                approach_start = None
                sit_start = None
                lie_start = None
                continue
            if state in {"STANDING", "WALKING"}:
                phase = "OUT"
                approach_start = None
                sit_start = None
                lie_start = None
                continue
            if phase == "APPROACH":
                if state == "SITTING_ON_BED":
                    sit_start = row_time if sit_start is None else sit_start
                    if next_time - sit_start >= return_sit_seconds:
                        phase = "SIT"
                continue
            if state == "SITTING_ON_BED":
                lie_start = None
                continue
            if state == "LYING_IN_BED":
                lie_start = row_time if lie_start is None else lie_start
                if next_time - lie_start >= return_lie_seconds:
                    events.append(
                        {
                            "type": "bed_return",
                            "start_time": approach_start,
                            "confirmed_time": row_time,
                            "previous_state": "OUT",
                            "current_state": "LYING_IN_BED",
                            "confidence": event_confidence(
                                ordered, float(approach_start), row_time
                            ),
                        }
                    )
                    phase = "IN_BED"
                    last_in_bed_end = next_time
                    previous_in_bed_state = "LYING_IN_BED"
                    approach_start = None
                    sit_start = None
                    lie_start = None
                continue

        if state in in_bed_states:
            phase = "IN_BED"
            last_in_bed_end = next_time
            previous_in_bed_state = state
            candidate_start = None
            unknown_duration = 0.0
            away_start = None
            out_of_view_start = None
            continue

        if phase == "IN_BED" and (
            state in {"STANDING", "WALKING"}
            or (state == "OUT_OF_BED" and not bool(row.present))
        ):
            phase = "CANDIDATE"
            candidate_start = (
                last_in_bed_end if last_in_bed_end is not None else row_time
            )

        if phase != "CANDIDATE" or candidate_start is None:
            continue

        if state == "UNKNOWN":
            unknown_duration += step
            if unknown_duration <= unknown_hold:
                continue
            away_start = None
            out_of_view_start = None
            continue

        if state in in_bed_states:
            phase = "IN_BED"
            candidate_start = None
            continue

        unknown_duration = 0.0
        if not bool(row.present):
            out_of_view_start = (
                row_time if out_of_view_start is None else out_of_view_start
            )
            away_start = None
            if next_time - out_of_view_start >= out_of_view_seconds:
                events.append(
                    {
                        "type": "bed_exit",
                        "start_time": candidate_start,
                        "confirmed_time": row_time,
                        "previous_state": previous_in_bed_state,
                        "current_state": state,
                        "confidence": event_confidence(
                            ordered,
                            float(candidate_start),
                            row_time,
                        ) * 0.8,
                        "note": "via_out_of_view",
                    }
                )
                phase = "OUT"
            continue

        out_of_view_start = None
        if distance >= away_distance:
            away_start = row_time if away_start is None else away_start
            if next_time - away_start >= away_seconds:
                events.append(
                    {
                        "type": "bed_exit",
                        "start_time": candidate_start,
                        "confirmed_time": row_time,
                        "previous_state": previous_in_bed_state,
                        "current_state": state,
                        "confidence": event_confidence(
                            ordered,
                            float(candidate_start),
                            row_time,
                        ),
                    }
                )
                phase = "OUT"
        else:
            away_start = None

    return events
