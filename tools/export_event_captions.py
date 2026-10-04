"""Export event labels as a WebVTT sidecar for the debug video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def vtt_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def event_text(event: dict[str, Any]) -> str:
    confidence = event.get("confidence")
    if isinstance(confidence, (int, float)):
        confidence_text = f"{float(confidence):.2f}"
    else:
        confidence_text = str(confidence or "n/a")
    lines = [
        f"EVENT: {event['type']}",
        f"{event['previous_state']} -> {event['current_state']}",
        f"confidence: {confidence_text}",
    ]
    if event.get("note"):
        lines.append(f"note: {event['note']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=Path("data/events.json"))
    parser.add_argument("--summary", type=Path, default=Path("data/summary.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/debug_overlay_events.vtt"),
    )
    parser.add_argument(
        "--post-confirmation-sec",
        type=float,
        default=5.0,
        help="How long an event label remains visible after confirmation.",
    )
    args = parser.parse_args()

    events = json.loads(args.events.read_text(encoding="utf-8"))
    if not isinstance(events, list):
        raise ValueError("Events JSON must contain a list")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    duration_summary = summary.get("duration_summary", {})
    duration_by_state = duration_summary.get("duration_by_state_sec", {})
    duration_lines = [
        "DURATION SUMMARY",
        f"in bed: {duration_summary.get('time_in_bed_sec', 0.0):.1f}s",
        f"out of bed: {duration_summary.get('time_out_of_bed_sec', 0.0):.1f}s",
        (
            "longest out period: "
            f"{duration_summary.get('longest_out_of_bed_period_sec', 0.0):.1f}s"
        ),
        f"final state: {duration_summary.get('final_state', 'UNKNOWN')}",
        (
            f"exits: {duration_summary.get('bed_exit_count', 0)}  "
            f"returns: {duration_summary.get('bed_return_count', 0)}"
        ),
        (
            "duration check: "
            f"{'PASS' if duration_summary.get('duration_sum_check_passed') else 'FAIL'}"
        ),
    ]
    if duration_by_state:
        duration_lines.append(
            "states: "
            + ", ".join(
                f"{state}={float(seconds):.1f}s"
                for state, seconds in duration_by_state.items()
            )
        )

    cues = ["WEBVTT", ""]
    cues.extend(
        [
            "summary",
            f"00:00:00.000 --> 00:00:10.000",
            "\n".join(duration_lines),
            "",
        ]
    )
    for index, event in enumerate(events, start=1):
        start = float(event["start_time"])
        confirmed = float(event["confirmed_time"])
        end = max(confirmed + args.post_confirmation_sec, start + 0.1)
        cues.extend(
            [
                str(index),
                f"{vtt_timestamp(start)} --> {vtt_timestamp(end)}",
                event_text(event),
                "",
            ]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(cues), encoding="utf-8")
    print(f"Saved {len(events)} event captions to {args.output}")


if __name__ == "__main__":
    main()
