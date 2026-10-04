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

    cues = ["WEBVTT", ""]
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
