"""Detect bed exits from the smoothed timeline and normalized bed distance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.events import detect_exits
from src.decisions import (
    decide_bed_exit,
    decision_level,
    evaluate_decision,
    run_alerts,
)
from src.frame_sampler import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--segments", type=Path, default=Path("data/processed/timeline.parquet"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/features.parquet"))
    parser.add_argument("--events", type=Path, default=Path("data/processed/events.json"))
    parser.add_argument("--summary", type=Path, default=Path("data/processed/summary.json"))
    parser.add_argument("--alerts", type=Path, default=Path("data/processed/alerts.json"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    segments_path = args.segments if args.segments.is_absolute() else config_path.parent / args.segments
    features_path = args.features if args.features.is_absolute() else config_path.parent / args.features
    events_path = args.events if args.events.is_absolute() else config_path.parent / args.events
    summary_path = args.summary if args.summary.is_absolute() else config_path.parent / args.summary
    alerts_path = args.alerts if args.alerts.is_absolute() else config_path.parent / args.alerts
    if (
        events_path.exists() or summary_path.exists() or alerts_path.exists()
    ) and not args.overwrite:
        raise FileExistsError("Event outputs exist; use --overwrite to regenerate them")

    segments = pd.read_parquet(segments_path).to_dict("records")
    features = pd.read_parquet(features_path)
    events = detect_exits(segments, features, config)
    for event in events:
        if event["type"] == "bed_exit":
            event_decision = decide_bed_exit(event, config["alerts"])
            event.update(
                {
                    "decision": event_decision["decision"],
                    "reason": event_decision["reason"],
                    "rule": event_decision["rule"],
                    "decision_note": event_decision["note"],
                }
            )
        elif event["type"] == "bed_return":
            event.update(
                {
                    "decision": "NORMAL",
                    "reason": "bed_return confirmed; person is back in bed",
                    "rule": "confirmed_bed_return",
                }
            )
    video_path = Path(config["video"])
    if not video_path.is_absolute():
        video_path = config_path.parent / video_path
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()
    video_duration = frame_count / fps if fps > 0 else float(segments[-1]["end"])
    durations: dict[str, float] = {}
    for segment in segments:
        state = str(segment["state"])
        durations[state] = durations.get(state, 0.0) + (
            float(segment["end"]) - float(segment["start"])
        )
    in_bed_states = set(config["in_bed_states"])
    time_in_bed = sum(
        duration for state, duration in durations.items() if state in in_bed_states
    )
    time_out_of_bed = sum(
        duration for state, duration in durations.items() if state not in in_bed_states
    )
    exit_times = [float(event["confirmed_time"]) for event in events if event["type"] == "bed_exit"]
    return_times = [float(event["confirmed_time"]) for event in events if event["type"] == "bed_return"]
    out_periods = []
    for index, exit_time in enumerate(exit_times):
        next_return = next((value for value in return_times if value >= exit_time), video_duration)
        out_periods.append(next_return - exit_time)
    duration_sum = sum(durations.values())
    if abs(duration_sum - video_duration) >= 1.0:
        raise ValueError(
            f"Duration sum mismatch: {duration_sum:.3f}s vs video {video_duration:.3f}s"
        )
    final_state = str(segments[-1]["state"])
    summary: dict[str, Any] = {
        "event_counts": {
            "bed_exit": sum(event["type"] == "bed_exit" for event in events),
            "bed_return": sum(event["type"] == "bed_return" for event in events),
        },
        "events": events,
        "final_decision": evaluate_decision(
            segments,
            events,
            config["alerts"],
            video_duration=video_duration,
        ),
        "decision_timeline": run_alerts(
            segments,
            events,
            config["alerts"],
            video_duration=video_duration,
        ),
        "duration_summary": {
            "duration_by_state_sec": durations,
            "time_in_bed_sec": time_in_bed,
            "time_out_of_bed_sec": time_out_of_bed,
            "longest_out_of_bed_period_sec": max(out_periods, default=0.0),
            "final_state": final_state,
            "bed_exit_count": sum(event["type"] == "bed_exit" for event in events),
            "bed_return_count": sum(event["type"] == "bed_return" for event in events),
            "video_duration_sec": video_duration,
            "duration_sum_sec": duration_sum,
            "duration_sum_check_passed": abs(duration_sum - video_duration) < 1.0,
        },
        "configuration": config["events"],
    }
    summary["alerts"] = [
        {
            "time": decision["t"],
            "level": decision["decision"],
            "rule": decision["rule"],
            "reason": decision["reason"],
        }
        for decision in summary["decision_timeline"]
        if decision["decision"] == "ALERT"
    ]
    summary["overall_decision"] = decision_level(
        summary["decision_timeline"] + events
    )
    events_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    alerts_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text(json.dumps(events, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    alerts_path.write_text(json.dumps(summary["alerts"], indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(events)} events to {events_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved {len(summary['alerts'])} alerts to {alerts_path}")


if __name__ == "__main__":
    main()
