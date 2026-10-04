"""Detect bed exits from the smoothed timeline and normalized bed distance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.events import detect_exits
from src.frame_sampler import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--segments", type=Path, default=Path("data/timeline.parquet"))
    parser.add_argument("--features", type=Path, default=Path("data/features.parquet"))
    parser.add_argument("--events", type=Path, default=Path("data/events.json"))
    parser.add_argument("--summary", type=Path, default=Path("data/summary.json"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    segments_path = args.segments if args.segments.is_absolute() else config_path.parent / args.segments
    features_path = args.features if args.features.is_absolute() else config_path.parent / args.features
    events_path = args.events if args.events.is_absolute() else config_path.parent / args.events
    summary_path = args.summary if args.summary.is_absolute() else config_path.parent / args.summary
    if (events_path.exists() or summary_path.exists()) and not args.overwrite:
        raise FileExistsError("Event outputs exist; use --overwrite to regenerate them")

    segments = pd.read_parquet(segments_path).to_dict("records")
    features = pd.read_parquet(features_path)
    events = detect_exits(segments, features, config)
    summary: dict[str, Any] = {
        "event_counts": {
            "bed_exit": sum(event["type"] == "bed_exit" for event in events),
            "bed_return": sum(event["type"] == "bed_return" for event in events),
        },
        "events": events,
        "configuration": config["events"],
    }
    events_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text(json.dumps(events, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(events)} events to {events_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
