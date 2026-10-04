"""Run event detection on hand-annotated ground-truth state segments."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.events import detect_exits
from src.frame_sampler import load_config
from tools.score_states import parse_timestamp


def load_ground_truth_segments(path: Path) -> list[dict]:
    truth = pd.read_csv(path)
    required = {"start", "end", "state"}
    if required.issubset(truth.columns):
        return _state_segments(truth)
    event_required = {"event", "start_time", "confirmed_time"}
    if event_required.issubset(truth.columns):
        return _segments_from_event_annotations(read_event_annotations(path))
    raise ValueError(
        f"{path} must contain either state columns {sorted(required)} "
        f"or event columns {sorted(event_required)}"
    )


def _state_segments(truth: pd.DataFrame) -> list[dict]:
    segments = []
    for row in truth.itertuples(index=False):
        start = parse_timestamp(row.start)
        end = parse_timestamp(row.end)
        if end <= start:
            raise ValueError(f"Ground-truth interval ends before it starts: {row}")
        segments.append(
            {
                "start": start,
                "end": end,
                "state": str(row.state),
                "mean_confidence": 1.0,
            }
        )
    return sorted(segments, key=lambda segment: segment["start"])


def _segments_from_event_annotations(truth: pd.DataFrame) -> list[dict]:
    """Turn the project's event annotation CSV into test state segments."""
    segments: list[dict] = []
    cursor = 0.0
    for row in truth.sort_values("start_time").itertuples(index=False):
        start = parse_timestamp(row.start_time)
        end = parse_timestamp(row.confirmed_time)
        if end <= start:
            raise ValueError(f"Ground-truth event ends before it starts: {row}")
        if start > cursor:
            segments.append(
                {
                    "start": cursor,
                    "end": start,
                    "state": "LYING_IN_BED",
                    "mean_confidence": 1.0,
                }
            )
        note = str(getattr(row, "note", "")).lower()
        event = str(row.event)
        if event == "bed_exit":
            if "camera view" in note:
                segments.append(
                    {
                        "start": start,
                        "end": end,
                        "state": "OUT_OF_BED",
                        "mean_confidence": 1.0,
                        "_present": False,
                    }
                )
            else:
                sit_end = min(start + 2.0, end)
                segments.extend(
                    [
                        {
                            "start": start,
                            "end": sit_end,
                            "state": "SITTING_ON_BED",
                            "mean_confidence": 1.0,
                        },
                        {
                            "start": sit_end,
                            "end": end,
                            "state": "WALKING",
                            "mean_confidence": 1.0,
                        },
                    ]
                )
        elif event == "bed_return":
            sit_end = min(start + 2.0, end)
            segments.extend(
                [
                    {
                        "start": start,
                        "end": sit_end,
                        "state": "SITTING_ON_BED",
                        "mean_confidence": 1.0,
                    },
                    {
                        "start": sit_end,
                        "end": end,
                        "state": "LYING_IN_BED",
                        "mean_confidence": 1.0,
                    },
                ]
            )
        else:
            segments.append(
                {
                    "start": start,
                    "end": end,
                    "state": "SITTING_ON_BED",
                    "mean_confidence": 1.0,
                }
            )
        cursor = max(cursor, end)
    return segments


def read_event_annotations(path: Path) -> pd.DataFrame:
    """Read event CSVs whose free-text notes may contain unquoted commas."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows or rows[0][:3] != ["event", "start_time", "confirmed_time"]:
        raise ValueError(
            f"{path} must start with event,start_time,confirmed_time,note"
        )
    records = [
        {
            "event": row[0],
            "start_time": row[1],
            "confirmed_time": row[2],
            "note": ",".join(row[3:]),
        }
        for row in rows[1:]
        if len(row) >= 3
    ]
    return pd.DataFrame(records)


def features_from_segments(
    segments: list[dict],
    sample_fps: float,
) -> pd.DataFrame:
    end_time = max(float(segment["end"]) for segment in segments)
    timestamps = np.arange(0.0, end_time, 1.0 / sample_fps)
    rows = []
    for timestamp in timestamps:
        segment = next(
            (
                segment
                for segment in segments
                if float(segment["start"]) <= timestamp < float(segment["end"])
            ),
            None,
        )
        state = str(segment["state"]) if segment else "UNKNOWN"
        if state in {"LYING_IN_BED", "SITTING_ON_BED"}:
            distance = 0.0
        elif state in {"STANDING", "WALKING", "OUT_OF_BED"}:
            distance = 1.0
        else:
            distance = 0.5
        rows.append(
            {
                "t": timestamp,
                "dist_to_bed": distance,
                "present": bool(segment.get("_present", True)) if segment else False,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--gt-segments",
        type=Path,
        default=Path("data/gt_events.csv"),
        help="CSV with start,end,state ground-truth state segments.",
    )
    parser.add_argument("--expected-exits", type=int, default=2)
    parser.add_argument("--expected-returns", type=int, default=2)
    parser.add_argument("--sample-fps", type=float, default=None)
    args = parser.parse_args()

    config_path = args.config.resolve()
    gt_path = args.gt_segments if args.gt_segments.is_absolute() else config_path.parent / args.gt_segments
    if not gt_path.exists():
        raise FileNotFoundError(
            f"Ground-truth file not found: {gt_path}. "
            "Create it with start,end,state rows before running this comparison."
        )
    config = load_config(config_path)
    segments = load_ground_truth_segments(gt_path)
    sample_fps = args.sample_fps or float(config["sample_fps"])
    features = features_from_segments(segments, sample_fps)
    events = detect_exits(segments, features, config)
    exits = sum(event["type"] == "bed_exit" for event in events)
    returns = sum(event["type"] == "bed_return" for event in events)
    print(f"Ground-truth segments: {len(segments)}")
    print(f"Detected exits: {exits} (expected {args.expected_exits})")
    print(f"Detected returns: {returns} (expected {args.expected_returns})")
    for event in events:
        print(
            f"{event['type']}: {event['start_time']:.2f}s -> "
            f"{event['confirmed_time']:.2f}s"
        )
    if exits != args.expected_exits or returns != args.expected_returns:
        raise SystemExit("Ground-truth event comparison failed")
    print("Ground-truth event comparison passed")


if __name__ == "__main__":
    main()
