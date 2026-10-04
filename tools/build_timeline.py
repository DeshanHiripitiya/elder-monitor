"""Collapse smoothed states into duration-filtered timeline segments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.frame_sampler import load_config

PRESERVE_BRIEF_STANDING = {"SITTING_ON_BED", "LYING_IN_BED"}


def format_time(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def make_segments(
    states: pd.Series,
    times: pd.Series,
    confidences: pd.Series,
) -> list[dict[str, Any]]:
    if len(states) == 0:
        return []
    segments: list[dict[str, Any]] = []
    start = 0
    values = states.tolist()
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            end = index
            segments.append(
                {
                    "start": float(times.iloc[start]),
                    "end": float(times.iloc[end]) if end < len(times) else float(times.iloc[-1]),
                    "state": values[start],
                    "mean_confidence": float(confidences.iloc[start:end].mean()),
                    "_start_index": start,
                    "_end_index": end,
                }
            )
            start = index

    if len(times) > 1:
        step = float(np.median(np.diff(times.to_numpy(dtype=float))))
        segments[-1]["end"] += step
    return segments


def merge_short_segments(
    segments: list[dict[str, Any]],
    min_segment_sec: float,
) -> list[dict[str, Any]]:
    """Merge short runs, preserving brief standing between bed transitions."""
    changed = True
    while changed and len(segments) > 1:
        changed = False
        for index, segment in enumerate(segments):
            duration = segment["end"] - segment["start"]
            if duration >= min_segment_sec:
                continue
            if (
                segment["state"] == "STANDING"
                and 0 < index < len(segments) - 1
                and segments[index - 1]["state"] in PRESERVE_BRIEF_STANDING
                and segments[index + 1]["state"] in PRESERVE_BRIEF_STANDING
            ):
                continue

            neighbours = []
            if index > 0:
                neighbours.append((segments[index - 1]["end"] - segments[index - 1]["start"], index - 1))
            if index < len(segments) - 1:
                neighbours.append((segments[index + 1]["end"] - segments[index + 1]["start"], index + 1))
            target_index = max(neighbours)[1]
            target = segments[target_index]
            target_duration = target["end"] - target["start"]
            combined_duration = target_duration + duration
            target_confidence = (
                target["mean_confidence"] * target_duration
                + segment["mean_confidence"] * duration
            ) / max(combined_duration, 1e-9)
            target["start"] = min(target["start"], segment["start"])
            target["end"] = max(target["end"], segment["end"])
            target["mean_confidence"] = target_confidence
            segments.pop(index)
            changed = True
            break
    return segments


def collapse_adjacent_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    collapsed: list[dict[str, Any]] = []
    for segment in segments:
        if collapsed and collapsed[-1]["state"] == segment["state"]:
            previous = collapsed[-1]
            previous_duration = previous["end"] - previous["start"]
            duration = segment["end"] - segment["start"]
            previous["end"] = segment["end"]
            previous["mean_confidence"] = (
                previous["mean_confidence"] * previous_duration
                + segment["mean_confidence"] * duration
            ) / max(previous_duration + duration, 1e-9)
        else:
            collapsed.append(segment)
    return collapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--states", type=Path, default=Path("data/smoothed_states.parquet"))
    parser.add_argument("--scores", type=Path, default=Path("data/state_scores.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/timeline.parquet"))
    parser.add_argument("--text-output", type=Path, default=Path("data/timeline.txt"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = args.output.resolve()
    text_path = args.text_output.resolve()
    if (output_path.exists() or text_path.exists()) and not args.overwrite:
        raise FileExistsError("Timeline outputs exist; use --overwrite to regenerate them")

    config_path = args.config.resolve()
    config = load_config(config_path)
    states_path = args.states if args.states.is_absolute() else config_path.parent / args.states
    scores_path = args.scores if args.scores.is_absolute() else config_path.parent / args.scores
    smoothed = pd.read_parquet(states_path).sort_values("frame_idx").reset_index(drop=True)
    scores = pd.read_parquet(scores_path).sort_values("frame_idx").reset_index(drop=True)
    if not smoothed["frame_idx"].equals(scores["frame_idx"]):
        raise ValueError("Smoothed states and scores do not contain matching frame indices")
    video_path = Path(config["video"])
    if not video_path.is_absolute():
        video_path = config_path.parent / video_path
    capture = cv2.VideoCapture(str(video_path))
    video_fps = capture.get(cv2.CAP_PROP_FPS)
    video_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()
    video_duration = video_frames / video_fps if video_fps > 0 else float(smoothed["t"].max())

    confidences = np.array(
        [
            scores.loc[index, state]
            for index, state in enumerate(smoothed["smoothed_state"])
        ],
        dtype=float,
    )
    segments = make_segments(
        smoothed["smoothed_state"],
        smoothed["t"],
        pd.Series(confidences),
    )
    segments = merge_short_segments(
        segments,
        float(config["min_segment_sec"]),
    )
    segments = collapse_adjacent_segments(segments)
    segments[-1]["end"] = video_duration
    for segment in segments:
        segment.pop("_start_index", None)
        segment.pop("_end_index", None)

    timeline = pd.DataFrame(segments)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    timeline.to_parquet(output_path, index=False)
    lines = [
        f"{format_time(row.start)} – {format_time(row.end)} "
        f"{row.state} (confidence={row.mean_confidence:.2f})"
        for row in timeline.itertuples()
    ]
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    totals = timeline.assign(duration=timeline["end"] - timeline["start"]).groupby("state")["duration"].sum()
    total_duration = float(timeline["end"].max() - timeline["start"].min())
    print(f"Saved {len(timeline)} segments to {output_path}")
    print(f"Saved timeline text to {text_path}")
    print("Timeline:")
    print("\n".join(lines))
    print("Totals (seconds):", {state: round(value, 2) for state, value in totals.items()})
    print(f"Total covered duration: {total_duration:.2f}s")
    print(f"Duration sum: {float(totals.sum()):.2f}s")


if __name__ == "__main__":
    main()
