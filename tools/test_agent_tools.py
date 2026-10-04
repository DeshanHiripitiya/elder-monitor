"""Exercise agent tools against cached artifacts from the configured video."""

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

from src.agent_tools import (
    MockVLM,
    _video_duration,
    extend_window,
    get_pose_features,
    get_state_history,
    vlm_describe_clip,
)
from src.frame_sampler import load_config


def _time_seconds(value: str) -> float:
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return float(minutes * 60 + seconds)
    if len(parts) == 3 and parts[2] == 0:
        minutes, seconds, _trailing_zero = parts
        return float(minutes * 60 + seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours * 3600 + minutes * 60 + seconds)
    raise ValueError(f"Unsupported ground-truth time value: {value}")


def _ground_truth_window(path: Path, t0: float, t1: float) -> list[dict[str, Any]]:
    ground_truth = pd.read_csv(path)
    result = []
    for row in ground_truth.itertuples(index=False):
        start = _time_seconds(str(row.start))
        end = _time_seconds(str(row.end))
        if end > t0 and start < t1:
            result.append(
                {
                    "start": max(t0, start),
                    "end": min(t1, end),
                    "state": str(row.state),
                }
            )
    return result


def _state_agreement(
    predicted: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
) -> dict[str, float | None]:
    compared = 0.0
    matched = 0.0
    for prediction in predicted:
        for reference in ground_truth:
            overlap = max(
                0.0,
                min(float(prediction["end"]), float(reference["end"]))
                - max(float(prediction["start"]), float(reference["start"])),
            )
            compared += overlap
            if prediction["state"] == reference["state"]:
                matched += overlap
    return {
        "matched_sec": matched,
        "compared_sec": compared,
        "agreement_frac": matched / compared if compared else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--start", type=float, default=120.0)
    parser.add_argument("--end", type=float, default=145.0)
    parser.add_argument(
        "--mock-vlm",
        action="store_true",
        help="Use MockVLM; no external model or API call is made",
    )
    args = parser.parse_args()
    if not args.mock_vlm:
        parser.error(
            "A provider client is not configured yet. Pass --mock-vlm for the "
            "real-video offline smoke test."
        )

    config_path = args.config.resolve()
    config = load_config(config_path)
    video_path = Path(config["video"])
    if not video_path.is_absolute():
        video_path = config_path.parent / video_path
    duration = _video_duration(video_path)
    if args.start < 0 or args.end <= args.start or args.end > duration:
        parser.error(f"Require 0 <= start < end <= {duration:.3f} seconds")

    print(f"Video: {video_path}")
    print(f"Test window: {args.start:.1f}-{args.end:.1f}s")
    ground_truth = _ground_truth_window(
        config_path.parent / "data" / "gt.csv",
        args.start,
        args.end,
    )
    print("Ground-truth labels:")
    for segment in ground_truth:
        print(
            f"  {segment['start']:.1f}-{segment['end']:.1f}s "
            f"{segment['state']}"
        )

    state_history = get_state_history(args.start, args.end, config_path=config_path)
    print("\nget_state_history:")
    print(json.dumps(state_history, indent=2))
    agreement = _state_agreement(state_history["segments"], ground_truth)
    print(
        "Ground-truth state agreement: "
        f"{agreement['matched_sec']:.1f}/{agreement['compared_sec']:.1f}s "
        f"({agreement['agreement_frac']:.1%})"
        if agreement["agreement_frac"] is not None
        else "Ground-truth state agreement: no labeled overlap"
    )
    print("\nget_pose_features:")
    print(
        json.dumps(
            get_pose_features(args.start, args.end, config_path=config_path),
            indent=2,
        )
    )
    print("\nextend_window:")
    print(
        "  backward:",
        extend_window(
            args.start,
            args.end,
            "backward",
            video_end=duration,
        ),
    )
    print(
        "  forward:",
        extend_window(
            args.start,
            args.end,
            "forward",
            video_end=duration,
        ),
    )

    print("\nvlm_describe_clip (mock client with real video frames):")
    description = vlm_describe_clip(
        args.start,
        args.end,
        "Describe whether the person appears on the bed or beside it.",
        client=MockVLM(),
        config=config,
        config_path=config_path,
    )
    print(json.dumps(description, indent=2))
    print("\nSmoke test complete. VLM output is mocked, not a real model assessment.")


if __name__ == "__main__":
    main()
