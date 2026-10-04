"""Sample video frames at a configured rate while preserving source timestamps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import cv2
import yaml


def load_config(config_path: Path) -> dict:
    """Load the application configuration from YAML."""
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a mapping: {config_path}")
    return config


def sampled_frames(
    video_path: Path, sample_fps: float
) -> Iterator[tuple[int, float, object]]:
    """Yield ``(frame_idx, timestamp_sec, frame)`` at the requested sample rate."""
    if sample_fps <= 0:
        raise ValueError("sample_fps must be greater than zero")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    video_fps = capture.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        capture.release()
        raise RuntimeError(f"Video has an invalid FPS value: {video_fps}")

    frame_interval = max(1, round(video_fps / sample_fps))
    frame_idx = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_idx % frame_interval == 0:
                timestamp_sec = frame_idx / video_fps
                yield frame_idx, timestamp_sec, frame

            frame_idx += 1
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    video_path = Path(config["video"])
    if not video_path.is_absolute():
        video_path = config_path.parent / video_path

    sample_fps = float(config["sample_fps"])
    for frame_idx, timestamp_sec, _frame in sampled_frames(video_path, sample_fps):
        print(f"frame_idx={frame_idx} t={timestamp_sec:.6f}s")


if __name__ == "__main__":
    main()
