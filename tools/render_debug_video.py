"""Render the source video with state, bed, and frozen pose overlays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.frame_sampler import load_config

COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
COCO_SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


def resolve_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_path.parent / path


def draw_overlay(
    frame: np.ndarray,
    polygon: np.ndarray,
    state: str,
    confidence: float,
    timestamp: float,
    dist_to_bed: float | None,
    keypoints: np.ndarray | None,
    pose_timestamp: float | None,
    min_kp_conf: float,
    event: dict[str, Any] | None = None,
) -> np.ndarray:
    output = frame.copy()
    if len(polygon) >= 3:
        cv2.polylines(output, [polygon], isClosed=True, color=(0, 255, 255), thickness=3)
        overlay = output.copy()
        cv2.fillPoly(overlay, [polygon], color=(0, 180, 180))
        output = cv2.addWeighted(overlay, 0.12, output, 0.88, 0)

    if keypoints is not None:
        visible = [
            point
            for point in keypoints
            if len(point) >= 3 and np.isfinite(point[:2]).all()
        ]
        for start, end in COCO_SKELETON:
            if start >= len(keypoints) or end >= len(keypoints):
                continue
            first, second = keypoints[start], keypoints[end]
            if (
                len(first) >= 3
                and len(second) >= 3
                and first[2] >= min_kp_conf
                and second[2] >= min_kp_conf
            ):
                cv2.line(
                    output,
                    (int(first[0]), int(first[1])),
                    (int(second[0]), int(second[1])),
                    (255, 180, 0),
                    2,
                    cv2.LINE_AA,
                )
        for index, point in enumerate(keypoints[: len(COCO_KEYPOINT_NAMES)]):
            if len(point) < 3 or not np.isfinite(point[:2]).all():
                continue
            x, y, kp_conf = float(point[0]), float(point[1]), float(point[2])
            if kp_conf < min_kp_conf:
                color = (90, 90, 90)
            else:
                color = (
                    (0, 255, 0)
                    if kp_conf >= 0.7
                    else (0, 200, 255)
                )
            center = (int(x), int(y))
            cv2.circle(output, center, 7, color, -1, cv2.LINE_AA)
            cv2.circle(output, center, 9, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(
                output,
                f"{index}:{kp_conf:.2f}",
                (center[0] + 8, center[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )
        if pose_timestamp is not None:
            cv2.putText(
                output,
                f"pose sample t={pose_timestamp:.1f}s",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    distance_label = (
        f"dist_to_bed={dist_to_bed:.2f}"
        if dist_to_bed is not None
        else "dist_to_bed=n/a"
    )
    label = f"{state}  score={confidence:.2f}"
    event_lines: list[str] = []
    if event is not None:
        start_time = float(event["start_time"])
        confirmed_time = float(event["confirmed_time"])
        status = (
            "CONFIRMED"
            if timestamp >= confirmed_time
            else "CANDIDATE"
        )
        event_lines = [
            f"EVENT: {event['type']} [{status}]",
            f"from {start_time:.1f}s -> confirmed {confirmed_time:.1f}s",
            f"{event['previous_state']} -> {event['current_state']}",
        ]
        if isinstance(event.get("confidence"), (int, float)):
            event_lines.append(f"event confidence={float(event['confidence']):.2f}")
        elif event.get("confidence"):
            event_lines.append(f"event confidence={event['confidence']}")
        if event.get("note"):
            event_lines.append(f"note: {event['note']}")

    panel_bottom = max(108, 108 + 27 * len(event_lines))
    cv2.rectangle(output, (20, 20), (900, panel_bottom), (0, 0, 0), -1)
    cv2.putText(
        output,
        label,
        (35, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        distance_label,
        (35, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    for line_index, event_line in enumerate(event_lines):
        cv2.putText(
            output,
            event_line,
            (35, 135 + line_index * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 165, 255) if "CONFIRMED" in event_line else (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        output,
        f"t={timestamp:.1f}s",
        (output.shape[1] - 180, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--states", type=Path, default=Path("data/processed/smoothed_states.parquet"))
    parser.add_argument("--scores", type=Path, default=Path("data/processed/state_scores.parquet"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/features.parquet"))
    parser.add_argument("--pose", type=Path, default=Path("data/processed/raw_pose.parquet"))
    parser.add_argument("--events", type=Path, default=Path("data/processed/events.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/debug_overlay.mp4"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    output_path = args.output.resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing video: {output_path}")

    config = load_config(config_path)
    video_path = resolve_path(config_path, config["video"])
    states_path = args.states if args.states.is_absolute() else config_path.parent / args.states
    scores_path = args.scores if args.scores.is_absolute() else config_path.parent / args.scores
    states = pd.read_parquet(states_path).set_index("frame_idx")
    scores = pd.read_parquet(scores_path).set_index("frame_idx")
    features_path = args.features if args.features.is_absolute() else config_path.parent / args.features
    features = pd.read_parquet(features_path).set_index("frame_idx")
    pose_path = args.pose if args.pose.is_absolute() else config_path.parent / args.pose
    pose = pd.read_parquet(pose_path).set_index("frame_idx")
    events_path = args.events if args.events.is_absolute() else config_path.parent / args.events
    events: list[dict[str, Any]] = []
    if events_path.exists():
        events = json.loads(events_path.read_text(encoding="utf-8"))
        if not isinstance(events, list):
            raise ValueError(f"Events file must contain a JSON list: {events_path}")
    if not states.index.equals(scores.index):
        raise ValueError("State and score files do not have matching frame indices")
    if not states.index.equals(features.index):
        raise ValueError("State and feature files do not have matching frame indices")

    polygon = np.asarray(config["bed_polygon"], dtype=np.int32).reshape(-1, 1, 2)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Video has invalid FPS or dimensions")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to create output video: {output_path}")

    current_state = "UNKNOWN"
    current_confidence = 0.0
    current_dist_to_bed: float | None = None
    current_keypoints: np.ndarray | None = None
    pose_timestamp: float | None = None
    min_kp_conf = float(config["thresholds"]["min_kp_conf"])
    frame_idx = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_idx in states.index:
                current_state = str(states.loc[frame_idx, "smoothed_state"])
                current_confidence = float(
                    scores.loc[frame_idx, current_state]
                )
                raw_distance = features.loc[frame_idx, "dist_to_bed"]
                current_dist_to_bed = (
                    float(raw_distance)
                    if pd.notna(raw_distance)
                    else None
                )
            if frame_idx in pose.index and bool(pose.loc[frame_idx, "present"]):
                raw_keypoints = pose.loc[frame_idx, "keypoints"]
                current_keypoints = (
                    np.stack(raw_keypoints).astype(float)
                    if raw_keypoints is not None
                    else None
                )
                pose_timestamp = float(pose.loc[frame_idx, "t"])
            timestamp = frame_idx / fps
            current_event = next(
                (
                    event
                    for event in events
                    if float(event["start_time"]) <= timestamp
                    <= float(event["confirmed_time"]) + 5.0
                ),
                None,
            )
            writer.write(
                draw_overlay(
                    frame,
                    polygon,
                    current_state,
                    current_confidence,
                    timestamp,
                    current_dist_to_bed,
                    current_keypoints,
                    pose_timestamp,
                    min_kp_conf,
                    current_event,
                )
            )
            frame_idx += 1
    finally:
        capture.release()
        writer.release()

    print(f"Rendered {frame_idx} frames to {output_path}")
    print(f"Resolution: {width}x{height} at {fps:.3f} FPS")


if __name__ == "__main__":
    main()
