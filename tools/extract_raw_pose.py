"""Run pose tracking once and persist the patient track as Parquet."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import cv2
from ultralytics import YOLO

# Allow direct execution as `python tools\extract_raw_pose.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.frame_sampler import load_config


def resolve_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_path.parent / path


def result_rows(
    model: YOLO,
    video_path: Path,
    sample_fps: float,
) -> tuple[list[dict[str, Any]], Counter[int], list[tuple[int, float]]]:
    """Track sampled frames and return all detections plus early track counts."""
    detections: list[dict[str, Any]] = []
    early_track_counts: Counter[int] = Counter()
    sampled_metadata: list[tuple[int, float]] = []

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
            timestamp_sec = frame_idx / video_fps
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )
            result = results[0] if results else None
            boxes = result.boxes if result is not None else None
            keypoints = result.keypoints if result is not None else None
            if frame_idx % frame_interval == 0:
                sampled_metadata.append((frame_idx, timestamp_sec))
            if boxes is None or keypoints is None or len(boxes) == 0:
                frame_idx += 1
                continue

            if frame_idx % frame_interval == 0:
                box_values = boxes.xyxy.cpu().tolist()
                track_ids = (
                    boxes.id.int().cpu().tolist()
                    if boxes.id is not None
                    else [-1] * len(box_values)
                )
                keypoint_values = keypoints.data.cpu().tolist()
                for box, track_id, points in zip(box_values, track_ids, keypoint_values):
                    track_id = int(track_id)
                    row = {
                        "frame_idx": frame_idx,
                        "t": float(timestamp_sec),
                        "track_id": track_id,
                        "present": True,
                        "bbox": [float(value) for value in box],
                        "keypoints": [
                            [float(point[0]), float(point[1]), float(point[2])]
                            for point in points[:17]
                        ],
                    }
                    detections.append(row)
                    if timestamp_sec <= 30.0:
                        early_track_counts[track_id] += 1
            frame_idx += 1
    finally:
        capture.release()

    return detections, early_track_counts, sampled_metadata


def patient_rows(
    detections: list[dict[str, Any]],
    patient_track_id: int | None,
    sampled_metadata: list[tuple[int, float]],
) -> list[dict[str, Any]]:
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in detections:
        by_frame.setdefault(int(row["frame_idx"]), []).append(row)

    def area(row: dict[str, Any]) -> float:
        x1, y1, x2, y2 = row["bbox"]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)
    rows: list[dict[str, Any]] = []
    for frame_idx, timestamp_sec in sampled_metadata:
        frame_rows = by_frame.get(frame_idx, [])
        preferred = [
            row for row in frame_rows if row["track_id"] == patient_track_id
        ]
        row = max(preferred or frame_rows, key=area, default=None)
        if row is None:
            rows.append(
                {
                    "frame_idx": frame_idx,
                    "t": float(timestamp_sec),
                    "track_id": patient_track_id,
                    "present": False,
                    "bbox": None,
                    "keypoints": None,
                }
            )
        else:
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/raw_pose.parquet"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing raw pose output",
    )
    args = parser.parse_args()

    output_path = args.output.resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing raw pose output: {output_path}. "
            "Use --overwrite only to intentionally rerun pose tracking."
        )

    config_path = args.config.resolve()
    config = load_config(config_path)
    video_path = resolve_path(config_path, config["video"])
    model_path = resolve_path(config_path, config["pose_model"])
    model = YOLO(str(model_path))
    detections, early_counts, sampled_metadata = result_rows(
        model,
        video_path,
        float(config["sample_fps"]),
    )

    patient_track_id = early_counts.most_common(1)[0][0] if early_counts else None
    if patient_track_id is None:
        patient_track_id = -1
    rows = patient_rows(detections, patient_track_id, sampled_metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    print(f"Saved {len(rows)} rows to {output_path}")
    print(f"Patient track_id: {patient_track_id}")
    print(f"Tracked detections: {len(detections)}")


if __name__ == "__main__":
    main()
