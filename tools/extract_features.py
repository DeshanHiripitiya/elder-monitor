"""Extract normalized pose features from the frozen raw pose Parquet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.frame_sampler import load_config

SHOULDERS = (5, 6)
HIPS = (11, 12)
KNEES = (13, 14)
ANKLES = (15, 16)


def point_mean(keypoints: np.ndarray, indices: tuple[int, ...], threshold: float) -> np.ndarray | None:
    visible = [keypoints[index, :2] for index in indices if keypoints[index, 2] > threshold]
    return np.mean(visible, axis=0) if visible else None


def point_in_polygon(point: np.ndarray | None, polygon: np.ndarray) -> float:
    if point is None or len(polygon) < 3:
        return float("nan")
    return float(cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0)


def extract_row(
    row: pd.Series,
    polygon: np.ndarray,
    min_conf: float,
    previous_hip: np.ndarray | None,
    previous_time: float | None,
) -> tuple[dict[str, Any], np.ndarray | None]:
    output: dict[str, Any] = {
        "frame_idx": int(row["frame_idx"]),
        "t": float(row["t"]),
        "track_id": row["track_id"],
        "present": bool(row["present"]),
        "torso_angle": np.nan,
        "bbox_aspect": np.nan,
        "hip_in_bed": np.nan,
        "dist_to_bed": np.nan,
        "kp_in_bed_frac": np.nan,
        "hip_height_ratio": np.nan,
        "hip_knee_ratio": np.nan,
        "speed": np.nan,
        "vis": np.nan,
    }
    if not row["present"] or row["keypoints"] is None or row["bbox"] is None:
        return output, None

    keypoints = np.stack(row["keypoints"]).astype(float)
    bbox = np.asarray(row["bbox"], dtype=float)
    if keypoints.shape != (17, 3):
        raise ValueError(f"Expected 17 keypoints, got shape {keypoints.shape}")

    visible = keypoints[:, 2] > min_conf
    output["vis"] = float(np.mean(keypoints[:, 2]))
    output["kp_in_bed_frac"] = float(
        np.mean(
            [
                point_in_polygon(keypoints[index, :2], polygon)
                for index in range(17)
                if visible[index]
            ]
        )
    ) if np.any(visible) else np.nan

    shoulders = point_mean(keypoints, SHOULDERS, min_conf)
    hips = point_mean(keypoints, HIPS, min_conf)
    ankles = point_mean(keypoints, ANKLES, min_conf)
    if bbox[3] > bbox[1]:
        output["bbox_aspect"] = float((bbox[2] - bbox[0]) / (bbox[3] - bbox[1]))
        bbox_height = bbox[3] - bbox[1]
        hip_knee_ratios = [
            abs(float(keypoints[knee, 1] - keypoints[hip, 1])) / bbox_height
            for hip, knee in zip(HIPS, KNEES)
            if keypoints[hip, 2] > min_conf and keypoints[knee, 2] > min_conf
        ]
        if hip_knee_ratios:
            output["hip_knee_ratio"] = float(np.mean(hip_knee_ratios))

    torso_length = None
    if shoulders is not None and hips is not None:
        torso = hips - shoulders
        torso_length = float(np.linalg.norm(torso))
        if torso_length > 0:
            output["torso_angle"] = float(np.degrees(np.arctan2(abs(torso[0]), abs(torso[1]))))
            output["hip_in_bed"] = point_in_polygon(hips, polygon)
            if len(polygon) >= 3:
                signed_distance = cv2.pointPolygonTest(
                    polygon,
                    (float(hips[0]), float(hips[1])),
                    True,
                )
                output["dist_to_bed"] = max(
                    0.0,
                    -float(signed_distance) / torso_length,
                )
            if previous_hip is not None and previous_time is not None:
                elapsed = float(row["t"]) - previous_time
                if elapsed > 0:
                    output["speed"] = float(np.linalg.norm(hips - previous_hip) / elapsed / torso_length)

    if hips is not None and ankles is not None and bbox[3] > bbox[1]:
        output["hip_height_ratio"] = float((ankles[1] - hips[1]) / (bbox[3] - bbox[1]))

    return output, hips


def save_diagnostic_plot(features: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(features["t"], features["torso_angle"], linewidth=1)
    axes[0].set_ylabel("Torso angle (deg)")
    axes[0].grid(alpha=0.3)
    axes[1].plot(features["t"], features["kp_in_bed_frac"], linewidth=1)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Keypoints in bed")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.3)
    axes[2].plot(features["t"], features["dist_to_bed"], linewidth=1)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Distance to bed (torso lengths)")
    axes[2].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--input", type=Path, default=Path("data/processed/raw_pose.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/features.parquet"))
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("data/processed/feature_diagnostics.png"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = args.output.resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing features: {output_path}")

    config_path = args.config.resolve()
    config = load_config(config_path)
    input_path = args.input if args.input.is_absolute() else config_path.parent / args.input
    polygon = np.asarray(config["bed_polygon"], dtype=np.float32)
    min_conf = float(config["thresholds"]["min_kp_conf"])
    raw = pd.read_parquet(input_path).sort_values("frame_idx")

    rows: list[dict[str, Any]] = []
    previous_hip: np.ndarray | None = None
    previous_time: float | None = None
    for _, raw_row in raw.iterrows():
        feature_row, hip = extract_row(
            raw_row,
            polygon,
            min_conf,
            previous_hip,
            previous_time,
        )
        rows.append(feature_row)
        if hip is not None:
            previous_hip = hip
            previous_time = float(raw_row["t"])

    features = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    plot_path = args.plot if args.plot.is_absolute() else config_path.parent / args.plot
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    save_diagnostic_plot(features, plot_path)
    print(f"Saved {len(features)} feature rows to {output_path}")
    print(f"Saved diagnostic plot to {plot_path}")


if __name__ == "__main__":
    main()
