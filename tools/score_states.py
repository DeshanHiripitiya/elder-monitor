"""Compute normalized per-frame state scores from extracted features."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.frame_sampler import load_config

STATES = (
    "LYING_IN_BED",
    "SITTING_ON_BED",
    "SITTING_OUTSIDE_BED",
    "STANDING",
    "WALKING",
    "OUT_OF_BED",
    "UNKNOWN",
    "LYING_ON_FLOOR",
)


def parse_timestamp(value: str) -> float:
    parts = [float(part) for part in str(value).split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Invalid timestamp in gt.csv: {value}")


def ground_truth_labels(gt_path: Path, timestamps: pd.Series) -> pd.Series:
    truth = pd.read_csv(gt_path)
    required = {"start", "end", "state"}
    if not required.issubset(truth.columns):
        raise ValueError(f"{gt_path} must contain columns: {sorted(required)}")

    intervals = [
        (parse_timestamp(row.start), parse_timestamp(row.end), row.state)
        for row in truth.itertuples(index=False)
    ]
    unknown_states = sorted({state for _, _, state in intervals} - set(STATES))
    if unknown_states:
        raise ValueError(f"Unknown ground-truth states: {unknown_states}")

    labels: list[str | None] = []
    for timestamp in timestamps:
        matches = [
            state for start, end, state in intervals if start <= timestamp < end
        ]
        labels.append(matches[0] if matches else None)
    return pd.Series(labels, index=timestamps.index, dtype="string")


def print_accuracy(scores: pd.DataFrame, gt_path: Path) -> None:
    truth = ground_truth_labels(gt_path, scores["t"])
    comparable = truth.notna()
    if not comparable.any():
        print(f"No frame timestamps fall within {gt_path}")
        return

    predicted = scores.loc[comparable, "argmax_state"]
    actual = truth.loc[comparable]
    actual.name = "ground_truth"
    accuracy = float((predicted.to_numpy() == actual.to_numpy()).mean())
    print(
        f"Ground-truth accuracy: {accuracy:.1%} "
        f"({int((predicted.to_numpy() == actual.to_numpy()).sum())}/"
        f"{len(actual)} frames)"
    )
    print("Ground-truth frame counts:", Counter(actual).most_common())
    print("Confusion matrix:")
    print(pd.crosstab(actual, predicted, dropna=False).to_string())


def clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def finite(value: Any, default: float = 0.0) -> float:
    return float(value) if pd.notna(value) else default


def score_row(row: pd.Series, config: dict[str, Any]) -> np.ndarray:
    thresholds = config["thresholds"]
    scoring = config["state_scoring"]
    if not bool(row["present"]):
        scores = np.zeros(len(STATES), dtype=float)
        scores[STATES.index("OUT_OF_BED")] = 1.0
        return scores

    angle = finite(row["torso_angle"])
    aspect = finite(row["bbox_aspect"])
    bed_fraction = finite(row["kp_in_bed_frac"])
    hip_in_bed = finite(row["hip_in_bed"])
    hip_height = finite(row["hip_knee_ratio"])
    speed = finite(row["speed"])
    visibility = finite(row["vis"])

    upright = clip01(
        (scoring["upright_angle_deg"] - angle)
        / max(scoring["upright_angle_deg"], 1e-6)
    )
    lying_angle = clip01(
        (angle - thresholds["lying_angle_deg"] + scoring["upright_angle_deg"])
        / max(90 - thresholds["lying_angle_deg"], 1e-6)
    )
    floor_bed_fraction_max = float(scoring["floor_bed_fraction_max"])
    floor_torso_angle_min = float(scoring["floor_torso_angle_min_deg"])
    lying_shape = clip01((aspect - 0.8) / 1.2)
    bed_presence = bed_fraction
    sitting_height = clip01(
        (scoring["high_hip_knee_ratio"] - hip_height)
        / max(
            scoring["high_hip_knee_ratio"] - scoring["low_hip_knee_ratio"],
            1e-6,
        )
    )
    standing_height = clip01(
        (hip_height - scoring["low_hip_knee_ratio"])
        / max(
            scoring["high_hip_knee_ratio"] - scoring["low_hip_knee_ratio"],
            1e-6,
        )
    )
    stillness = clip01(1 - speed / max(scoring["low_speed"], 1e-6))
    walking = clip01(speed / max(thresholds["walk_speed"], 1e-6))
    confidence = clip01(visibility)

    raw = np.array(
        [
            max(lying_angle, lying_shape) * bed_presence,
            upright * sitting_height * max(bed_presence, hip_in_bed),
            upright * sitting_height * (1 - hip_in_bed),
            upright * standing_height * stillness,
            upright * standing_height * walking,
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )
    if (
        pd.notna(row["kp_in_bed_frac"])
        and bed_fraction <= floor_bed_fraction_max
        and angle >= floor_torso_angle_min
    ):
        raw[STATES.index("LYING_ON_FLOOR")] = max(lying_angle, 1.0)
    if confidence < scoring["visibility_unknown"]:
        raw[STATES.index("UNKNOWN")] = 1.0 - confidence
    if float(raw.max(initial=0.0)) < thresholds["unknown_score"]:
        raw[STATES.index("UNKNOWN")] = max(
            raw[STATES.index("UNKNOWN")],
            thresholds["unknown_score"],
        )
    total = float(raw.sum())
    return raw / total if total > 0 else np.eye(1, len(STATES), STATES.index("UNKNOWN"))[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--input", type=Path, default=Path("data/features.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/state_scores.parquet"))
    parser.add_argument("--matrix", type=Path, default=Path("data/frame_scores.npy"))
    parser.add_argument("--gt", type=Path, default=Path("data/gt.csv"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = args.output.resolve()
    matrix_path = args.matrix.resolve()
    if (output_path.exists() or matrix_path.exists()) and not args.overwrite:
        raise FileExistsError("State outputs exist; use --overwrite to regenerate them")

    config_path = args.config.resolve()
    config = load_config(config_path)
    input_path = args.input if args.input.is_absolute() else config_path.parent / args.input
    features = pd.read_parquet(input_path).sort_values("frame_idx").reset_index(drop=True)
    matrix = np.vstack([score_row(row, config) for _, row in features.iterrows()])

    scores = features[["frame_idx", "t", "present"]].copy()
    for index, state in enumerate(STATES):
        scores[state] = matrix[:, index]
    scores["argmax_state"] = [STATES[index] for index in matrix.argmax(axis=1)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(output_path, index=False)
    np.save(matrix_path, matrix)
    print(f"Saved {len(scores)} frame scores to {output_path}")
    print(f"Saved matrix with shape {matrix.shape} to {matrix_path}")
    print("Argmax counts:", scores["argmax_state"].value_counts().to_dict())
    gt_path = args.gt if args.gt.is_absolute() else config_path.parent / args.gt
    if gt_path.exists():
        print_accuracy(scores, gt_path)
    else:
        print(f"Ground-truth file not found; accuracy comparison skipped: {gt_path}")


if __name__ == "__main__":
    main()
