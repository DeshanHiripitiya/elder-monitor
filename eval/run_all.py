"""Compare state predictions with ground truth on a one-second time grid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.events import detect_exits
from src.frame_sampler import load_config
from tools.compare_ground_truth_events import (
    features_from_segments,
    read_event_annotations,
)
from tools.score_states import parse_timestamp

SHIFT_SECONDS = tuple(range(-5, 6))
BOUNDARY_TOLERANCES = (1.0, 2.0)
UNKNOWN = "UNKNOWN"
REPORT_STATES = (
    "LYING_IN_BED",
    "SITTING_ON_BED",
    "STANDING",
    "WALKING",
    "SITTING_OUTSIDE_BED",
    "OUT_OF_BED",
    "UNKNOWN",
    "LYING_ON_FLOOR",
)


def to_grid(
    segments: list[dict[str, Any]],
    duration: float,
    step: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample half-open state segments at regular intervals from time zero."""
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be a finite positive number")
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be a finite positive number")

    times = np.arange(0.0, duration, step, dtype=float)
    labels = np.full(times.shape, UNKNOWN, dtype=object)
    occupied = np.zeros(times.shape, dtype=bool)
    for segment in segments:
        start = float(segment["start"])
        end = float(segment["end"])
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            raise ValueError(f"Invalid segment interval: {segment}")
        state = str(segment["state"])
        mask = (times >= start) & (times < end)
        if np.any(occupied & mask):
            raise ValueError(f"Prediction segments overlap at {start}..{end}")
        labels[mask] = state
        occupied[mask] = True
    return times, labels


def load_ground_truth(path: Path) -> list[dict[str, Any]]:
    truth = pd.read_csv(path)
    required = {"start", "end", "state"}
    if not required.issubset(truth.columns):
        raise ValueError(f"{path} must contain columns: {sorted(required)}")
    segments = [
        {
            "start": parse_timestamp(row.start),
            "end": parse_timestamp(row.end),
            "state": str(row.state),
        }
        for row in truth.itertuples(index=False)
    ]
    _validate_segments(segments, str(path))
    return sorted(segments, key=lambda item: float(item["start"]))


def _validate_segments(segments: list[dict[str, Any]], name: str) -> None:
    previous_end = 0.0
    for segment in sorted(segments, key=lambda item: float(item["start"])):
        start = float(segment["start"])
        end = float(segment["end"])
        if start < 0 or end <= start:
            raise ValueError(f"{name} has an invalid interval: {segment}")
        if start < previous_end:
            raise ValueError(f"{name} contains overlapping intervals: {segment}")
        previous_end = end


def _samples_to_segments(
    samples: pd.DataFrame,
    label_column: str,
    duration: float,
) -> list[dict[str, Any]]:
    required = ["t", label_column]
    if "_prediction_confidence" in samples.columns:
        required.append("_prediction_confidence")
    ordered = samples[required].sort_values("t").reset_index(drop=True)
    if ordered.empty:
        raise ValueError(f"{label_column} contains no prediction samples")
    times = ordered["t"].to_numpy(dtype=float)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError(f"{label_column} timestamps must be finite and increasing")

    labels = ordered[label_column].astype(str).to_numpy()
    segments: list[dict[str, Any]] = []
    start_index = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start_index]:
            start = float(times[start_index])
            end = float(times[index]) if index < len(times) else duration
            if end > start:
                segment = {
                    "start": start,
                    "end": end,
                    "state": labels[start_index],
                }
                if "_prediction_confidence" in ordered.columns:
                    confidence = ordered["_prediction_confidence"].iloc[
                        start_index:index
                    ]
                    segment["mean_confidence"] = float(confidence.mean())
                else:
                    segment["mean_confidence"] = 1.0
                segments.append(segment)
            start_index = index
    return segments


def _add_prediction_confidence(
    samples: pd.DataFrame,
    label_column: str,
    score_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the probability of each selected state to its sampled row."""
    if "frame_idx" not in samples or "frame_idx" not in score_frame:
        raise ValueError("Frame-level confidence alignment requires frame_idx")
    score_lookup = score_frame.set_index("frame_idx")
    missing = set(samples["frame_idx"]) - set(score_lookup.index)
    if missing:
        raise ValueError(
            f"State scores are missing {len(missing)} prediction frame indices"
        )
    result = samples.copy()
    confidences = []
    for row in result[["frame_idx", label_column]].itertuples(index=False):
        state_score = score_lookup.loc[row.frame_idx, row[1]]
        confidences.append(float(state_score))
    result["_prediction_confidence"] = confidences
    return result


def _score_shift(
    truth: np.ndarray,
    predictions: np.ndarray,
    step: float,
    shift_seconds: int,
) -> dict[str, Any]:
    _, shifted = to_grid(
        [
            {
                "start": float(index * step + shift_seconds),
                "end": float((index + 1) * step + shift_seconds),
                "state": state,
            }
            for index, state in enumerate(predictions)
        ],
        len(truth) * step,
        step,
    )
    comparable = truth != UNKNOWN
    matches = shifted[comparable] == truth[comparable]
    correct = int(matches.sum())
    total = int(comparable.sum())
    return {
        "shift_sec": shift_seconds,
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
        "unknown_predictions": int(np.count_nonzero(shifted[comparable] == UNKNOWN)),
    }


def evaluate_mode(
    truth: np.ndarray,
    predictions: np.ndarray,
    step: float,
    boundary_times: list[float] | None = None,
) -> dict[str, Any]:
    """Calculate strict, class-wise, selective, boundary, and shift metrics."""
    if len(truth) != len(predictions):
        raise ValueError("Truth and prediction grids must have matching lengths")
    labeled = truth != UNKNOWN
    actual = truth[labeled].astype(str)
    predicted = predictions[labeled].astype(str)
    if not len(actual):
        raise ValueError("No ground-truth samples are available to evaluate")

    class_labels = sorted(set(actual))
    predicted_labels = sorted(set(predicted) | set(class_labels))
    confusion_counts = {
        label: {prediction: 0 for prediction in predicted_labels}
        for label in class_labels
    }
    for expected, output in zip(actual, predicted):
        confusion_counts[expected][output] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values = []
    for label in class_labels:
        true_positive = confusion_counts[label][label]
        false_positive = sum(
            confusion_counts[other][label]
            for other in class_labels
            if other != label
        )
        false_negative = sum(
            count
            for output, count in confusion_counts[label].items()
            if output != label
        )
        support = sum(confusion_counts[label].values())
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1_values.append(f1)

    sample_times = np.arange(len(truth), dtype=float) * step
    transition_times = boundary_times or []
    tolerant_accuracy = {}
    for tolerance in BOUNDARY_TOLERANCES:
        near_boundary = np.zeros(len(truth), dtype=bool)
        for transition in transition_times:
            near_boundary |= np.abs(sample_times - transition) <= tolerance
        scored = labeled & ~near_boundary
        total = int(np.count_nonzero(scored))
        correct = int(np.count_nonzero(truth[scored] == predictions[scored]))
        tolerant_accuracy[f"{tolerance:g}_sec"] = {
            "correct": correct,
            "total": total,
            "ignored_samples": int(np.count_nonzero(labeled & near_boundary)),
            "accuracy": correct / total if total else None,
        }

    covered = labeled & (predictions != UNKNOWN)
    coverage = int(np.count_nonzero(covered))
    selective_correct = int(
        np.count_nonzero(truth[covered] == predictions[covered])
    )
    total_labeled = int(np.count_nonzero(labeled))
    strict_correct = int(np.count_nonzero(truth[labeled] == predictions[labeled]))

    off_diagonal = sorted(
        (
            (count, expected, output)
            for expected, row in confusion_counts.items()
            for output, count in row.items()
            if expected != output and count
        ),
        reverse=True,
    )
    top_confusions = [
        {
            "ground_truth": expected,
            "predicted": output,
            "count": count,
            "sentence": (
                f"{expected} was predicted as {output} for {count} labeled "
                f"samples ({count * step:g} seconds)."
            ),
        }
        for count, expected, output in off_diagonal[:3]
    ]

    shifts = [
        _score_shift(truth, predictions, step, shift)
        for shift in SHIFT_SECONDS
    ]
    best = max(
        shifts,
        key=lambda item: (
            -1.0 if item["accuracy"] is None else item["accuracy"],
            -abs(item["shift_sec"]),
            -item["shift_sec"],
        ),
    )
    zero = next(item for item in shifts if item["shift_sec"] == 0)
    confusion_normalized = {
        expected: {
            output: count / sum(row.values()) if sum(row.values()) else 0.0
            for output, count in row.items()
        }
        for expected, row in confusion_counts.items()
    }
    return {
        "overall_accuracy": strict_correct / total_labeled,
        "strict_accuracy": {
            "correct": strict_correct,
            "total": total_labeled,
            "accuracy": strict_correct / total_labeled,
        },
        "coverage": {
            "committed_samples": coverage,
            "total_labeled_samples": total_labeled,
            "coverage": coverage / total_labeled,
            "selective_correct": selective_correct,
            "selective_accuracy": (
                selective_correct / coverage if coverage else None
            ),
            "abstained_samples": total_labeled - coverage,
            "abstained_fraction": (total_labeled - coverage) / total_labeled,
        },
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion_matrix": {
            "ground_truth_labels": class_labels,
            "predicted_labels": predicted_labels,
            "counts": confusion_counts,
            "row_normalized": confusion_normalized,
        },
        "boundary_tolerant_accuracy": tolerant_accuracy,
        "top_off_diagonal_confusions": top_confusions,
        "zero_shift": zero,
        "best_shift": best,
        "shifts": shifts,
    }


def duration_summary(
    segments: list[dict[str, Any]],
    duration: float,
    in_bed_states: set[str],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Sum state and bed periods from segment boundaries over the video span."""
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be a finite positive number")
    ordered = sorted(segments, key=lambda segment: float(segment["start"]))
    totals = {state: 0.0 for state in REPORT_STATES}
    cursor = 0.0
    for segment in ordered:
        start = float(segment["start"])
        end = float(segment["end"])
        if end <= 0 or start >= duration:
            continue
        start = max(0.0, start)
        end = min(duration, end)
        if start < cursor - 1e-6:
            raise ValueError("Duration segments overlap")
        if start > cursor:
            totals[UNKNOWN] += start - cursor
        if end > start:
            state = str(segment["state"])
            if state not in totals:
                totals[state] = 0.0
            totals[state] += end - start
            cursor = end
    if cursor < duration:
        totals[UNKNOWN] += duration - cursor

    total_duration = sum(totals.values())
    if abs(total_duration - duration) >= 1.0:
        raise ValueError(
            f"Predicted durations sum to {total_duration:.3f}s, "
            f"not video duration {duration:.3f}s"
        )

    ordered_events = sorted(
        (
            event
            for event in events
            if event.get("type") in {"bed_exit", "bed_return"}
        ),
        key=lambda event: float(event["confirmed_time"]),
    )
    exits = [
        float(event["confirmed_time"])
        for event in ordered_events
        if event["type"] == "bed_exit"
    ]
    returns = [
        float(event["confirmed_time"])
        for event in ordered_events
        if event["type"] == "bed_return"
    ]
    out_periods = []
    for index, exit_time in enumerate(exits):
        next_return = next(
            (return_time for return_time in returns if return_time >= exit_time),
            duration,
        )
        out_periods.append(max(0.0, next_return - exit_time))

    in_bed_time = sum(
        time for state, time in totals.items() if state in in_bed_states
    )
    out_of_bed_time = sum(totals.values()) - in_bed_time
    return {
        "duration_by_state_sec": totals,
        "duration_sum_sec": total_duration,
        "video_duration_sec": duration,
        "duration_sum_check_passed": abs(total_duration - duration) < 1.0,
        "time_in_bed_sec": in_bed_time,
        "time_out_of_bed_sec": out_of_bed_time,
        "exit_count": len(exits),
        "return_count": len(returns),
        "longest_out_of_bed_period_sec": max(out_periods, default=0.0),
    }


def compare_durations(
    ground_truth: dict[str, Any],
    predicted: dict[str, Any],
) -> dict[str, Any]:
    """Compare each state duration and summarize total misattributed time."""
    rows = []
    absolute_errors = []
    for state in REPORT_STATES:
        gt_duration = float(ground_truth["duration_by_state_sec"].get(state, 0.0))
        predicted_duration = float(
            predicted["duration_by_state_sec"].get(state, 0.0)
        )
        absolute_error = abs(predicted_duration - gt_duration)
        percent_of_gt = (
            100.0 * absolute_error / gt_duration if gt_duration else None
        )
        rows.append(
            {
                "state": state,
                "ground_truth_sec": gt_duration,
                "predicted_sec": predicted_duration,
                "absolute_error_sec": absolute_error,
                "percent_of_ground_truth": percent_of_gt,
            }
        )
        absolute_errors.append(absolute_error)
    sum_absolute_error = sum(absolute_errors)
    video_duration = float(ground_truth["video_duration_sec"])
    return {
        "states": rows,
        "mean_absolute_error_sec": float(np.mean(absolute_errors)),
        "sum_absolute_error_sec": sum_absolute_error,
        "total_misattributed_time_sec": sum_absolute_error / 2.0,
        "total_misattributed_time_percent_of_video": (
            (sum_absolute_error / 2.0) / video_duration * 100.0
            if video_duration
            else None
        ),
        "predicted_duration_sum_sec": float(predicted["duration_sum_sec"]),
        "video_duration_sec": video_duration,
        "bed_summary": {
            "time_in_bed_sec": {
                "ground_truth": ground_truth["time_in_bed_sec"],
                "predicted": predicted["time_in_bed_sec"],
            },
            "time_out_of_bed_sec": {
                "ground_truth": ground_truth["time_out_of_bed_sec"],
                "predicted": predicted["time_out_of_bed_sec"],
            },
            "exit_count": {
                "ground_truth": ground_truth["exit_count"],
                "predicted": predicted["exit_count"],
            },
            "return_count": {
                "ground_truth": ground_truth["return_count"],
                "predicted": predicted["return_count"],
            },
            "longest_out_of_bed_period_sec": {
                "ground_truth": ground_truth[
                    "longest_out_of_bed_period_sec"
                ],
                "predicted": predicted[
                    "longest_out_of_bed_period_sec"
                ],
            },
        },
        "predicted_duration_sum_check_passed": predicted[
            "duration_sum_check_passed"
        ],
        "duration_caution": (
            "Duration error can hide state swaps: overcounting one state and "
            "undercounting another by the same amount can leave other totals "
            "unchanged. Review the confusion matrix alongside this table."
        ),
    }


def write_duration_summary(
    metrics: dict[str, Any],
    output_path: Path,
) -> None:
    """Write the requested concise duration and bed summary table."""
    lines = [
        "# State duration evaluation",
        "",
        "| State | Ground truth | Predicted | Absolute error | % of GT |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in metrics["states"]:
        percent = (
            "n/a"
            if row["percent_of_ground_truth"] is None
            else f"{row['percent_of_ground_truth']:.1f}%"
        )
        lines.append(
            f"| {row['state'].replace('_', ' ').title()} | "
            f"{row['ground_truth_sec']:.1f}s | {row['predicted_sec']:.1f}s | "
            f"{row['absolute_error_sec']:.1f}s | {percent} |"
        )
    lines.extend(
        [
            "",
            f"- Mean absolute error across {len(metrics['states'])} states: "
            f"{metrics['mean_absolute_error_sec']:.1f}s.",
            f"- Total misattributed time: "
            f"{metrics['total_misattributed_time_sec']:.1f}s "
            f"({metrics['total_misattributed_time_percent_of_video']:.1f}% "
            "of video; sum of absolute state errors / 2).",
            "",
            "| Bed summary | Ground truth | Predicted |",
            "|---|---:|---:|",
        ]
    )
    for key, values in metrics["bed_summary"].items():
        label = key.replace("_", " ").title()
        if key.endswith("_sec"):
            gt = f"{values['ground_truth']:.1f}s"
            prediction = f"{values['predicted']:.1f}s"
        else:
            gt = str(values["ground_truth"])
            prediction = str(values["predicted"])
        lines.append(f"| {label} | {gt} | {prediction} |")
    lines.extend(
        [
            "",
            "**Duration sums:** predicted "
            f"{metrics['predicted_duration_sum_sec']:.1f}s; "
            f"video {metrics['video_duration_sec']:.1f}s; "
            f"within 1s: {metrics['predicted_duration_sum_check_passed']}.",
            "",
            "**Caution:** Duration error can hide mistakes. A 5s overcount of "
            "walking and a 5s undercount of standing can leave other duration "
            "totals unchanged. Read the confusion matrix next to this table.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_confusion_heatmap(
    evaluation: dict[str, Any],
    mode: str,
    output_path: Path,
) -> None:
    """Write a row-normalized confusion heatmap with raw counts per cell."""
    matrix = evaluation["confusion_matrix"]
    rows = matrix["ground_truth_labels"]
    columns = matrix["predicted_labels"]
    counts = matrix["counts"]
    normalized = matrix["row_normalized"]
    values = np.array(
        [
            [normalized[row].get(column, 0.0) for column in columns]
            for row in rows
        ],
        dtype=float,
    )
    count_values = np.array(
        [
            [counts.get(row, {}).get(column, 0) for column in columns]
            for row in rows
        ],
        dtype=int,
    )
    figure_size = max(8.0, 1.0 + 0.9 * max(len(rows), len(columns)))
    figure, axis = plt.subplots(figsize=(figure_size, figure_size * 0.82))
    image = axis.imshow(values, cmap="Blues", vmin=0.0, vmax=1.0)
    axis.set(
        xticks=np.arange(len(columns)),
        yticks=np.arange(len(rows)),
        xticklabels=columns,
        yticklabels=rows,
        xlabel="Predicted state",
        ylabel="Ground-truth state",
        title=f"{mode.replace('_', ' ').title()} confusion matrix\n"
        "row-normalized; cells show count and row percentage",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = float(values.max()) / 2 if values.size else 0.0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            count = count_values[row, column]
            color = "white" if values[row, column] > threshold else "black"
            axis.text(
                column,
                row,
                f"{count}\n{values[row, column]:.0%}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    figure.colorbar(image, ax=axis, label="Fraction of ground-truth row")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def ground_truth_detector_segments(
    annotations: pd.DataFrame,
    duration: float,
) -> list[dict[str, Any]]:
    """Build a deterministic state trace from annotated events and trap windows."""
    rows = []
    for row in annotations.to_dict("records"):
        rows.append(
            {
                **row,
                "start": parse_timestamp(str(row["start_time"])),
                "confirmed": parse_timestamp(str(row["confirmed_time"])),
            }
        )
    rows.sort(key=lambda row: row["start"])

    segments: list[dict[str, Any]] = []

    def append(
        start: float,
        end: float,
        state: str,
        *,
        present: bool = True,
    ) -> None:
        if end <= start:
            return
        segment: dict[str, Any] = {
            "start": start,
            "end": end,
            "state": state,
            "mean_confidence": 1.0,
        }
        if not present:
            segment["_present"] = False
        segments.append(segment)

    cursor = 0.0
    gap_state = "LYING_IN_BED"
    gap_present = True
    for row in rows:
        start = float(row["start"])
        confirmed = float(row["confirmed"])
        if start < cursor or confirmed <= start:
            raise ValueError(f"Invalid or overlapping ground-truth event: {row}")
        append(cursor, start, gap_state, present=gap_present)
        kind = str(row["event"])
        note = str(row.get("note", "")).lower()
        if kind == "bed_exit":
            if "camera view" in note:
                append(start, confirmed, "OUT_OF_BED", present=False)
                gap_state = "OUT_OF_BED"
                gap_present = False
            else:
                sit_end = min(start + 2.0, confirmed)
                stand_end = min(sit_end + 3.0, confirmed)
                append(start, sit_end, "SITTING_ON_BED")
                append(sit_end, stand_end, "STANDING")
                append(stand_end, confirmed, "WALKING")
                gap_state = "OUT_OF_BED"
                gap_present = True
        elif kind == "bed_return":
            sit_end = min(start + 2.0, confirmed)
            append(start, sit_end, "SITTING_ON_BED", present=True)
            append(sit_end, confirmed, "LYING_IN_BED", present=True)
            gap_state = "LYING_IN_BED"
            gap_present = True
        elif kind == "not_an_exit":
            if "stood 5s" in note:
                stand_end = min(start + 5.0, confirmed)
                append(start, stand_end, "STANDING")
                append(stand_end, confirmed, "SITTING_ON_BED")
            elif "sit up" in note:
                sit_end = min(start + 5.0, confirmed)
                append(start, sit_end, "SITTING_ON_BED")
                append(sit_end, confirmed, "LYING_IN_BED")
            else:
                append(start, confirmed, "SITTING_ON_BED")
            gap_state = "LYING_IN_BED"
            gap_present = True
        else:
            raise ValueError(f"Unsupported ground-truth event type: {kind}")
        cursor = confirmed
    append(cursor, duration, gap_state, present=gap_present)
    return segments


def match_events(
    predicted: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    tolerance: float,
) -> dict[str, Any]:
    """Greedily match each GT event to its nearest unused same-kind prediction."""
    used: set[int] = set()
    matches: list[dict[str, Any]] = []
    for truth_index, truth_event in sorted(
        enumerate(ground_truth),
        key=lambda item: float(item[1]["confirmed_time"]),
    ):
        candidates = [
            (
                abs(
                    float(prediction["confirmed_time"])
                    - float(truth_event["confirmed_time"])
                ),
                prediction_index,
            )
            for prediction_index, prediction in enumerate(predicted)
            if prediction_index not in used
            and prediction["type"] == truth_event["type"]
            and abs(
                float(prediction["confirmed_time"])
                - float(truth_event["confirmed_time"])
            )
            <= tolerance
        ]
        if not candidates:
            continue
        _, prediction_index = min(candidates)
        used.add(prediction_index)
        prediction = predicted[prediction_index]
        matches.append(
            _event_timing_record(
                truth_index,
                prediction_index,
                str(truth_event["type"]),
                prediction,
                truth_event,
            )
        )
    unmatched_predictions = [
        index for index in range(len(predicted)) if index not in used
    ]
    matched_truth = {match["ground_truth_event_index"] for match in matches}
    return {
        "matches": matches,
        "unmatched_prediction_indices": unmatched_predictions,
        "unmatched_ground_truth_indices": [
            index for index in range(len(ground_truth)) if index not in matched_truth
        ],
    }


def _event_timing_record(
    truth_index: int,
    prediction_index: int,
    kind: str,
    prediction: dict[str, Any],
    truth_event: dict[str, Any],
) -> dict[str, Any]:
    predicted_start = float(prediction["start_time"])
    predicted_confirmed = float(prediction["confirmed_time"])
    truth_start = float(truth_event["start_time"])
    truth_confirmed = float(truth_event["confirmed_time"])
    return {
        "ground_truth_event_index": truth_index,
        "predicted_event_index": prediction_index,
        "kind": kind,
        "ground_truth_start_time": truth_start,
        "predicted_start_time": predicted_start,
        "start_time_error_sec": predicted_start - truth_start,
        "ground_truth_confirmed_time": truth_confirmed,
        "predicted_confirmed_time": predicted_confirmed,
        "confirmed_time_error_sec": predicted_confirmed - truth_confirmed,
        "ground_truth_duration_sec": truth_confirmed - truth_start,
        "detection_delay_from_gt_start_sec": predicted_confirmed - truth_start,
        "detector_confirmation_latency_sec": predicted_confirmed - predicted_start,
    }


def _nearest_timing_candidates(
    predicted: list[dict[str, Any]],
    truth: list[dict[str, Any]],
    unmatched_truth_indices: set[int],
    matched_prediction_indices: set[int],
) -> list[dict[str, Any]]:
    available_truth = set(unmatched_truth_indices)
    available_predictions = set(range(len(predicted))) - matched_prediction_indices
    pairs = sorted(
        (
            abs(
                float(predicted[prediction_index]["confirmed_time"])
                - float(truth[truth_index]["confirmed_time"])
            ),
            truth_index,
            prediction_index,
        )
        for truth_index in available_truth
        for prediction_index in available_predictions
        if truth[truth_index]["type"] == predicted[prediction_index]["type"]
    )
    selected: dict[int, int] = {}
    for _distance, truth_index, prediction_index in pairs:
        if (
            truth_index in available_truth
            and prediction_index in available_predictions
        ):
            selected[truth_index] = prediction_index
            available_truth.remove(truth_index)
            available_predictions.remove(prediction_index)

    candidates = []
    for truth_index in sorted(unmatched_truth_indices):
        if truth_index not in selected:
            candidates.append(
                {
                    "ground_truth_event_index": truth_index,
                    "kind": truth[truth_index]["type"],
                    "scored_as_match": False,
                    "nearest_prediction_available": False,
                }
            )
            continue
        prediction_index = selected[truth_index]
        prediction = predicted[prediction_index]
        truth_event = truth[truth_index]
        distance = abs(
            float(prediction["confirmed_time"])
            - float(truth_event["confirmed_time"])
        )
        candidates.append(
            {
                **_event_timing_record(
                    truth_index,
                    prediction_index,
                    str(truth_event["type"]),
                    prediction,
                    truth_event,
                ),
                "scored_as_match": False,
                "nearest_prediction_available": True,
                "outside_tolerance_sec": distance,
            }
        )
    return candidates


def _event_count_metric(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    precision = tp / precision_denominator if precision_denominator else 0.0
    recall = tp / recall_denominator if recall_denominator else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": {
            "correct": tp,
            "total": precision_denominator,
            "value": precision,
        },
        "recall": {
            "correct": tp,
            "total": recall_denominator,
            "value": recall,
        },
        "f1": f1,
    }


def evaluate_event_mode(
    predictions: list[dict[str, Any]],
    truth: list[dict[str, Any]],
    tolerance: float,
    traps: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score exits and returns independently and label unmatched exit traps."""
    results: dict[str, Any] = {}
    for kind in ("bed_exit", "bed_return"):
        predicted_kind = [
            event for event in predictions if event.get("type") == kind
        ]
        truth_kind = [event for event in truth if event.get("type") == kind]
        matched = match_events(predicted_kind, truth_kind, tolerance)
        tp = len(matched["matches"])
        fp = len(matched["unmatched_prediction_indices"])
        fn = len(matched["unmatched_ground_truth_indices"])
        false_predictions = [
            {
                "predicted_event_index": index,
                **predicted_kind[index],
            }
            for index in matched["unmatched_prediction_indices"]
        ]
        if kind == "bed_exit":
            for event in false_predictions:
                overlaps = [
                    trap["name"]
                    for trap in traps
                    if float(event["start_time"]) < trap["end"]
                    and float(event["confirmed_time"]) > trap["start"]
                ]
                if overlaps:
                    event["classification"] = "annotated_trap_false_positive"
                    event["explanation"] = (
                        "overlaps annotated non-exit trap(s): "
                        + ", ".join(overlaps)
                    )
                else:
                    nearest = min(
                        (
                            (
                                abs(
                                    float(event["confirmed_time"])
                                    - float(truth_event["confirmed_time"])
                                ),
                                truth_index,
                                truth_event,
                            )
                            for truth_index, truth_event in enumerate(truth)
                            if truth_event["type"] == kind
                        ),
                        default=None,
                    )
                    if nearest is None:
                        event["classification"] = "false_positive_without_nearby_gt"
                        event["explanation"] = (
                            "not linked to an annotated non-exit trap or a "
                            "same-kind ground-truth event"
                        )
                    else:
                        distance, truth_index, truth_event = nearest
                        event["classification"] = (
                            "outside_matching_tolerance"
                            if distance > tolerance
                            else "duplicate_or_conflicting_match"
                        )
                        event["nearest_ground_truth"] = {
                            "event_index": truth_index,
                            "confirmed_time": float(truth_event["confirmed_time"]),
                            "confirmed_time_error_sec": (
                                float(event["confirmed_time"])
                                - float(truth_event["confirmed_time"])
                            ),
                        }
                        event["explanation"] = (
                            f"not linked to an annotated trap; nearest GT "
                            f"bed_exit is {distance:.1f}s away, outside the "
                            f"+/-{tolerance:g}s matching tolerance"
                            if distance > tolerance
                            else "not linked to a trap; conflicting with a GT match"
                        )
        results[kind] = {
            **_event_count_metric(tp, fp, fn),
            "matches": matched["matches"],
            "nearest_candidates_for_missed_ground_truth": _nearest_timing_candidates(
                predicted_kind,
                truth_kind,
                set(matched["unmatched_ground_truth_indices"]),
                {
                    match["predicted_event_index"]
                    for match in matched["matches"]
                },
            ),
            "false_predictions": false_predictions,
            "missed_ground_truth_indices": matched[
                "unmatched_ground_truth_indices"
            ],
        }

    trap_results = []
    exit_events = [event for event in predictions if event.get("type") == "bed_exit"]
    for trap in traps:
        matching_exits = [
            event
            for event in exit_events
            if float(event["start_time"]) < trap["end"]
            and float(event["confirmed_time"]) > trap["start"]
        ]
        trap_results.append(
            {
                "name": trap["name"],
                "start_time": trap["start"],
                "end_time": trap["end"],
                "correctly_ignored": not matching_exits,
                "triggered_exit_count": len(matching_exits),
                "triggered_exits": matching_exits,
            }
        )
    return {
        "tolerance_sec": tolerance,
        "bed_exit": results["bed_exit"],
        "bed_return": results["bed_return"],
        "trap_results": {
            "correctly_ignored": sum(
                item["correctly_ignored"] for item in trap_results
            ),
            "total": len(trap_results),
            "cases": trap_results,
        },
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_event_summary(
    event_report: dict[str, Any],
    output_path: Path,
) -> None:
    """Write a human-readable event scorecard with timing and trap evidence."""
    lines = [
        "# Bed-event evaluation",
        "",
        "There are only 2 annotated bed exits and 2 annotated bed returns "
        "(4 positive events total). One miss changes per-kind recall by 50 "
        "percentage points; four events cannot support strong claims.",
        "",
        "Matching uses one-to-one nearest same-kind events by confirmed time. "
        "Timing error is predicted minus ground truth. Detection delay is "
        "predicted confirmation minus the GT event start; detector confirmation "
        "latency (confirmation minus predicted start) is shown separately.",
        "",
        "| Detector input | Kind | Tolerance | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, result in event_report["modes"].items():
        for tolerance, metrics in result["tolerances"].items():
            for kind in ("bed_exit", "bed_return"):
                score = metrics[kind]
                precision = score["precision"]
                recall = score["recall"]
                lines.append(
                    f"| {mode} | {kind} | +/-{tolerance}s | "
                    f"{score['tp']} | {score['fp']} | {score['fn']} | "
                    f"{precision['correct']}/{precision['total']} "
                    f"({precision['value']:.1%}) | "
                    f"{recall['correct']}/{recall['total']} "
                    f"({recall['value']:.1%}) | {score['f1']:.1%} |"
                )
    lines.extend(["", "## Negative non-exit traps", ""])
    for mode, result in event_report["modes"].items():
        metrics = result["tolerances"]["10"]
        trap_score = metrics["trap_results"]
        lines.append(
            f"- **{mode}: {trap_score['correctly_ignored']} of "
            f"{trap_score['total']} traps correctly ignored.**"
        )
        for trap in trap_score["cases"]:
            verdict = (
                "correctly ignored"
                if trap["correctly_ignored"]
                else f"triggered {trap['triggered_exit_count']} false exit(s)"
            )
            lines.append(f"  - {trap['name']}: {verdict}.")
    lines.append(
        "- UNKNOWN blip is not one of the 3 annotated traps; the existing "
        "`test_unknown_two_second_blip_does_not_create_exit` unit test covers it."
    )
    lines.extend(["", "## False exits at +/-10s matching tolerance", ""])
    found_false_exit = False
    for mode, result in event_report["modes"].items():
        false_exits = result["tolerances"]["10"]["bed_exit"]["false_predictions"]
        for event in false_exits:
            found_false_exit = True
            lines.append(
                f"- {mode}: confirmed at {event['confirmed_time']:.1f}s; "
                f"{event['explanation']}."
            )
    if not found_false_exit:
        lines.append("- No unmatched false exits were detected.")
    lines.extend(["", "## Timing", ""])
    for mode, result in event_report["modes"].items():
        for kind in ("bed_exit", "bed_return"):
            score = result["tolerances"]["10"][kind]
            records = [
                (record, True) for record in score["matches"]
            ] + [
                (record, False)
                for record in score["nearest_candidates_for_missed_ground_truth"]
            ]
            for record, matched in records:
                match_note = "matched" if matched else "nearest but outside tolerance"
                if not record.get("nearest_prediction_available", True):
                    lines.append(
                        f"- {mode} {kind}, GT "
                        f"{record['ground_truth_event_index']}: no same-kind "
                        "prediction was available for a timing comparison."
                    )
                    continue
                lines.append(
                    f"- {mode} {kind}, GT {record['ground_truth_start_time']:.1f}s/"
                    f"{record['ground_truth_confirmed_time']:.1f}s: {match_note}; "
                    f"start error {record['start_time_error_sec']:+.1f}s, "
                    f"confirmed error {record['confirmed_time_error_sec']:+.1f}s, "
                    f"detection delay from GT start "
                    f"{record['detection_delay_from_gt_start_sec']:+.1f}s, "
                    f"detector latency "
                    f"{record['detector_confirmation_latency_sec']:.1f}s."
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, default=Path("data/raw/gt.csv"))
    parser.add_argument(
        "--gt-events",
        type=Path,
        default=Path("data/raw/gt_events.csv"),
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path("data/processed/state_scores.parquet"),
    )
    parser.add_argument(
        "--smoothed",
        type=Path,
        default=Path("data/processed/smoothed_states.parquet"),
    )
    parser.add_argument(
        "--timeline",
        type=Path,
        default=Path("data/processed/timeline.parquet"),
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("data/processed/events.json"),
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/features.parquet"),
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--step", type=float, default=1.0)
    args = parser.parse_args()

    resolve = lambda path: path if path.is_absolute() else PROJECT_ROOT / path
    gt_path = resolve(args.gt)
    gt_events_path = resolve(args.gt_events)
    scores_path = resolve(args.scores)
    smoothed_path = resolve(args.smoothed)
    timeline_path = resolve(args.timeline)
    events_path = resolve(args.events)
    features_path = resolve(args.features)
    config_path = resolve(args.config)
    results_path = resolve(args.results)

    for path in (
        gt_path,
        gt_events_path,
        scores_path,
        smoothed_path,
        timeline_path,
        events_path,
        features_path,
        config_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required evaluation input does not exist: {path}")

    truth_segments = load_ground_truth(gt_path)
    gt_events = read_event_annotations(gt_events_path)
    stored_predicted_events = _read_json(events_path)
    if not isinstance(stored_predicted_events, list):
        raise ValueError(f"Predicted events must be a JSON list: {events_path}")
    annotated_events = gt_events.to_dict("records")
    truth_event_list = [
        {
            "type": str(row["event"]),
            "start_time": parse_timestamp(str(row["start_time"])),
            "confirmed_time": parse_timestamp(str(row["confirmed_time"])),
            "note": str(row.get("note", "")),
        }
        for row in annotated_events
        if str(row["event"]) in {"bed_exit", "bed_return"}
    ]
    traps = [
        {
            "name": (
                "sit-up"
                if any(
                    phrase in str(row.get("note", "")).lower()
                    for phrase in ("sat up", "sit up")
                )
                else (
                    "edge sitting"
                    if "edge" in str(row.get("note", "")).lower()
                    else "brief stand"
                )
            ),
            "start": parse_timestamp(str(row["start_time"])),
            "end": parse_timestamp(str(row["confirmed_time"])),
        }
        for row in annotated_events
        if str(row["event"]) == "not_an_exit"
    ]
    if not traps:
        raise ValueError(f"No negative not_an_exit traps are annotated in {gt_events_path}")

    timeline_frame = pd.read_parquet(timeline_path)
    if not {"start", "end", "state", "mean_confidence"}.issubset(
        timeline_frame.columns
    ):
        raise ValueError(
            f"{timeline_path} must contain start, end, state, and mean_confidence"
        )
    timeline_segments = timeline_frame.to_dict("records")
    _validate_segments(timeline_segments, str(timeline_path))
    duration = max(
        max(float(segment["end"]) for segment in truth_segments),
        max(float(segment["end"]) for segment in timeline_segments),
    )
    ground_truth_event_segments = ground_truth_detector_segments(
        gt_events,
        duration,
    )
    config = load_config(config_path)
    ground_truth_event_features = features_from_segments(
        ground_truth_event_segments,
        float(config["sample_fps"]),
    )
    ground_truth_detector_events = detect_exits(
        ground_truth_event_segments,
        ground_truth_event_features,
        config,
    )
    predicted_features = pd.read_parquet(features_path)
    predicted_detector_events = detect_exits(
        timeline_segments,
        predicted_features,
        config,
    )
    event_signature = lambda events: sorted(
        (
            str(event.get("type")),
            round(float(event["start_time"]), 6),
            round(float(event["confirmed_time"]), 6),
        )
        for event in events
    )

    scores = pd.read_parquet(scores_path)
    if not {"t", "argmax_state"}.issubset(scores.columns):
        raise ValueError(f"{scores_path} must contain t and argmax_state columns")
    smoothed = pd.read_parquet(smoothed_path)
    if not {"t", "smoothed_state"}.issubset(smoothed.columns):
        raise ValueError(f"{smoothed_path} must contain t and smoothed_state columns")
    raw_samples = _add_prediction_confidence(
        scores,
        "argmax_state",
        scores,
    )
    smoothed_samples = _add_prediction_confidence(
        smoothed,
        "smoothed_state",
        scores,
    )

    _, truth = to_grid(truth_segments, duration, args.step)
    transitions = [
        float(segment["start"])
        for index, segment in enumerate(truth_segments[1:], start=1)
        if segment["state"] != truth_segments[index - 1]["state"]
    ]
    modes = {
        "raw_argmax": _samples_to_segments(
            raw_samples,
            "argmax_state",
            duration,
        ),
        "smoothed": _samples_to_segments(
            smoothed_samples,
            "smoothed_state",
            duration,
        ),
        "timeline": timeline_segments,
    }
    evaluations: dict[str, Any] = {}
    times: np.ndarray | None = None
    for mode, segments in modes.items():
        mode_times, labels = to_grid(segments, duration, args.step)
        if times is None:
            times = mode_times
        elif not np.array_equal(times, mode_times):
            raise ValueError("Evaluation modes produced different time grids")
        evaluations[mode] = evaluate_mode(
            truth,
            labels,
            args.step,
            boundary_times=transitions,
        )

    gt_summary_events = [
        {
            "type": event["type"],
            "start_time": event["start_time"],
            "confirmed_time": event["confirmed_time"],
        }
        for event in truth_event_list
    ]
    ground_truth_durations = duration_summary(
        truth_segments,
        duration,
        set(config["in_bed_states"]),
        gt_summary_events,
    )
    duration_evaluations = {}
    for mode, segments in modes.items():
        detected_mode_events = detect_exits(
            segments,
            predicted_features,
            config,
        )
        mode_duration = duration_summary(
            segments,
            duration,
            set(config["in_bed_states"]),
            detected_mode_events,
        )
        duration_evaluations[mode] = {
            **compare_durations(ground_truth_durations, mode_duration),
            "ground_truth_duration_summary": ground_truth_durations,
            "predicted_duration_summary": mode_duration,
        }

    event_modes = {
        "ground_truth_segments": ground_truth_detector_events,
        "predicted_segments": predicted_detector_events,
    }
    event_evaluations: dict[str, Any] = {}
    for mode, predicted_events in event_modes.items():
        event_evaluations[mode] = {
            "detected_events": predicted_events,
            "tolerances": {
                str(tolerance): evaluate_event_mode(
                    predicted_events,
                    truth_event_list,
                    tolerance,
                    traps,
                )
                for tolerance in (5, 10)
            },
        }

    for evaluation in evaluations.values():
        counts = evaluation["confusion_matrix"]["counts"]
        evaluation["expected_confusions"] = {
            "lying_vs_sitting_on_bed": any(
                counts.get(actual, {}).get(predicted, 0) > 0
                for actual, predicted in (
                    ("LYING_IN_BED", "SITTING_ON_BED"),
                    ("SITTING_ON_BED", "LYING_IN_BED"),
                )
            ),
            "standing_vs_walking": any(
                counts.get(actual, {}).get(predicted, 0) > 0
                for actual, predicted in (
                    ("STANDING", "WALKING"),
                    ("WALKING", "STANDING"),
                )
            ),
        }

    shifts = {
        mode: evaluation["best_shift"]["shift_sec"]
        for mode, evaluation in evaluations.items()
    }
    report = {
        "grid": {
            "step_sec": args.step,
            "duration_sec": duration,
            "samples": len(truth),
            "ground_truth_labeled_samples": int(np.count_nonzero(truth != UNKNOWN)),
            "shift_range_sec": [min(SHIFT_SECONDS), max(SHIFT_SECONDS)],
            "positive_shift_means": "predictions are shifted later relative to GT",
        },
        "inputs": {
            "ground_truth_segments": str(gt_path.relative_to(PROJECT_ROOT)),
            "ground_truth_events": str(gt_events_path.relative_to(PROJECT_ROOT)),
            "predicted_events": str(events_path.relative_to(PROJECT_ROOT)),
            "ground_truth_event_count": len(gt_events),
            "positive_ground_truth_event_count": len(truth_event_list),
            "negative_trap_count": len(traps),
        },
        "modes": evaluations,
        "duration_evaluation": {
            "states": list(REPORT_STATES),
            "mean_absolute_error_definition": (
                "Mean of the absolute duration errors across the listed states."
            ),
            "total_misattributed_time_definition": (
                "Sum of absolute per-state duration errors divided by two, "
                "reported as seconds and a share of the video."
            ),
            "bed_period_definition": (
                "Time in bed is the sum of configured in-bed states; time out "
                "of bed is all remaining labeled or UNKNOWN time. Longest "
                "out-of-bed period pairs each exit confirmation with the next "
                "return confirmation, or video end if there is no return."
            ),
            "modes": duration_evaluations,
        },
        "bed_event_evaluation": {
            "sample_size_note": (
                "There are only 2 annotated bed exits and 2 annotated bed returns "
                "(4 positive events); one miss changes per-kind recall by 50 "
                "percentage points. These counts cannot support strong claims."
            ),
            "matching_rule": (
                "One-to-one nearest same-kind event matching using confirmed_time."
            ),
            "reference_trace_note": (
                "The GT-segment detector run uses a deterministic state trace "
                "constructed from gt_events.csv event intervals and notes. It is "
                "an annotation-derived idealized trace, not an independently "
                "frame-labeled reconstruction; trap behavior is encoded from "
                "the annotated notes."
            ),
            "timing_error_definition": (
                "Predicted timestamp minus annotated timestamp; detection delay "
                "is predicted confirmed_time minus annotated start_time. Detector "
                "latency is predicted confirmed_time minus predicted start_time."
            ),
            "trap_matching_rule": (
                "A trap is counted as triggered when a detected bed_exit interval "
                "overlaps the annotated not_an_exit interval."
            ),
            "unannotated_trap_note": (
                "UNKNOWN blip is not one of the 3 gt_events.csv negative traps; "
                "the existing test_unknown_two_second_blip_does_not_create_exit "
                "unit test covers that case separately."
            ),
            "modes": event_evaluations,
            "stored_predicted_event_output": {
                "path": str(events_path.relative_to(PROJECT_ROOT)),
                "event_count": len(stored_predicted_events),
                "agrees_with_rerun_on_predicted_segments": (
                    event_signature(stored_predicted_events)
                    == event_signature(predicted_detector_events)
                ),
            },
        },
    }

    results_path.mkdir(parents=True, exist_ok=True)
    report_path = results_path / "alignment.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    event_report_path = results_path / "bed_event_metrics.md"
    write_event_summary(report["bed_event_evaluation"], event_report_path)
    duration_report_path = results_path / "duration_metrics.md"
    write_duration_summary(
        duration_evaluations["timeline"],
        duration_report_path,
    )
    for mode, evaluation in evaluations.items():
        save_confusion_heatmap(
            evaluation,
            mode,
            results_path / f"confusion_matrix_{mode}.png",
        )
    nonzero_modes = [mode for mode, shift in shifts.items() if shift != 0]
    if nonzero_modes:
        notes_path = results_path / "NOTES.md"
        lines = [
            "# Evaluation notes",
            "",
            "The highest state accuracy occurred at a non-zero prediction offset:",
            "",
        ]
        lines.extend(f"- `{mode}`: {shift:+d} s" for mode, shift in shifts.items())
        lines.extend(
            [
                "",
                "Do not shift prediction outputs to improve this score. Review the "
                "ground-truth timestamps and clap-sync reference, correct the "
                "ground truth if it is misaligned, then rerun the evaluation.",
            ]
        )
        notes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for mode, evaluation in evaluations.items():
        zero = evaluation["zero_shift"]
        best = evaluation["best_shift"]
        print(
            f"{mode}: {zero['correct']}/{zero['total']} "
            f"({zero['accuracy']:.1%}) strict accuracy; macro F1 "
            f"{evaluation['macro_f1']:.1%}; coverage "
            f"{evaluation['coverage']['coverage']:.1%}; selective accuracy "
            f"{evaluation['coverage']['selective_accuracy']:.1%}; best "
            f"{best['accuracy']:.1%} at {best['shift_sec']:+d}s"
        )
        for tolerance, metric in evaluation["boundary_tolerant_accuracy"].items():
            print(
                f"  Boundary-tolerant ({tolerance}): "
                f"{metric['correct']}/{metric['total']} "
                f"({metric['accuracy']:.1%}); ignored "
                f"{metric['ignored_samples']} samples"
            )
        for confusion in evaluation["top_off_diagonal_confusions"]:
            print(f"  {confusion['sentence']}")
        print(f"  Expected confusions: {evaluation['expected_confusions']}")
    for mode, result in event_evaluations.items():
        detected = result["detected_events"]
        print(f"{mode} detector events: {len(detected)}")
        for tolerance, metrics in result["tolerances"].items():
            for kind in ("bed_exit", "bed_return"):
                score = metrics[kind]
                precision = score["precision"]
                recall = score["recall"]
                print(
                    f"  tol={tolerance}s {kind}: TP={score['tp']} "
                    f"FP={score['fp']} FN={score['fn']}; "
                    f"precision {precision['correct']}/{precision['total']} "
                    f"({precision['value']:.1%}); "
                    f"recall {recall['correct']}/{recall['total']} "
                    f"({recall['value']:.1%}); F1={score['f1']:.1%}"
                )
            traps_result = metrics["trap_results"]
            print(
                f"  Non-exit traps correctly ignored: "
                f"{traps_result['correctly_ignored']}/{traps_result['total']}"
            )
            for item in metrics["bed_exit"]["false_predictions"]:
                print(
                    f"  False exit at {item['confirmed_time']:.1f}s; "
                    f"{item['explanation']}"
                )
        timing_matches = result["tolerances"]["10"]["bed_exit"]["matches"] + result[
            "tolerances"
        ]["10"]["bed_return"]["matches"]
        for match in timing_matches:
            print(
                f"  {match['kind']} timing: start error "
                f"{match['start_time_error_sec']:+.1f}s; confirmed error "
                f"{match['confirmed_time_error_sec']:+.1f}s; detection delay "
                f"{match['detection_delay_from_gt_start_sec']:+.1f}s from GT "
                f"start; detector latency "
                f"{match['detector_confirmation_latency_sec']:.1f}s"
            )
    print(
        f"Ground-truth events: {len(gt_events)}; "
        f"stored predicted events: {len(stored_predicted_events)}"
    )
    print(
        "Timeline duration metrics: "
        f"MAE={duration_evaluations['timeline']['mean_absolute_error_sec']:.1f}s; "
        f"misattributed="
        f"{duration_evaluations['timeline']['total_misattributed_time_sec']:.1f}s "
        f"({duration_evaluations['timeline']['total_misattributed_time_percent_of_video']:.1f}% "
        "of video)"
    )
    if not duration_evaluations["timeline"]["predicted_duration_sum_check_passed"]:
        raise ValueError("Predicted timeline durations failed the <1s sum assertion")
    print(f"Saved confusion heatmaps under: {results_path}")
    print(f"Saved alignment report: {report_path}")
    print(f"Saved bed-event scorecard: {event_report_path}")
    print(f"Saved duration scorecard: {duration_report_path}")


if __name__ == "__main__":
    main()
