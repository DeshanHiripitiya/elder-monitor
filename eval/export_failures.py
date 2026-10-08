"""Export representative, data-selected evaluation failure examples."""

from __future__ import annotations

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

from eval.run_all import (
    UNKNOWN,
    _samples_to_segments,
    find_mismatch_intervals,
    load_ground_truth,
    to_grid,
)
from src.frame_sampler import load_config
from tools.render_debug_video import COCO_SKELETON


OUTPUT_SIZE = (960, 540)
MIN_KEYPOINT_CONFIDENCE = 0.25


def _state_at(segments: list[dict[str, Any]], timestamp: float) -> str:
    starts = np.asarray([float(segment["start"]) for segment in segments])
    index = int(np.searchsorted(starts, timestamp, side="right") - 1)
    if index < 0:
        return UNKNOWN
    segment = segments[index]
    if timestamp >= float(segment["end"]):
        return UNKNOWN
    return str(segment["state"])


def _feature_summary(
    features: pd.DataFrame,
    start: float,
    end: float,
) -> dict[str, float | None]:
    window = features[(features["t"] >= start) & (features["t"] < end)]
    summary: dict[str, float | None] = {}
    for column in (
        "torso_angle",
        "hip_height_ratio",
        "hip_knee_ratio",
        "dist_to_bed",
        "speed",
        "vis",
        "kp_in_bed_frac",
    ):
        if column not in window:
            continue
        values = pd.to_numeric(window[column], errors="coerce").dropna()
        summary[column] = float(values.mean()) if not values.empty else None
    if "present" in window:
        summary["present_fraction"] = (
            float(window["present"].fillna(False).astype(bool).mean())
            if not window.empty
            else None
        )
    return summary


def _draw_frame(
    frame: np.ndarray,
    timestamp: float,
    case: dict[str, Any],
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    pose_times: np.ndarray,
    poses: list[Any],
    polygon_points: np.ndarray,
) -> np.ndarray:
    height, width = frame.shape[:2]
    output = frame.copy()
    polygon = polygon_points.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(output, [polygon], True, (0, 0, 255), 4, cv2.LINE_AA)

    pose_index = int(np.searchsorted(pose_times, timestamp))
    candidates = [
        index
        for index in (pose_index - 1, pose_index)
        if 0 <= index < len(pose_times)
    ]
    if candidates:
        nearest = min(candidates, key=lambda index: abs(pose_times[index] - timestamp))
        if abs(float(pose_times[nearest]) - timestamp) <= 0.12:
            raw_points = poses[nearest]
            if raw_points is not None:
                points = np.stack(raw_points).astype(float)
                for first_index, second_index in COCO_SKELETON:
                    if max(first_index, second_index) >= len(points):
                        continue
                    first, second = points[first_index], points[second_index]
                    if (
                        len(first) >= 3
                        and len(second) >= 3
                        and np.isfinite(first[:2]).all()
                        and np.isfinite(second[:2]).all()
                        and first[2] >= MIN_KEYPOINT_CONFIDENCE
                        and second[2] >= MIN_KEYPOINT_CONFIDENCE
                    ):
                        cv2.line(
                            output,
                            tuple(np.rint(first[:2]).astype(int)),
                            tuple(np.rint(second[:2]).astype(int)),
                            (0, 255, 0),
                            3,
                            cv2.LINE_AA,
                        )
                for point in points:
                    if (
                        len(point) >= 3
                        and np.isfinite(point[:2]).all()
                        and point[2] >= MIN_KEYPOINT_CONFIDENCE
                    ):
                        cv2.circle(
                            output,
                            tuple(np.rint(point[:2]).astype(int)),
                            5,
                            (0, 255, 0),
                            -1,
                            cv2.LINE_AA,
                        )

    expected = _state_at(ground_truth, timestamp)
    predicted = _state_at(predictions, timestamp)
    cv2.rectangle(output, (0, 0), (width, 108), (0, 0, 0), -1)
    cv2.putText(
        output,
        case["title"],
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"t={timestamp:.2f}s  PRED: {predicted}",
        (18, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"GT: {expected}",
        (18, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return cv2.resize(output, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)


def _read_annotated_frame(
    capture: cv2.VideoCapture,
    timestamp: float,
    case: dict[str, Any],
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    pose_times: np.ndarray,
    poses: list[Any],
    polygon: np.ndarray,
) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Unable to read source frame at {timestamp:.2f}s")
    return _draw_frame(
        frame,
        timestamp,
        case,
        ground_truth,
        predictions,
        pose_times,
        poses,
        polygon,
    )


def _write_case(
    case: dict[str, Any],
    output_dir: Path,
    video_path: Path,
    fps: float,
    ground_truth: list[dict[str, Any]],
    modes: dict[str, list[dict[str, Any]]],
    pose_times: np.ndarray,
    poses: list[Any],
    polygon: np.ndarray,
) -> None:
    predictions = modes[case["mode"]]
    start = float(case["clip_start_sec"])
    end = float(case["clip_end_sec"])
    if end - start < 10 or end - start > 20:
        raise ValueError(f"Failure clip must be 10-20s: {case['name']}")

    frame_capture = cv2.VideoCapture(str(video_path))
    if not frame_capture.isOpened():
        raise RuntimeError(f"Unable to open source video: {video_path}")
    try:
        times = np.linspace(
            float(case["frames_start_sec"]),
            float(case["frames_end_sec"]),
            4,
        )
        thumbnails = [
            _read_annotated_frame(
                frame_capture,
                float(timestamp),
                case,
                ground_truth,
                predictions,
                pose_times,
                poses,
                polygon,
            )
            for timestamp in times
        ]
    finally:
        frame_capture.release()
    strip = cv2.hconcat(thumbnails)
    strip_path = output_dir / f"{case['name']}_frames.png"
    if not cv2.imwrite(str(strip_path), strip):
        raise RuntimeError(f"Unable to write frame strip: {strip_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    clip_path = output_dir / f"{case['name']}.mp4"
    writer = cv2.VideoWriter(
        str(clip_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        OUTPUT_SIZE,
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to create failure clip: {clip_path}")
    frame_index = max(0, int(round(start * fps)))
    final_frame = int(round(end * fps))
    try:
        while frame_index < final_frame:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / fps
            writer.write(
                _draw_frame(
                    frame,
                    timestamp,
                    case,
                    ground_truth,
                    predictions,
                    pose_times,
                    poses,
                    polygon,
                )
            )
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    if not clip_path.is_file() or clip_path.stat().st_size == 0:
        raise RuntimeError(f"Failure clip was not written: {clip_path}")
    case["frame_strip"] = str(strip_path.relative_to(PROJECT_ROOT))
    case["clip"] = str(clip_path.relative_to(PROJECT_ROOT))


def main() -> None:
    results_dir = PROJECT_ROOT / "results"
    results_path = results_dir / "alignment.json"
    if not results_path.is_file():
        raise FileNotFoundError(
            "Run `python -m eval.run_all` before exporting failure examples"
        )
    report = json.loads(results_path.read_text(encoding="utf-8"))
    config_path = PROJECT_ROOT / "config.yaml"
    config = load_config(config_path)
    video_path = Path(config["video"])
    if not video_path.is_absolute():
        video_path = config_path.parent / video_path
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    capture.release()
    if fps <= 0 or duration <= 0:
        raise RuntimeError(f"Source video has invalid timing metadata: {video_path}")

    ground_truth = load_ground_truth(PROJECT_ROOT / "data/raw/gt.csv")
    timeline_frame = pd.read_parquet(PROJECT_ROOT / "data/processed/timeline.parquet")
    timeline = timeline_frame.to_dict("records")
    score_frame = pd.read_parquet(PROJECT_ROOT / "data/processed/state_scores.parquet")
    raw_segments = _samples_to_segments(score_frame, "argmax_state", duration)
    smoothed_frame = pd.read_parquet(
        PROJECT_ROOT / "data/processed/smoothed_states.parquet"
    )
    smoothed_segments = _samples_to_segments(
        smoothed_frame,
        "smoothed_state",
        duration,
    )
    modes = {
        "timeline": timeline,
        "raw_argmax": raw_segments,
        "smoothed": smoothed_segments,
    }

    step = float(report["grid"]["step_sec"])
    _, truth_grid = to_grid(ground_truth, duration, step)
    _, timeline_grid = to_grid(timeline, duration, step)
    confusion_mask = (
        (truth_grid == "STANDING") & (timeline_grid == "WALKING")
    )
    confusion_runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for index in range(len(confusion_mask) + 1):
        active = index < len(confusion_mask) and bool(confusion_mask[index])
        if active and run_start is None:
            run_start = index
        elif not active and run_start is not None:
            confusion_runs.append((run_start, index))
            run_start = None
    if not confusion_runs:
        raise ValueError("No measured STANDING-to-WALKING failure interval exists")
    confusion_start, confusion_end = max(
        confusion_runs,
        key=lambda run: (run[1] - run[0], -run[0]),
    )
    error_start = confusion_start * step
    error_end = confusion_end * step
    confusion_case = {
        "name": "standing_predicted_walking",
        "title": "State confusion: standing predicted as walking",
        "category": "state_confusion",
        "mode": "timeline",
        "error_start_sec": error_start,
        "error_end_sec": error_end,
        "frames_start_sec": error_start,
        "frames_end_sec": error_end,
        "clip_start_sec": max(0.0, error_start - 5.0),
        "clip_end_sec": min(duration, error_end + 5.0),
        "impact": (
            f"GT STANDING was predicted WALKING for "
            f"{(confusion_end - confusion_start) * step:.1f}s."
        ),
        "features": _feature_summary(
            pd.read_parquet(PROJECT_ROOT / "data/processed/features.parquet"),
            error_start,
            error_end,
        ),
    }

    event_report = report["bed_event_evaluation"]["modes"]["predicted_segments"]
    missed_exits = event_report["tolerances"]["10"]["bed_exit"][
        "nearest_candidates_for_missed_ground_truth"
    ]
    event_failure = next(
        (
            candidate
            for candidate in missed_exits
            if candidate.get("nearest_prediction_available")
        ),
        None,
    )
    if event_failure is None:
        raise ValueError("No measured missed-exit candidate is available to export")
    gt_start = float(event_failure["ground_truth_start_time"])
    gt_confirm = float(event_failure["ground_truth_confirmed_time"])
    pred_start = float(event_failure["predicted_start_time"])
    pred_confirm = float(event_failure["predicted_confirmed_time"])
    clip_end = min(duration, max(gt_confirm, pred_confirm))
    clip_start = max(0.0, clip_end - 20.0)
    event_case = {
        "name": "early_bed_exit_confirmation",
        "title": "Bed-exit detector confirmed before GT",
        "category": "event_timing_error",
        "mode": "timeline",
        "error_start_sec": min(gt_start, pred_start),
        "error_end_sec": max(gt_confirm, pred_confirm),
        "frames_start_sec": min(gt_start, pred_start),
        "frames_end_sec": max(gt_confirm, pred_confirm),
        "clip_start_sec": clip_start,
        "clip_end_sec": clip_end,
        "impact": (
            f"Predicted confirmation {pred_confirm:.1f}s versus GT "
            f"{gt_confirm:.1f}s ({pred_confirm - gt_confirm:+.1f}s)."
        ),
        "timing": event_failure,
    }

    _, truth_fine = to_grid(ground_truth, duration, 0.2)
    raw_fine_segments = raw_segments
    _, raw_fine = to_grid(raw_fine_segments, duration, 0.2)
    scored_raw_unknown = np.where(
        (truth_fine != UNKNOWN) & (raw_fine == UNKNOWN),
        "ABSTAINED",
        truth_fine,
    )
    unknown_intervals = find_mismatch_intervals(
        truth_fine,
        scored_raw_unknown,
        0.2,
    )
    unknown_interval = next(
        (
            interval
            for interval in unknown_intervals
            if interval["state_pairs"]
            and all(
                pair["prediction"] == "ABSTAINED"
                for pair in interval["state_pairs"]
            )
        ),
        None,
    )
    if unknown_interval is None:
        raise ValueError("No UNKNOWN abstention interval against labeled GT exists")
    unknown_start = float(unknown_interval["start"])
    unknown_end = float(unknown_interval["end"])
    unknown_case = {
        "name": "raw_argmax_unknown_abstention",
        "title": "Raw argmax abstained during labeled ground truth",
        "category": "abstention",
        "mode": "raw_argmax",
        "error_start_sec": unknown_start,
        "error_end_sec": unknown_end,
        "frames_start_sec": unknown_start,
        "frames_end_sec": unknown_end,
        "clip_start_sec": max(0.0, unknown_start - 7.0),
        "clip_end_sec": min(duration, unknown_start + 8.0),
        "impact": (
            f"Raw argmax emitted UNKNOWN for {unknown_end - unknown_start:.1f}s "
            "while GT was labeled."
        ),
        "features": _feature_summary(
            pd.read_parquet(PROJECT_ROOT / "data/processed/features.parquet"),
            unknown_start,
            unknown_end,
        ),
    }
    cases = [confusion_case, event_case, unknown_case]

    pose_frame = pd.read_parquet(PROJECT_ROOT / "data/processed/raw_pose.parquet")
    if not {"t", "keypoints"}.issubset(pose_frame.columns):
        raise ValueError("raw_pose.parquet must contain t and keypoints")
    pose_frame = pose_frame.sort_values("t")
    pose_times = pose_frame["t"].to_numpy(dtype=float)
    poses = pose_frame["keypoints"].tolist()
    polygon = np.asarray(config["bed_polygon"], dtype=float)
    output_dir = results_dir / "failure_cases"
    output_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        _write_case(
            case,
            output_dir,
            video_path,
            fps,
            ground_truth,
            modes,
            pose_times,
            poses,
            polygon,
        )
        case["features"] = case.get("features") or {}
    summary_path = output_dir / "case_summary.json"
    summary_path.write_text(
        json.dumps(cases, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Exported {len(cases)} failure examples to {output_dir}")
    for case in cases:
        print(
            f"- {case['name']}: {case['error_start_sec']:.1f}-"
            f"{case['error_end_sec']:.1f}s ({case['category']})"
        )


if __name__ == "__main__":
    main()
