"""Cached-data tools used by bounded agent triage."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import pandas as pd

from src.frame_sampler import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class VLMFrame:
    """An annotated JPEG frame and its source-video timestamp."""

    timestamp_sec: float
    jpeg_bytes: bytes


class VLMClient(Protocol):
    """Provider-neutral interface for a vision-language model."""

    @property
    def model(self) -> str: ...

    def describe(self, frames: list[VLMFrame], prompt: str) -> str: ...


class MockVLM:
    """Deterministic JSON-returning client for tests and offline smoke checks."""

    model = "mock-vlm-v1"

    def describe(self, frames: list[VLMFrame], prompt: str) -> str:
        return json.dumps(
            {
                "patient_visible": False,
                "location": "not_visible",
                "posture": "unknown",
                "other_person_present": False,
                "confidence": 0,
                "evidence": "Mock response; no model inference was performed.",
            }
        )


def _resolve_path(path: str | Path, config_path: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else config_path.parent / candidate


def _load_cached_table(
    table: pd.DataFrame | None,
    path: str | Path,
    config_path: Path,
) -> pd.DataFrame:
    if table is not None:
        return table
    return pd.read_parquet(_resolve_path(path, config_path))


def _numeric_summary(values: pd.Series) -> dict[str, float | None]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": float(clean.mean()),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def _feature_window(
    features: pd.DataFrame,
    t0: float,
    t1: float,
) -> pd.DataFrame:
    if t1 < t0:
        raise ValueError("t1 must be greater than or equal to t0")
    if "t" not in features:
        raise ValueError("Feature data must contain a 't' timestamp column")
    return features[(features["t"] >= t0) & (features["t"] <= t1)]


def get_state_history(
    t0: float,
    t1: float,
    *,
    timeline: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
    timeline_path: str | Path = "data/processed/timeline.parquet",
    features_path: str | Path = "data/processed/features.parquet",
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Summarize cached timeline segments and bed distance in [t0, t1]."""
    config_file = Path(config_path).resolve()
    timeline_data = _load_cached_table(timeline, timeline_path, config_file)
    feature_data = _load_cached_table(features, features_path, config_file)
    if t1 < t0:
        raise ValueError("t1 must be greater than or equal to t0")

    overlaps = timeline_data[
        (timeline_data["end"] > t0) & (timeline_data["start"] < t1)
    ].sort_values("start")
    segments = [
        {
            "start": max(float(row.start), float(t0)),
            "end": min(float(row.end), float(t1)),
            "state": str(row.state),
            "conf": float(row.mean_confidence),
        }
        for row in overlaps.itertuples(index=False)
    ]
    feature_window = _feature_window(feature_data, float(t0), float(t1))
    bed_dist = _numeric_summary(feature_window["dist_to_bed"])
    return {
        "segments": segments,
        "bed_dist": [bed_dist["min"], bed_dist["max"]],
    }


def get_pose_features(
    t0: float,
    t1: float,
    *,
    features: pd.DataFrame | None = None,
    features_path: str | Path = "data/processed/features.parquet",
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Summarize cached pose features in [t0, t1], without returning arrays."""
    config_file = Path(config_path).resolve()
    feature_data = _load_cached_table(features, features_path, config_file)
    window = _feature_window(feature_data, float(t0), float(t1))
    torso = _numeric_summary(window["torso_angle"])
    in_bed = _numeric_summary(window["kp_in_bed_frac"])
    speed = _numeric_summary(window["speed"])
    visibility = _numeric_summary(window["vis"])
    distance = pd.to_numeric(window["dist_to_bed"], errors="coerce").dropna()
    presence = (
        window["present"].fillna(False).astype(bool)
        if not window.empty
        else pd.Series(dtype=bool)
    )
    return {
        "torso_angle": torso,
        "kp_in_bed_frac": in_bed["mean"],
        "dist_to_bed": {
            "start": float(distance.iloc[0]) if not distance.empty else None,
            "end": float(distance.iloc[-1]) if not distance.empty else None,
        },
        "speed_mean": speed["mean"],
        "present_frac": float(presence.mean()) if not presence.empty else None,
        "vis_mean": visibility["mean"],
    }


def _video_duration(video_path: Path) -> float:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0 or frame_count <= 0:
            raise RuntimeError(f"Video has invalid duration metadata: {video_path}")
        return float(frame_count / fps)
    finally:
        capture.release()


def extend_window(
    t0: float,
    t1: float,
    direction: str,
    seconds: float = 10,
    *,
    video_end: float | None = None,
    video_path: str | Path | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
    max_window_sec: float = 60,
) -> tuple[float, float]:
    """Extend a window, clipping to the video and a maximum total span."""
    start = float(t0)
    end = float(t1)
    amount = float(seconds)
    limit = float(max_window_sec)
    if not all(math.isfinite(value) for value in (start, end, amount, limit)):
        raise ValueError("Window values must be finite numbers")
    if start < 0 or end < start:
        raise ValueError("Require 0 <= t0 <= t1")
    if amount < 0 or limit <= 0:
        raise ValueError("seconds must be nonnegative and max_window_sec positive")
    if end - start > limit:
        raise ValueError("Original window exceeds max_window_sec")
    if video_end is None:
        config_file = Path(config_path).resolve()
        config = load_config(config_file)
        source = Path(video_path) if video_path is not None else Path(config["video"])
        video_end = _video_duration(_resolve_path(source, config_file))
    video_end = float(video_end)
    if not math.isfinite(video_end) or video_end < 0:
        raise ValueError("video_end must be a finite nonnegative number")
    if end > video_end:
        raise ValueError("Window ends beyond the video duration")

    capacity = max(0.0, limit - (end - start))
    if direction == "backward":
        amount = min(amount, capacity)
        start = max(0.0, start - amount)
    elif direction == "forward":
        amount = min(amount, capacity)
        end = min(video_end, end + amount)
    elif direction == "both":
        extension = min(2 * amount, capacity)
        left = min(extension / 2, start)
        right = min(extension / 2, video_end - end)
        remaining = extension - left - right
        if remaining > 0:
            extra_left = min(remaining, start - left)
            left += extra_left
            remaining -= extra_left
            right += min(remaining, video_end - end - right)
        start -= left
        end += right
    else:
        raise ValueError("direction must be 'backward', 'forward', or 'both'")
    return start, end


def _annotated_frames(
    video_path: Path,
    t0: float,
    t1: float,
    frame_count: int,
    bed_polygon: list[list[float]],
    brightness_factor: float,
) -> list[VLMFrame]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_total <= 0:
            raise RuntimeError(f"Video has invalid frame metadata: {video_path}")
        duration = frame_total / fps
        if t0 < 0 or t1 < t0 or t1 > duration:
            raise ValueError(f"Clip window must be within 0..{duration:.3f} seconds")
        if frame_count < 1:
            raise ValueError("agent.vlm_frames must be at least 1")

        times = [float(t0), float(t1)]
        times.extend(
            float(t0 + (t1 - t0) * index / (frame_count + 1))
            for index in range(1, frame_count + 1)
        )
        polygon = np.asarray(bed_polygon, dtype=np.int32)
        result: list[VLMFrame] = []
        frame_indices = sorted(
            {
                min(frame_total - 1, max(0, round(timestamp * fps)))
                for timestamp in times
            }
        )
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Unable to read video frame {frame_index}"
                )
            actual_time = frame_index / fps
            if len(polygon) >= 3:
                cv2.polylines(
                    frame,
                    [polygon.reshape((-1, 1, 2))],
                    isClosed=True,
                    color=(0, 0, 255),
                    thickness=max(2, frame.shape[1] // 800),
                )
            if brightness_factor < 1:
                frame = cv2.convertScaleAbs(frame, alpha=brightness_factor, beta=0)
            cv2.putText(
                frame,
                f"t={actual_time:.2f}s",
                (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            encoded, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            if not encoded:
                raise RuntimeError(f"Unable to encode annotated frame {frame_index}")
            result.append(VLMFrame(actual_time, buffer.tobytes()))
        return result
    finally:
        capture.release()


def _cache_key(
    video_path: Path,
    t0: float,
    t1: float,
    question: str,
    model: str,
    frame_count: int,
    bed_polygon: list[list[float]],
    brightness_factor: float,
) -> str:
    stat = video_path.stat()
    identity = {
        "video_path": str(video_path.resolve()),
        "video_size": stat.st_size,
        "video_mtime_ns": stat.st_mtime_ns,
        "t0": float(t0),
        "t1": float(t1),
        "question": question,
        "model": model,
        "vlm_frames": frame_count,
        "bed_polygon": bed_polygon,
        "brightness_factor": brightness_factor,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def vlm_describe_clip(
    t0: float,
    t1: float,
    question: str,
    *,
    client: VLMClient,
    config: dict[str, Any] | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
    video_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    brightness_factor: float = 1.0,
) -> dict[str, Any]:
    """Describe cached source-video frames via a validated, cacheable VLM call."""
    from src.vlm import (
        VLMUnavailableError,
        parse_and_validate_vlm_json,
        unknown_vlm_result,
    )

    config_file = Path(config_path).resolve()
    if config is None:
        config = load_config(config_file)
    source = Path(video_path) if video_path is not None else Path(config["video"])
    source = _resolve_path(source, config_file).resolve()
    frame_count = int(config["agent"]["vlm_frames"])
    bed_polygon = config.get("bed_polygon", [])
    brightness_factor = float(brightness_factor)
    if not math.isfinite(brightness_factor) or not 0 < brightness_factor <= 1:
        raise ValueError("brightness_factor must be greater than 0 and at most 1")
    cache_root = (
        Path(cache_dir)
        if cache_dir is not None
        else PROJECT_ROOT / ".cache" / "agent_vlm"
    )
    if not cache_root.is_absolute():
        cache_root = config_file.parent / cache_root
    cache_key = _cache_key(
        source,
        float(t0),
        float(t1),
        question,
        client.model,
        frame_count,
        bed_polygon,
        brightness_factor,
    )
    cache_file = cache_root / f"{cache_key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    frames = _annotated_frames(
        source,
        float(t0),
        float(t1),
        frame_count,
        bed_polygon,
        brightness_factor,
    )
    frame_count_text = (
        f"{len(frames)} frame{'s' if len(frames) != 1 else ''}"
    )
    prompt = (
        f"These are {frame_count_text} from a fixed camera, in time order, "
        "with timestamps. The red polygon is the bed. Describe ONLY the "
        "elderly patient. Ignore any other person and do not describe them. "
        "Report whether another person is present in the other_person_present "
        "field.\n"
        f"Answer this narrow question: {question}\n"
        "Reply with JSON only, using exactly these fields and types:\n"
        '{"patient_visible": false, "location": "not_visible", '
        '"posture": "unknown", "other_person_present": false, '
        '"confidence": 0.0, "evidence": "One short sentence."}\n'
        'location must be one of "on_bed", "beside_bed", "floor", "chair", '
        '"elsewhere", or "not_visible". posture must be one of "lying", '
        '"sitting", "standing", "walking", or "unknown". If you cannot tell, '
        'use "unknown" or "not_visible". Do not guess. Use booleans for the '
        "boolean fields and a numeric confidence from 0 to 1."
    )
    response: dict[str, Any] | None = None
    validation_error = ""
    for attempt in range(2):
        attempt_prompt = prompt
        if attempt:
            attempt_prompt += (
                "\nYour previous response failed JSON/schema validation: "
                f"{validation_error}. Retry once. Return only the exact JSON object."
            )
        try:
            response = parse_and_validate_vlm_json(
                client.describe(frames, attempt_prompt)
            )
            break
        except ValueError as error:
            validation_error = str(error)
        except VLMUnavailableError as error:
            return {
                "status": "vlm_unavailable",
                "result": unknown_vlm_result(),
                "error": str(error),
            }

    if response is None:
        return {
            "status": "invalid_response",
            "result": unknown_vlm_result(),
            "error": validation_error,
        }
    result = {"status": "ok", "result": response}

    cache_root.mkdir(parents=True, exist_ok=True)
    temp_file = cache_file.with_suffix(f".{id(result)}.tmp")
    try:
        temp_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temp_file.replace(cache_file)
    finally:
        if temp_file.exists():
            temp_file.unlink()
    return result
