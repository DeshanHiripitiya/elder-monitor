"""Interactively select and save a mattress or chair polygon."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a mapping: {config_path}")
    return config


def save_polygon(
    config_path: Path,
    config: dict[str, Any],
    points: list[list[int]],
    zone: str,
) -> None:
    config[f"{zone}_polygon"] = points
    with config_path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file, sort_keys=False)


def draw_polygon(frame: Any, points: list[list[int]]) -> Any:
    overlay = frame.copy()
    if not points:
        return overlay

    coordinates = [(point[0], point[1]) for point in points]
    for point in coordinates:
        cv2.circle(overlay, point, 5, (0, 0, 255), -1)
    if len(coordinates) >= 2:
        cv2.polylines(
            overlay,
            [np.array(coordinates, dtype="int32")],
            isClosed=len(coordinates) >= 3,
            color=(0, 255, 0),
            thickness=2,
        )
    return overlay


def display_scale(frame: Any) -> float:
    height, width = frame.shape[:2]
    return min(
        1.0,
        MAX_DISPLAY_WIDTH / width,
        MAX_DISPLAY_HEIGHT / height,
    )


def select_polygon(frame: Any, zone: str) -> list[list[int]]:
    points: list[list[int]] = []
    scale = display_scale(frame)
    window_name = zone

    def on_click(event: int, x: int, y: int, _flags: int, _userdata: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append(
                [
                    min(frame.shape[1] - 1, max(0, round(x / scale))),
                    min(frame.shape[0] - 1, max(0, round(y / scale))),
                ]
            )

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    display_width = max(1, round(frame.shape[1] * scale))
    display_height = max(1, round(frame.shape[0] * scale))
    cv2.resizeWindow(window_name, display_width, display_height)
    cv2.setMouseCallback(window_name, on_click)
    print(
        f"Click the {zone} corners in order. Press 's' to save or 'q' to cancel. "
        f"Display scale: {scale:.3f}"
    )

    while True:
        display_frame = draw_polygon(frame, points)
        if scale < 1.0:
            display_frame = cv2.resize(
                display_frame,
                (display_width, display_height),
                interpolation=cv2.INTER_AREA,
            )
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("s"):
            if len(points) < 3:
                print("Select at least 3 mattress corners before saving.")
                continue
            return points
        if key == ord("q") or key == 27:
            raise KeyboardInterrupt("Polygon selection cancelled")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--zone",
        choices=("bed", "chair"),
        default="bed",
        help="Zone to select and save",
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Video to use instead of the path configured in config.yaml",
    )
    parser.add_argument(
        "--frame-time",
        type=float,
        default=0.0,
        help="Timestamp in seconds for the frame used for selection (default: 0)",
    )
    parser.add_argument("--output-overlay", type=Path)
    args = parser.parse_args()
    if args.frame_time < 0:
        parser.error("--frame-time must be zero or greater")

    config_path = args.config.resolve()
    config = load_config(config_path)
    video_path = args.video if args.video is not None else Path(config["video"])
    if not video_path.is_absolute():
        video_path = config_path.parent / video_path
    if not video_path.exists():
        available_videos = sorted(config_path.parent.glob("data/*.mp4"))
        available_message = ", ".join(str(path.relative_to(config_path.parent)) for path in available_videos)
        hint = f" Available videos: {available_message}." if available_message else ""
        raise FileNotFoundError(f"Video does not exist: {video_path}.{hint}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    video_fps = capture.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        capture.release()
        raise RuntimeError(f"Video has an invalid FPS value: {video_fps}")
    frame_index = round(args.frame_time * video_fps)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(
            f"Unable to read frame at {args.frame_time:.3f}s "
            f"(frame {frame_index}) from: {video_path}"
        )

    try:
        points = select_polygon(frame, args.zone)
    finally:
        cv2.destroyAllWindows()

    save_polygon(config_path, config, points, args.zone)
    overlay_path = args.output_overlay or Path(f"data/{args.zone}_polygon_overlay.png")
    if not overlay_path.is_absolute():
        overlay_path = config_path.parent / overlay_path
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(overlay_path), draw_polygon(frame, points)):
        raise RuntimeError(f"Unable to save polygon overlay: {overlay_path}")
    print(f"Saved {len(points)} points to {config_path}")
    print(f"Saved {args.frame_time:.3f}s overlay to {overlay_path}")


if __name__ == "__main__":
    main()
