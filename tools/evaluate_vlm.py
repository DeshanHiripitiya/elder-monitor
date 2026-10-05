"""Evaluate constrained VLM posture descriptions on labelled real-video windows."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent_tools import MockVLM, vlm_describe_clip
from src.frame_sampler import load_config
from src.vlm import OllamaVLMClient


def _time_seconds(value: str) -> float:
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        return float(parts[0] * 60 + parts[1])
    if len(parts) == 3 and parts[2] == 0:
        return float(parts[0] * 60 + parts[1])
    if len(parts) == 3:
        return float(parts[0] * 3600 + parts[1] * 60 + parts[2])
    raise ValueError(f"Unsupported ground-truth time value: {value}")


def _verify_window_label(
    ground_truth: pd.DataFrame,
    window: dict[str, Any],
) -> None:
    overlap_by_state: Counter[str] = Counter()
    start = float(window["start"])
    end = float(window["end"])
    for row in ground_truth.itertuples(index=False):
        row_start = _time_seconds(str(row.start))
        row_end = _time_seconds(str(row.end))
        overlap = max(0.0, min(end, row_end) - max(start, row_start))
        if overlap:
            overlap_by_state[str(row.state)] += overlap
    if not overlap_by_state:
        raise ValueError(f"No ground-truth labels overlap window {window['id']}")
    expected_state, expected_duration = overlap_by_state.most_common(1)[0]
    if (
        expected_state != window["ground_truth_state"]
        or expected_duration < end - start
    ):
        raise ValueError(
            f"Window {window['id']} must lie entirely in "
            f"{window['ground_truth_state']}; got {dict(overlap_by_state)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("scenarios/vlm_eval_windows.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/vlm_eval_results.json"),
    )
    parser.add_argument(
        "--mock-vlm",
        action="store_true",
        help="Run plumbing checks only; mock responses are excluded from accuracy",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    cases_path = args.cases
    if not cases_path.is_absolute():
        cases_path = config_path.parent / cases_path
    output_path = args.output
    if not output_path.is_absolute():
        output_path = config_path.parent / output_path
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing evaluation: {output_path}; "
            "use --overwrite"
        )

    raw_dir = config_path.parent / "data" / "raw"
    source_video = Path(config["video"])
    if not source_video.is_absolute():
        source_video = config_path.parent / source_video
    ground_truth_path = raw_dir / "gt.csv"
    if not source_video.exists():
        raise FileNotFoundError(f"Configured video does not exist: {source_video}")
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground-truth state labels do not exist: {ground_truth_path}")

    case_document = json.loads(cases_path.read_text(encoding="utf-8"))
    windows = case_document["windows"]
    agent_config = config["agent"]
    if len(windows) > int(agent_config["max_vlm_calls_total"]):
        raise ValueError("Evaluation window count exceeds max_vlm_calls_total")
    if len(windows) * 2 > int(agent_config["max_vlm_calls_total"]):
        raise ValueError(
            "Evaluation retry budget exceeds max_vlm_calls_total; reduce windows "
            "or raise the configured call budget"
        )
    ground_truth = pd.read_csv(ground_truth_path)

    if args.mock_vlm:
        client = MockVLM()
        print("Using MockVLM. This run validates plumbing only, not model accuracy.")
    else:
        if agent_config["vlm_provider"] != "ollama":
            raise ValueError(
                f"Unsupported configured local provider: {agent_config['vlm_provider']}"
            )
        client = OllamaVLMClient(
            base_url=str(agent_config["vlm_base_url"]),
            model=str(agent_config["vlm_model"]),
            timeout_sec=float(agent_config["vlm_timeout_sec"]),
            temperature=float(agent_config["vlm_temperature"]),
            context_length=int(agent_config["vlm_context_length"]),
        )
        print(f"Local VLM: {agent_config['vlm_model']} at {agent_config['vlm_base_url']}")
        print("Frames are sent only to the configured local Ollama endpoint.")

    results: list[dict[str, Any]] = []
    for window in windows:
        _verify_window_label(ground_truth, window)
        tags = list(window.get("tags", []))
        question = (
            "Identify the elderly patient and report their visible posture and "
            f"location. Review tags: {', '.join(tags) if tags else 'standard posture'}."
        )
        described = vlm_describe_clip(
            float(window["start"]),
            float(window["end"]),
            question,
            client=client,
            config=config,
            config_path=config_path,
            video_path=source_video,
            brightness_factor=float(window.get("brightness_factor", 1.0)),
        )
        actual_posture = described["result"]["posture"]
        scored = described["status"] == "ok" and not args.mock_vlm
        item = {
            "id": window["id"],
            "start": float(window["start"]),
            "end": float(window["end"]),
            "ground_truth_state": window["ground_truth_state"],
            "expected_posture": window["expected_posture"],
            "tags": tags,
            "brightness_factor": float(window.get("brightness_factor", 1.0)),
            "status": described["status"],
            "vlm_result": described["result"],
            "scored": scored,
            "correct": (
                actual_posture == window["expected_posture"] if scored else None
            ),
        }
        if "error" in described:
            item["error"] = described["error"]
        results.append(item)
        score = (
            "NOT SCORED (mock)"
            if args.mock_vlm
            else (
                "NOT AVAILABLE"
                if described["status"] != "ok"
                else ("correct" if item["correct"] else "incorrect")
            )
        )
        print(
            f"{window['id']} {window['start']:.1f}-{window['end']:.1f}s "
            f"GT={window['expected_posture']} VLM={actual_posture} "
            f"status={described['status']} {score}"
        )

    scored_results = [result for result in results if result["scored"]]
    correct_count = sum(bool(result["correct"]) for result in scored_results)
    unavailable_count = sum(
        result["status"] == "vlm_unavailable" for result in results
    )
    invalid_count = sum(
        result["status"] == "invalid_response" for result in results
    )
    report = {
        "model": client.model,
        "mock_run": args.mock_vlm,
        "ground_truth_file": str(ground_truth_path.relative_to(config_path.parent)),
        "windows_total": len(results),
        "windows_scored": len(scored_results),
        "posture_correct": correct_count,
        "posture_accuracy": (
            correct_count / len(scored_results) if scored_results else None
        ),
        "vlm_unavailable_count": unavailable_count,
        "invalid_response_count": invalid_count,
        "dim_light_note": (
            "The source clip has no independently annotated naturally dim-light "
            "interval; the tagged case is a brightness-reduced robustness check."
        ),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Posture accuracy: {correct_count}/{len(scored_results)}"
        if scored_results
        else "Posture accuracy: not measured (no successful non-mock VLM responses)"
    )
    print(
        f"Unavailable: {unavailable_count}; invalid response: {invalid_count}; "
        f"saved report: {output_path}"
    )


if __name__ == "__main__":
    main()
