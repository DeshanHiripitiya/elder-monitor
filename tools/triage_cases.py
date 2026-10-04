"""List timeline windows requiring bounded agent review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import build_cases
from src.frame_sampler import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--segments", type=Path, default=Path("data/timeline.parquet"))
    parser.add_argument("--features", type=Path, default=Path("data/features.parquet"))
    parser.add_argument("--events", type=Path, default=Path("data/events.json"))
    parser.add_argument("--output", type=Path, help="Optional JSON path for the case list")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else config_path.parent / path

    segment_path = resolve(args.segments)
    feature_path = resolve(args.features)
    event_path = resolve(args.events)
    segments = pd.read_parquet(segment_path).to_dict("records")
    features = pd.read_parquet(feature_path)
    events: list[dict[str, Any]] = json.loads(
        event_path.read_text(encoding="utf-8")
    )
    cases = build_cases(segments, features, events, config)

    agent_config = config["agent"]
    print(f"Agent triage cases: {len(cases)}")
    print(
        "Budgets: "
        f"{agent_config['max_steps_per_case']} steps/case, "
        f"{agent_config['max_vlm_calls_per_case']} VLM calls/case, "
        f"{agent_config['max_vlm_calls_total']} VLM calls total, "
        f"{agent_config['vlm_frames']} frames/VLM call"
    )
    for case in cases:
        trigger_names = ", ".join(
            sorted({str(trigger["kind"]) for trigger in case["triggers"]})
        )
        print(
            f"{case['case_id']} {case['start']:.1f}-{case['end']:.1f}s "
            f"({case['duration_sec']:.1f}s): {trigger_names}"
        )
        for trigger in case["triggers"]:
            print(f"  - {trigger['kind']}: {trigger['evidence']}")
    print("This trigger-inventory step makes no VLM calls.")

    if args.output is not None:
        output_path = resolve(args.output)
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing case output: {output_path}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {len(cases)} cases to {output_path}")


if __name__ == "__main__":
    main()
