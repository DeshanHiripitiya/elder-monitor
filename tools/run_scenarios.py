"""Run priority alert decisions on explicitly synthetic JSON scenarios."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.decisions import decision_level, run_alerts
from src.frame_sampler import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config_demo.yaml"))
    parser.add_argument("--scenarios", type=Path, default=Path("scenarios"))
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    scenario_dir = args.scenarios
    if not scenario_dir.is_absolute():
        scenario_dir = config_path.parent / scenario_dir
    paths = sorted(scenario_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No scenario JSON files found in {scenario_dir}")

    for path in paths:
        scenario: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if scenario.get("synthetic") is not True:
            raise ValueError(f"Scenario must explicitly set synthetic=true: {path}")
        decisions = run_alerts(
            scenario["segments"],
            scenario["events"],
            config["alerts"],
            video_duration=float(scenario["video_duration"]),
        )
        overall = decision_level(decisions)
        print(f"SYNTHETIC: {scenario['name']}")
        print(f"  {scenario['description']}")
        print(f"  overall_decision: {overall}")
        if not decisions:
            print("  no decision triggered")
        for item in decisions:
            print(
                f"  t={item['t']:.1f}s {item['decision']} "
                f"{item['rule']}: {item['reason']}"
            )


if __name__ == "__main__":
    main()
