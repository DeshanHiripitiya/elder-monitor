"""Apply Viterbi smoothing to per-frame state scores."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.score_states import STATES, ground_truth_labels
from src.frame_sampler import load_config


def transition_matrix(
    self_probability: float = 0.98,
    unknown_probability: float = 0.005,
) -> np.ndarray:
    """Build a row-normalized previous-state to next-state matrix."""
    state_count = len(STATES)
    transitions = np.zeros((state_count, state_count), dtype=float)
    unknown = STATES.index("UNKNOWN")

    allowed_moves = {
        "LYING_IN_BED": {"SITTING_ON_BED", "LYING_ON_FLOOR"},
        "SITTING_ON_BED": {"LYING_IN_BED", "STANDING", "SITTING_OUTSIDE_BED", "LYING_ON_FLOOR"},
        "SITTING_OUTSIDE_BED": {"STANDING", "SITTING_ON_BED", "WALKING", "LYING_ON_FLOOR"},
        "STANDING": {"SITTING_ON_BED", "SITTING_OUTSIDE_BED", "WALKING", "LYING_ON_FLOOR"},
        "WALKING": {"STANDING", "SITTING_OUTSIDE_BED", "LYING_ON_FLOOR"},
        "OUT_OF_BED": {"WALKING", "STANDING", "SITTING_OUTSIDE_BED", "LYING_ON_FLOOR"},
        "LYING_ON_FLOOR": {"STANDING", "WALKING", "UNKNOWN"},
        "UNKNOWN": set(STATES) - {"UNKNOWN"},
    }

    for source, destinations in allowed_moves.items():
        source_index = STATES.index(source)
        transitions[source_index, source_index] = self_probability
        if source_index != unknown:
            transitions[source_index, unknown] = unknown_probability
        remaining = 1.0 - self_probability - unknown_probability
        destinations = destinations - {source, "UNKNOWN"}
        if destinations:
            share = remaining / len(destinations)
            for destination in destinations:
                transitions[source_index, STATES.index(destination)] = share
        else:
            transitions[source_index, source_index] += remaining

    if np.any(transitions.sum(axis=1) <= 0):
        raise ValueError("Transition matrix contains an empty source state")
    return transitions / transitions.sum(axis=1, keepdims=True)


def viterbi(log_emit: np.ndarray, log_trans: np.ndarray) -> list[int]:
    """Return the highest-probability state path through the emissions."""
    if log_emit.ndim != 2:
        raise ValueError("log_emit must have shape (T, S)")
    if log_trans.shape != (log_emit.shape[1], log_emit.shape[1]):
        raise ValueError("log_trans must have shape (S, S)")

    frame_count, state_count = log_emit.shape
    dp = np.full((frame_count, state_count), -np.inf)
    back = np.zeros((frame_count, state_count), dtype=int)
    dp[0] = log_emit[0]
    for frame_index in range(1, frame_count):
        candidates = dp[frame_index - 1][:, None] + log_trans
        back[frame_index] = candidates.argmax(axis=0)
        dp[frame_index] = candidates.max(axis=0) + log_emit[frame_index]

    path = [int(dp[-1].argmax())]
    for frame_index in range(frame_count - 1, 0, -1):
        path.append(int(back[frame_index, path[-1]]))
    return path[::-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--input", type=Path, default=Path("data/state_scores.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/smoothed_states.parquet"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = args.output.resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    config_path = args.config.resolve()
    config = load_config(config_path)
    input_path = args.input if args.input.is_absolute() else config_path.parent / args.input
    scores = pd.read_parquet(input_path).sort_values("frame_idx").reset_index(drop=True)
    emissions = np.clip(scores[list(STATES)].to_numpy(dtype=float), 1e-12, 1.0)
    smoothing = config.get("smoothing", {})
    transition = transition_matrix(
        self_probability=float(smoothing.get("self_probability", 0.98)),
        unknown_probability=float(smoothing.get("unknown_probability", 0.005)),
    )
    path = viterbi(np.log(emissions), np.log(np.clip(transition, 1e-12, 1.0)))

    smoothed = scores[["frame_idx", "t", "present", "argmax_state"]].copy()
    smoothed["smoothed_state"] = [STATES[index] for index in path]
    smoothed["changed_from_argmax"] = (
        smoothed["smoothed_state"] != smoothed["argmax_state"]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    smoothed.to_parquet(output_path, index=False)

    print(f"Saved {len(smoothed)} smoothed rows to {output_path}")
    print("Transition matrix:")
    print(pd.DataFrame(transition, index=STATES, columns=STATES).round(4).to_string())
    print("Smoothed state counts:", smoothed["smoothed_state"].value_counts().to_dict())
    print(
        "Argmax changes:",
        int(smoothed["changed_from_argmax"].sum()),
        f"({smoothed['changed_from_argmax'].mean():.1%})",
    )

    gt_path = config_path.parent / "data/gt.csv"
    if gt_path.exists():
        truth = ground_truth_labels(gt_path, smoothed["t"])
        comparable = truth.notna()
        predicted = smoothed.loc[comparable, "smoothed_state"].to_numpy()
        actual = truth.loc[comparable].to_numpy()
        accuracy = float((predicted == actual).mean()) if len(actual) else 0.0
        print(
            f"Ground-truth smoothed accuracy: {accuracy:.1%} "
            f"({int((predicted == actual).sum())}/{len(actual)} frames)"
        )
    else:
        print(f"Ground-truth file not found; evaluation skipped: {gt_path}")


if __name__ == "__main__":
    main()
