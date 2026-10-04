import unittest

import pandas as pd

from src.events import detect_exits


CONFIG = {
    "in_bed_states": ["LYING_IN_BED", "SITTING_ON_BED"],
    "events": {
        "exit_away_sec": 4,
        "exit_away_dist": 0.8,
        "return_approach_dist": 0.8,
        "unknown_hold_sec": 4,
        "return_sit_sec": 2,
        "return_lie_sec": 5,
        "out_of_view_exit_sec": 10,
    },
}


def frames(states, distances=None, present=None):
    distances = distances or [0.0] * len(states)
    present = present or [True] * len(states)
    return pd.DataFrame(
        {
            "t": [float(index) for index in range(len(states))],
            "dist_to_bed": distances,
            "present": present,
        }
    )


def segments(states):
    return [
        {
            "start": float(index),
            "end": float(index + 1),
            "state": state,
            "mean_confidence": 0.9,
        }
        for index, state in enumerate(states)
    ]


class ExitDetectionTests(unittest.TestCase):
    def test_brief_stand_is_cancelled(self):
        result = detect_exits(
            segments(["LYING_IN_BED", "STANDING", "SITTING_ON_BED"]),
            frames(["LYING_IN_BED", "STANDING", "SITTING_ON_BED"]),
            CONFIG,
        )
        self.assertEqual(result, [])

    def test_away_duration_confirms_exit(self):
        states = ["LYING_IN_BED"] + ["STANDING"] * 5
        result = detect_exits(
            segments(states),
            frames(states, [0.0] + [1.0] * 5),
            CONFIG,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "bed_exit")
        self.assertEqual(result[0]["previous_state"], "LYING_IN_BED")

    def test_unknown_gap_does_not_cancel_candidate(self):
        states = ["LYING_IN_BED", "STANDING", "UNKNOWN", "UNKNOWN", "STANDING", "STANDING", "STANDING", "STANDING"]
        distances = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        result = detect_exits(segments(states), frames(states, distances), CONFIG)
        self.assertEqual(len(result), 1)

    def test_out_of_view_exit_is_low_confidence(self):
        states = ["LYING_IN_BED"] + ["OUT_OF_BED"] * 10
        present = [True] + [False] * 10
        result = detect_exits(
            segments(states),
            frames(states, present=present),
            CONFIG,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["confidence"], "low")
        self.assertEqual(result[0]["note"], "via_out_of_view")

    def test_exit_then_return_confirms_bed_return(self):
        states = (
            ["LYING_IN_BED"]
            + ["STANDING"] * 5
            + ["STANDING"]
            + ["SITTING_ON_BED"] * 2
            + ["LYING_IN_BED"] * 5
        )
        distances = [0.0] + [1.0] * 5 + [0.5] * 8
        result = detect_exits(
            segments(states),
            frames(states, distances),
            CONFIG,
        )
        self.assertEqual(
            [event["type"] for event in result],
            ["bed_exit", "bed_return"],
        )
        self.assertEqual(result[1]["previous_state"], "OUT")
        self.assertEqual(result[1]["current_state"], "LYING_IN_BED")

    def test_sitting_then_walking_does_not_confirm_return(self):
        states = (
            ["LYING_IN_BED"]
            + ["STANDING"] * 5
            + ["SITTING_ON_BED"] * 3
            + ["WALKING"] * 3
        )
        distances = [0.0] + [1.0] * 5 + [0.5] * 6
        result = detect_exits(
            segments(states),
            frames(states, distances),
            CONFIG,
        )
        self.assertEqual([event["type"] for event in result], ["bed_exit"])


if __name__ == "__main__":
    unittest.main()
