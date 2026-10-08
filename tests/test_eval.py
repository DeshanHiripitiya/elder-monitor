import numpy as np
import pytest

from eval.run_all import (
    compare_durations,
    duration_summary,
    evaluate_event_mode,
    evaluate_mode,
)


def test_perfect_predictions_have_perfect_frame_event_and_duration_scores():
    truth = np.array(["LYING_IN_BED"] * 5 + ["WALKING"] * 5)
    prediction = truth.copy()
    segments = [
        {"start": 0, "end": 5, "state": "LYING_IN_BED"},
        {"start": 5, "end": 10, "state": "WALKING"},
    ]
    events = [
        {"type": "bed_exit", "start_time": 5, "confirmed_time": 6},
        {"type": "bed_return", "start_time": 8, "confirmed_time": 9},
    ]

    frame_metrics = evaluate_mode(truth, prediction, step=1)
    truth_durations = duration_summary(
        segments,
        duration=10,
        in_bed_states={"LYING_IN_BED"},
        events=events,
    )
    predicted_durations = duration_summary(
        segments,
        duration=10,
        in_bed_states={"LYING_IN_BED"},
        events=events,
    )
    duration_metrics = compare_durations(truth_durations, predicted_durations)
    event_metrics = evaluate_event_mode(events, events, tolerance=0, traps=[])

    assert frame_metrics["strict_accuracy"]["accuracy"] == 1.0
    assert frame_metrics["macro_f1"] == 1.0
    assert duration_metrics["mean_absolute_error_sec"] == 0
    assert duration_metrics["total_misattributed_time_sec"] == 0
    assert duration_metrics["predicted_duration_sum_sec"] == 10
    for kind in ("bed_exit", "bed_return"):
        score = event_metrics[kind]
        assert score["precision"]["value"] == 1.0
        assert score["recall"]["value"] == 1.0


def test_events_shifted_by_three_seconds_match_only_with_five_second_tolerance():
    truth = [
        {"type": "bed_exit", "start_time": 8, "confirmed_time": 10},
    ]
    prediction = [
        {"type": "bed_exit", "start_time": 11, "confirmed_time": 13},
    ]

    within_five = evaluate_event_mode(prediction, truth, tolerance=5, traps=[])
    within_two = evaluate_event_mode(prediction, truth, tolerance=2, traps=[])

    assert within_five["bed_exit"]["tp"] == 1
    assert within_five["bed_exit"]["fp"] == 0
    assert within_five["bed_exit"]["fn"] == 0
    assert within_two["bed_exit"]["tp"] == 0
    assert within_two["bed_exit"]["fp"] == 1
    assert within_two["bed_exit"]["fn"] == 1


def test_ten_second_example_has_expected_confusion_matrix():
    truth = np.array(["A"] * 5 + ["B"] * 5)
    prediction = np.array(["A", "A", "B", "B", "B", "B", "B", "A", "A", "B"])

    result = evaluate_mode(truth, prediction, step=1)
    matrix = result["confusion_matrix"]["counts"]

    assert matrix == {
        "A": {"A": 2, "B": 3},
        "B": {"A": 2, "B": 3},
    }
    assert result["strict_accuracy"]["accuracy"] == 0.5
    assert result["confusion_matrix"]["row_normalized"] == {
        "A": {"A": 0.4, "B": 0.6},
        "B": {"A": 0.4, "B": 0.6},
    }


@pytest.mark.parametrize(
    "segments",
    [
        [],
        [{"start": 0, "end": 4, "state": "LYING_IN_BED"}],
        [
            {"start": 1, "end": 3, "state": "LYING_IN_BED"},
            {"start": 5, "end": 9, "state": "WALKING"},
        ],
    ],
)
def test_duration_summary_sums_any_segment_list_to_video_length(segments):
    result = duration_summary(
        segments,
        duration=10,
        in_bed_states={"LYING_IN_BED"},
        events=[],
    )

    assert result["duration_sum_sec"] == pytest.approx(10)
    assert result["duration_sum_check_passed"]
