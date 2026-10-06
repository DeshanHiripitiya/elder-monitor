import numpy as np
import pandas as pd
import pytest

from eval.run_all import (
    compare_durations,
    duration_summary,
    evaluate_mode,
    evaluate_event_mode,
    ground_truth_detector_segments,
    load_ground_truth,
    match_events,
    save_confusion_heatmap,
    to_grid,
)


def test_to_grid_uses_half_open_intervals_and_unknown_for_gaps():
    _, labels = to_grid(
        [
            {"start": 0, "end": 2, "state": "LYING_IN_BED"},
            {"start": 3, "end": 4, "state": "WALKING"},
        ],
        duration=5,
    )

    assert labels.tolist() == [
        "LYING_IN_BED",
        "LYING_IN_BED",
        "UNKNOWN",
        "WALKING",
        "UNKNOWN",
    ]


def test_to_grid_rejects_overlapping_segments():
    with pytest.raises(ValueError, match="overlap"):
        to_grid(
            [
                {"start": 0, "end": 2, "state": "LYING_IN_BED"},
                {"start": 1, "end": 3, "state": "WALKING"},
            ],
            duration=4,
        )


def test_evaluate_mode_finds_known_positive_offset():
    truth = np.array(["LYING", "LYING", "SITTING", "WALKING", "WALKING"])
    predictions = np.array(["UNKNOWN", "LYING", "LYING", "SITTING", "WALKING"])

    result = evaluate_mode(truth, predictions, step=1)

    assert result["zero_shift"]["accuracy"] == 0.4
    assert result["best_shift"]["shift_sec"] == -1
    assert result["best_shift"]["accuracy"] == 0.8


def test_load_ground_truth_accepts_corrected_minute_second_labels(tmp_path):
    path = tmp_path / "gt.csv"
    path.write_text(
        "start,end,state\n0:00,2:53,LYING_IN_BED\n"
        "2:53,2:54,SITTING_ON_BED\n"
        "2:54,4:48,STANDING\n"
        "4:48,5:10,LYING_IN_BED\n",
        encoding="utf-8",
    )

    result = load_ground_truth(path)

    assert result[1] == {
        "start": 173.0,
        "end": 174.0,
        "state": "SITTING_ON_BED",
    }


def test_load_ground_truth_rejects_overlapping_intervals(tmp_path):
    path = tmp_path / "gt.csv"
    path.write_text(
        "start,end,state\n0:00,1:00,LYING_IN_BED\n"
        "0:30,1:30,SITTING_ON_BED\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="overlapping"):
        load_ground_truth(path)


def test_metrics_report_strict_unknown_coverage_f1_and_boundary_tolerance():
    truth = np.array(["A"] * 5 + ["B"] * 5)
    predictions = np.array(
        ["A", "A", "A", "A", "UNKNOWN", "B", "B", "B", "B", "B"]
    )

    result = evaluate_mode(
        truth,
        predictions,
        step=1,
        boundary_times=[5],
    )

    assert result["strict_accuracy"] == {
        "correct": 9,
        "total": 10,
        "accuracy": 0.9,
    }
    assert result["coverage"]["coverage"] == 0.9
    assert result["coverage"]["selective_accuracy"] == 1.0
    assert result["macro_f1"] == pytest.approx(17 / 18)
    assert result["boundary_tolerant_accuracy"]["1_sec"] == {
        "correct": 7,
        "total": 7,
        "ignored_samples": 3,
        "accuracy": 1.0,
    }
    assert result["boundary_tolerant_accuracy"]["2_sec"]["total"] == 5


def test_heatmap_saves_counts_and_row_normalized_confusion(tmp_path):
    truth = np.array(["A", "A", "B", "B"])
    predictions = np.array(["A", "UNKNOWN", "B", "A"])
    result = evaluate_mode(truth, predictions, step=1)
    output_path = tmp_path / "confusion.png"

    save_confusion_heatmap(result, "timeline", output_path)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert result["confusion_matrix"]["counts"]["A"]["UNKNOWN"] == 1
    assert result["confusion_matrix"]["row_normalized"]["B"]["A"] == 0.5


def test_event_matching_is_one_to_one_by_nearest_confirmed_time():
    ground_truth = [
        {"type": "bed_exit", "start_time": 8, "confirmed_time": 10},
        {"type": "bed_exit", "start_time": 18, "confirmed_time": 20},
    ]
    predictions = [
        {"type": "bed_exit", "start_time": 7, "confirmed_time": 9},
        {"type": "bed_exit", "start_time": 9, "confirmed_time": 10.5},
        {"type": "bed_exit", "start_time": 19, "confirmed_time": 20.5},
    ]

    result = match_events(predictions, ground_truth, tolerance=1)

    assert len(result["matches"]) == 2
    assert result["unmatched_prediction_indices"] == [0]
    assert result["unmatched_ground_truth_indices"] == []


def test_missed_gt_timing_diagnostics_do_not_reuse_one_prediction():
    ground_truth = [
        {"type": "bed_exit", "start_time": 9, "confirmed_time": 10},
        {"type": "bed_exit", "start_time": 19, "confirmed_time": 20},
    ]
    predictions = [
        {"type": "bed_exit", "start_time": 9, "confirmed_time": 10.5},
    ]

    result = evaluate_event_mode(predictions, ground_truth, 0, traps=[])

    assert result["bed_exit"]["tp"] == 0
    candidates = result["bed_exit"]["nearest_candidates_for_missed_ground_truth"]
    assert len(candidates) == 2
    assert sum(candidate["nearest_prediction_available"] for candidate in candidates) == 1


def test_event_metrics_report_false_exit_trap_and_one_to_one_counts():
    ground_truth = [
        {"type": "bed_exit", "start_time": 100, "confirmed_time": 110},
        {"type": "bed_return", "start_time": 200, "confirmed_time": 210},
    ]
    predictions = [
        {"type": "bed_exit", "start_time": 101, "confirmed_time": 109},
        {"type": "bed_exit", "start_time": 150, "confirmed_time": 155},
        {"type": "bed_return", "start_time": 201, "confirmed_time": 211},
    ]
    traps = [{"name": "brief stand", "start": 150, "end": 156}]

    result = evaluate_event_mode(predictions, ground_truth, 5, traps)

    assert result["bed_exit"]["tp"] == 1
    assert result["bed_exit"]["fp"] == 1
    assert result["bed_exit"]["fn"] == 0
    assert result["bed_exit"]["false_predictions"][0]["classification"] == (
        "annotated_trap_false_positive"
    )
    assert "brief stand" in result["bed_exit"]["false_predictions"][0]["explanation"]
    assert result["bed_return"]["tp"] == 1
    assert result["trap_results"]["correctly_ignored"] == 0


def test_ground_truth_detector_trace_encodes_non_exit_traps_and_out_of_view():
    annotations = pd.DataFrame(
        [
            {
                "event": "not_an_exit",
                "start_time": "00:10",
                "confirmed_time": "00:20",
                "note": "stood 5s then sat back down",
            },
            {
                "event": "bed_exit",
                "start_time": "00:30",
                "confirmed_time": "00:45",
                "note": "leaves camera view",
            },
        ]
    )

    segments = ground_truth_detector_segments(annotations, duration=50)

    assert segments[1]["state"] == "STANDING"
    assert segments[1]["end"] - segments[1]["start"] == 5
    assert any(
        segment["state"] == "OUT_OF_BED"
        and segment.get("_present") is False
        and segment["start"] == 30
        for segment in segments
    )


def test_duration_summary_fills_gaps_and_checks_total_video_duration():
    segments = [
        {"start": 0, "end": 4, "state": "LYING_IN_BED"},
        {"start": 5, "end": 8, "state": "WALKING"},
    ]
    events = [
        {"type": "bed_exit", "start_time": 4, "confirmed_time": 5},
        {"type": "bed_return", "start_time": 8, "confirmed_time": 9},
    ]

    result = duration_summary(
        segments,
        duration=10,
        in_bed_states={"LYING_IN_BED", "SITTING_ON_BED"},
        events=events,
    )

    assert result["duration_by_state_sec"]["LYING_IN_BED"] == 4
    assert result["duration_by_state_sec"]["WALKING"] == 3
    assert result["duration_by_state_sec"]["UNKNOWN"] == 3
    assert result["time_in_bed_sec"] == 4
    assert result["time_out_of_bed_sec"] == 6
    assert result["exit_count"] == 1
    assert result["longest_out_of_bed_period_sec"] == 4
    assert result["duration_sum_check_passed"]


def test_duration_comparison_reports_mae_and_total_misattributed_share():
    gt = duration_summary(
        [
            {"start": 0, "end": 5, "state": "LYING_IN_BED"},
            {"start": 5, "end": 10, "state": "WALKING"},
        ],
        duration=10,
        in_bed_states={"LYING_IN_BED"},
        events=[],
    )
    predicted = duration_summary(
        [
            {"start": 0, "end": 4, "state": "LYING_IN_BED"},
            {"start": 4, "end": 10, "state": "WALKING"},
        ],
        duration=10,
        in_bed_states={"LYING_IN_BED"},
        events=[],
    )

    result = compare_durations(gt, predicted)

    assert result["mean_absolute_error_sec"] == pytest.approx(2 / 8)
    assert result["total_misattributed_time_sec"] == 1
    assert result["total_misattributed_time_percent_of_video"] == 10
