# Bed-event evaluation

There are only 2 annotated bed exits and 2 annotated bed returns (4 positive events total). One miss changes per-kind recall by 50 percentage points; four events cannot support strong claims.

Matching uses one-to-one nearest same-kind events by confirmed time. Timing error is predicted minus ground truth. Detection delay is predicted confirmation minus the GT event start; detector confirmation latency (confirmation minus predicted start) is shown separately.

| Detector input | Kind | Tolerance | TP | FP | FN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ground_truth_segments | bed_exit | +/-5s | 0 | 3 | 2 | 0/3 (0.0%) | 0/2 (0.0%) | 0.0% |
| ground_truth_segments | bed_return | +/-5s | 2 | 1 | 0 | 2/3 (66.7%) | 2/2 (100.0%) | 80.0% |
| ground_truth_segments | bed_exit | +/-10s | 2 | 1 | 0 | 2/3 (66.7%) | 2/2 (100.0%) | 80.0% |
| ground_truth_segments | bed_return | +/-10s | 2 | 1 | 0 | 2/3 (66.7%) | 2/2 (100.0%) | 80.0% |
| predicted_segments | bed_exit | +/-5s | 0 | 1 | 2 | 0/1 (0.0%) | 0/2 (0.0%) | 0.0% |
| predicted_segments | bed_return | +/-5s | 0 | 1 | 2 | 0/1 (0.0%) | 0/2 (0.0%) | 0.0% |
| predicted_segments | bed_exit | +/-10s | 0 | 1 | 2 | 0/1 (0.0%) | 0/2 (0.0%) | 0.0% |
| predicted_segments | bed_return | +/-10s | 0 | 1 | 2 | 0/1 (0.0%) | 0/2 (0.0%) | 0.0% |

## Negative non-exit traps

- **ground_truth_segments: 2 of 3 traps correctly ignored.**
  - sit-up: correctly ignored.
  - edge sitting: correctly ignored.
  - brief stand: triggered 1 false exit(s).
- **predicted_segments: 3 of 3 traps correctly ignored.**
  - sit-up: correctly ignored.
  - edge sitting: correctly ignored.
  - brief stand: correctly ignored.
- UNKNOWN blip is not one of the 3 annotated traps; the existing `test_unknown_two_second_blip_does_not_create_exit` unit test covers it.

## False exits at +/-10s matching tolerance

- ground_truth_segments: confirmed at 163.8s; overlaps annotated non-exit trap(s): brief stand.
- predicted_segments: confirmed at 180.8s; not linked to an annotated trap; nearest GT bed_exit is 14.2s away, outside the +/-10s matching tolerance.

## Timing

- ground_truth_segments bed_exit, GT 180.0s/195.0s: matched; start error +2.0s, confirmed error -9.2s, detection delay from GT start +5.8s, detector latency 3.8s.
- ground_truth_segments bed_exit, GT 340.0s/355.0s: matched; start error +0.0s, confirmed error -5.2s, detection delay from GT start +9.8s, detector latency 9.8s.
- ground_truth_segments bed_return, GT 300.0s/308.0s: matched; start error +0.0s, confirmed error -1.2s, detection delay from GT start +6.8s, detector latency 6.8s.
- ground_truth_segments bed_return, GT 385.0s/395.0s: matched; start error +0.0s, confirmed error -3.2s, detection delay from GT start +6.8s, detector latency 6.8s.
- predicted_segments bed_exit, GT 180.0s/195.0s: nearest but outside tolerance; start error -5.4s, confirmed error -14.2s, detection delay from GT start +0.8s, detector latency 6.2s.
- predicted_segments bed_exit, GT 1: no same-kind prediction was available for a timing comparison.
- predicted_segments bed_return, GT 300.0s/308.0s: nearest but outside tolerance; start error -30.2s, confirmed error -13.8s, detection delay from GT start -5.8s, detector latency 24.4s.
- predicted_segments bed_return, GT 1: no same-kind prediction was available for a timing comparison.
