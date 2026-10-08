# Ablation results

All measured modes use the same ground truth, one-second grid, and 10-second confirmed-event tolerance. VLM-only and hybrid results are not available: no Gemini inference was run, and the agent remains out of scope.

| Mode | Frame acc. | Macro F1 | Exit precision | Exit recall | Duration MAE (s) | VLM calls | Segments | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| raw_argmax | 72.6% | 67.2% | 0/1 (0.0%) | 0/2 (0.0%) | 11.79 | 0 | 267 | not recorded (source inference predates this evaluation) |
| smoothed | 73.8% | 67.5% | 0/3 (0.0%) | 0/2 (0.0%) | 12.54 | 0 | 123 | not recorded (source inference predates this evaluation) |
| timeline | 78.1% | 72.1% | 0/1 (0.0%) | 0/2 (0.0%) | 8.70 | 0 | 37 | not recorded (source inference predates this evaluation) |
| vlm_only | n/a | n/a | n/a | n/a | n/a | n/a | n/a | not run: Gemini inference requires explicit user authorization |
| hybrid | n/a | n/a | n/a | n/a | n/a | n/a | n/a | not implemented: agent development is on hold |

Segment counts measure consecutive sampled labels before timeline duration filtering; fewer segments indicate less label flicker. The runtime is unavailable because the prediction artifacts were already generated. The evaluator does not present metric-computation time as model runtime.

The Gemini VLM-only row requires a consented, cached run using the constrained prompt. The hybrid row is intentionally not implemented while agent development is on hold.
