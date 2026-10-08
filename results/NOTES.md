# Evaluation notes

The highest state accuracy occurred at a non-zero prediction offset:

- `raw_argmax`: -1 s
- `smoothed`: -1 s
- `timeline`: -2 s

Do not shift prediction outputs to improve this score. Review the ground-truth timestamps and clap-sync reference, correct the ground truth if it is misaligned, then rerun the evaluation.

## Threshold tuning history

- 2026-10-08: No state or event threshold values were changed during evaluation/reporting. Thresholds were developed using this same video; the repository does not contain a complete earlier change-by-change tuning log.
