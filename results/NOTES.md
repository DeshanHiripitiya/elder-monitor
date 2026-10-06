# Evaluation notes

The highest state accuracy occurred at a non-zero prediction offset:

- `raw_argmax`: -1 s
- `smoothed`: -1 s
- `timeline`: -2 s

Do not shift prediction outputs to improve this score. Review the ground-truth timestamps and clap-sync reference, correct the ground truth if it is misaligned, then rerun the evaluation.
