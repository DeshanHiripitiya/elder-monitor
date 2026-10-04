# Elder monitor

## Select the mattress polygon

Run the interactive selector from the repository root:

```powershell
.\.venv\Scripts\python.exe tools\draw_bed.py
```

Click the four corners of the **mattress surface**, rather than the whole bed
frame. Press `s` to save the points to [`config.yaml`](config.yaml), or `q` to
cancel. The first-frame overlay is written to
[`data/bed_polygon_overlay.png`](data/bed_polygon_overlay.png).

Review the saved overlay here after selecting the polygon:

![Mattress polygon overlay](data/bed_polygon_overlay.png)

The video path and output location can be overridden:

```powershell
.\.venv\Scripts\python.exe tools\draw_bed.py `
  --config config.yaml `
  --output-overlay data\bed_polygon_overlay.png
```

If the first frame does not show the objects clearly, select a frame at a
specific timestamp. For example, to use the 12-second frame:

```powershell
.\.venv\Scripts\python.exe tools\draw_bed.py `
  --zone bed `
  --video data\clip.mp4 `
  --frame-time 12
```

The chair selector supports the same timestamp option:

```powershell
.\.venv\Scripts\python.exe tools\draw_bed.py `
  --zone chair `
  --video data\clip.mp4 `
  --frame-time 12
```

## Extract raw tracked poses

Install the project dependencies and run the pose pipeline once:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe tools\extract_raw_pose.py
```

The pipeline samples at `sample_fps`, tracks with ByteTrack, selects the track
with the most detections in the first 30 seconds, and writes one patient row
per sampled frame to [`data/raw_pose.parquet`](data/raw_pose.parquet). Rows
without the selected patient have `present=False`. The output is protected
against accidental reruns; use `--overwrite` only when intentionally
regenerating the raw pose data.

## Extract normalized features

Build features from the frozen raw pose output without rerunning pose inference:

```powershell
.\.venv\Scripts\python.exe tools\extract_features.py
```

This writes [`data/features.parquet`](data/features.parquet) and the diagnostic
plot [`data/feature_diagnostics.png`](data/feature_diagnostics.png). The plot
shows torso angle and the fraction of visible keypoints inside the bed polygon
over time. Feature output is protected against accidental overwrites.

![Feature diagnostics](data/feature_diagnostics.png)

## Score per-frame states

Generate normalized soft scores instead of hard labels:

```powershell
.\.venv\Scripts\python.exe tools\score_states.py
```

Outputs:

- [`data/state_scores.parquet`](data/state_scores.parquet): scores and argmax state
- [`data/frame_scores.npy`](data/frame_scores.npy): matrix with shape
  `(T, 7)` in the order configured in `tools/score_states.py`

The states are `LYING_IN_BED`, `SITTING_ON_BED`,
`SITTING_OUTSIDE_BED`, `STANDING`, `WALKING`, `OUT_OF_BED`, and `UNKNOWN`.
Scores sum to 1 for each frame. Low visibility and low maximum confidence
route frames toward `UNKNOWN`. If [`data/gt.csv`](data/gt.csv) is present, the
command also compares argmax states against its labeled time intervals and
prints rough accuracy and a confusion matrix.

## Smooth states with Viterbi

Frame-by-frame argmax labels can flicker. Viterbi smoothing finds the single
most likely whole sequence given both the evidence and which transitions are
physically possible:

```powershell
.\.venv\Scripts\python.exe tools\smooth_states.py --overwrite
```

The transition model keeps states sticky at the configured 5 FPS, allows
realistic movements such as `LYING_IN_BED` to `SITTING_ON_BED` to `STANDING`
to `WALKING`, forbids direct lying-to-walking transitions, and gives
`UNKNOWN` a small escape probability from every state. The output is written
to [`data/smoothed_states.parquet`](data/smoothed_states.parquet).

## Build the timeline

Collapse consecutive smoothed states into duration-filtered segments:

```powershell
.\.venv\Scripts\python.exe tools\build_timeline.py --overwrite
```

Outputs:

- [`data/timeline.parquet`](data/timeline.parquet): start, end, state, and mean confidence
- [`data/timeline.txt`](data/timeline.txt): human-readable timeline

Segments shorter than `min_segment_sec` are merged into a neighboring segment.
Short `STANDING` segments between bed sitting/lying segments are preserved for
the brief-stand case. State totals are printed and checked against the source
video duration.

## Render a visual debug video

Render the original clip with the bed polygon, timestamp, smoothed state,
state confidence, and the 17 frozen COCO pose keypoints:

```powershell
.\.venv\Scripts\python.exe tools\render_debug_video.py --overwrite
```

Each point is labeled as `keypoint_index:confidence`. Green points have high
confidence, yellow points are moderate, and gray points are below
`thresholds.min_kp_conf`. The skeleton is drawn only when both endpoints meet
that confidence threshold. Because raw pose data is stored at 5 FPS while the
video is 30 FPS, the most recent sampled pose is held until the next sample;
the overlay marks its source timestamp as `pose sample t=...`.

The rendering stage reuses the saved Parquet outputs and does not rerun pose
inference. The result is [`data/debug_overlay.mp4`](data/debug_overlay.mp4).

## Select the chair polygon

Use the same first-frame selector for the chair zone:

```powershell
.\.venv\Scripts\python.exe tools\draw_bed.py `
  --zone chair `
  --video data\clip.mp4
```

Click the corners of the chair seat/area and press `s`. The points are saved
to `chair_polygon` in [`config.yaml`](config.yaml), with the overlay written
to [`data/chair_polygon_overlay.png`](data/chair_polygon_overlay.png).
