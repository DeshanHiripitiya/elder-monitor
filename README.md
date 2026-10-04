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

The normalized `hip_knee_ratio` feature is computed from each visible
hip-knee pair as `abs(knee_y - hip_y) / bbox_height`. Both visible sides are
averaged; one side is sufficient. Ankle keypoints remain available for the
legacy `hip_height_ratio` feature but are not required for posture scoring.
The `dist_to_bed` feature is zero when the hip is inside the bed polygon and
otherwise stores the signed polygon-edge distance divided by torso length.
It distinguishes standing beside the bed from standing farther away.

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
state confidence, normalized `dist_to_bed`, bed-exit event details, and the
17 frozen COCO pose keypoints:

```powershell
.\.venv\Scripts\python.exe tools\render_debug_video.py --overwrite
```

Each point is labeled as `keypoint_index:confidence`. Green points have high
confidence, yellow points are moderate, and gray points are below
`thresholds.min_kp_conf`. The skeleton is drawn only when both endpoints meet
that confidence threshold. Because raw pose data is stored at 5 FPS while the
video is 30 FPS, the most recent sampled pose is held until the next sample;
the overlay marks its source timestamp as `pose sample t=...`.

The header also shows `dist_to_bed` in torso lengths. It is `0.00` while the
hip is inside the bed polygon and increases as the hip moves away from the
bed. The value is read from [`data/features.parquet`](data/features.parquet),
so rendering does not rerun pose inference.

When [`data/events.json`](data/events.json) exists, the renderer adds an event
panel around each event. The panel shows the event type, `CANDIDATE` or
`CONFIRMED` status, start and confirmation times, previous/current states,
event confidence, and optional notes. It remains visible for five seconds
after confirmation. Use `--events` to provide a different event JSON file.

An already-rendered MP4 cannot receive new burned-in labels without
re-encoding. If you do not want to render the video again, export a WebVTT
sidecar instead:

```powershell
.\.venv\Scripts\python.exe tools\export_event_captions.py
```

This creates [`data/debug_overlay_events.vtt`](data/debug_overlay_events.vtt).
Load it as a subtitle/caption track alongside
[`data/debug_overlay.mp4`](data/debug_overlay.mp4) in a player that supports
WebVTT. It includes the duration summary for the first ten seconds and both
`bed_exit` and `bed_return` labels at their event times. Regenerate only this
small sidecar after changing event or duration outputs.

The rendering stage reuses the saved Parquet outputs and does not rerun pose
inference. The result is [`data/debug_overlay.mp4`](data/debug_overlay.mp4).

## Detect bed-exit events

Bed exits are detected from the smoothed timeline and normalized
`dist_to_bed` feature. A `STANDING` or `WALKING` segment creates a candidate.
The candidate becomes a confirmed `bed_exit` only after the person remains at
least `events.exit_away_sec` seconds beyond `events.exit_away_dist`
torso-lengths. Returning to an in-bed state cancels the candidate. Short
`UNKNOWN` gaps preserve the current phase according to
`events.unknown_hold_sec`.

If the person disappears directly after an in-bed segment, an
`out_of_view_exit_sec` duration produces a lower-confidence event with the
note `via_out_of_view`.

Event confidence is an explainable heuristic, not a calibrated probability:
it is the minimum mean confidence of the segments involved in the event,
multiplied by `0.8` when an `UNKNOWN` segment occurs inside the event
interval. Out-of-view exits receive an additional `0.8` penalty because the
away period is not visually observed.

The event detector also writes a boundary-based duration summary. It sums
`segment.end - segment.start` for every timeline segment, reports time in and
out of bed, the longest exit-to-return interval, the final state, and event
counts. It verifies that the segment-duration sum matches the source video
duration within one second.

After a confirmed exit, the detector looks for a return only while in `OUT`.
When the person comes within `events.return_approach_dist` torso lengths of
the bed, it enters `APPROACH`. `SITTING_ON_BED` must last at least
`events.return_sit_sec`, followed by `LYING_IN_BED` for at least
`events.return_lie_sec`, to emit a `bed_return`. If the person stands or walks
away again, the approach is cancelled and the phase returns to `OUT`.

Run this lightweight event stage without rerunning pose inference or creating
a debug video:

```powershell
.\.venv\Scripts\python.exe tools\detect_events.py --overwrite
```

Outputs:

- [`data/events.json`](data/events.json): detected event list
- [`data/summary.json`](data/summary.json): `bed_exit`/`bed_return` counts, events, and event configuration

## Compare event logic with ground truth

Before evaluating predicted states, run the detector on hand-annotated state
segments. The current annotation file can use event annotations with:

```csv
event,start_time,confirmed_time,note
bed_exit,03:00,03:15,"sit up, stand, walk away"
bed_return,05:00,05:08,"approach bed, sit, lie down"
```

Then run:

```powershell
.\.venv\Scripts\python.exe tools\compare_ground_truth_events.py
```

The tool also accepts a `start,end,state` state-segment CSV. For event
annotations, it converts each annotated interval into deterministic
ground-truth state segments and derives only `dist_to_bed` and `present` for
the logic test; pose features are not involved. It checks for the expected two
`bed_exit` and two `bed_return` events. If this comparison fails, fix event
logic before investigating perception.
If this comparison fails, fix event logic before investigating perception.
