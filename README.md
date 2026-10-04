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

## Decision principles

- **NORMAL** means nothing needs a human. **MONITOR** means log it and show it
  on a dashboard, with no notification. **ALERT** means notify a caregiver now.
- Alerts are expensive, because false alarms cause alert fatigue and staff
  start ignoring them. So **ALERT** is reserved for situations that are
  potentially dangerous, and anything merely uncertain goes to **MONITOR**.
- Every decision carries a **reason string** (for example,
  `out_of_bed 18m > 15m night threshold`). A decision without a reason is hard
  to trust and hard to debug.

## Alert thresholds and demo profile

All alert thresholds are configurable under `alerts` in
[`config.yaml`](config.yaml). These values are assumptions for this
demonstration only; they are **not clinically validated values** and must not
be treated as medical guidance.

| Setting | `config.yaml` assumption | Rationale |
|---|---:|---|
| `edge_sit_monitor_sec` | 300 s | A few minutes of edge sitting can be normal, such as putting on slippers or resting. Very long sitting may indicate dizziness, confusion, or difficulty standing, so it is a MONITOR condition. |
| `unknown_monitor_sec` | 30 s | Continuous UNKNOWN is uncertain. Record and show it for review rather than generating an expensive caregiver alert. |
| `exit_low_conf` | 0.6 | An exit below this confidence is uncertain and should be MONITOR, not an immediate ALERT. |
| `out_of_bed_alert_sec.day` | 1,800 s | A toilet visit may take 5–15 minutes; being out much longer can raise fall risk. |
| `out_of_bed_alert_sec.night` | 900 s | The threshold is stricter at night because the person may be unsteady and fewer people may notice a problem. |
| `out_of_view_alert_sec.day` | 900 s | Being out of view is stricter than simply being out of bed because the system cannot observe the person. |
| `out_of_view_alert_sec.night` | 600 s | Night-time absence from view uses a shorter threshold for the same observability and risk reasons. |
| `floor_lying_alert_sec` | 10 s | Floor lying may be dangerous; a short delay filters brief bending or picking something up. |
| `night_window` | 22:00–06:00 | Defines which day/night duration thresholds apply. |
| `video_start_clock` | 02:00 | The video has no clock metadata; this is an assumed local start time used to map its relative timestamps to the night window. |

For a short demonstration, [`config_demo.yaml`](config_demo.yaml) keeps the
same video and posture configuration but scales alert durations down (for
example, 20-second edge sitting and 90-second daytime out-of-bed duration) so
the rules can be exercised within one clip. These accelerated values are
demonstration-only; production values remain in `config.yaml` and differ
substantially.

The source video is short relative to the alert durations. In this repository
its measured duration is about 401 seconds (6 minutes 41 seconds); no 15- or
30-minute threshold can fire during a single playback. Use `config_demo.yaml`
for a live demo, and state clearly that its thresholds are scaled and
unvalidated.

### Decision rules and priorities

The engine evaluates the timeline in time order and returns decisions with
reasons. Threshold values are assumptions, not clinical recommendations:

| Priority | Rule and decision | Threshold | Why this rule/threshold exists |
|---:|---|---|---|
| 1 | Unreturned bed exit → **ALERT** | Day 1,800 s; night 900 s | Toilet visits may take 5–15 minutes; substantially longer may indicate risk. Night uses a shorter interval because the person may be less steady and less observed. |
| 2 | Out of view after exit → **ALERT** | Day 900 s; night 600 s | The system cannot observe the person, so the allowed unseen interval is shorter than the general out-of-bed limit. Night is stricter. |
| 3 | Horizontal `LYING_ON_FLOOR` → **ALERT** | 10 s | Potentially dangerous; the brief delay filters short bending or picking something up. |
| 4 | Confirmed bed exit → **MONITOR** (with `night_exit` note at night) | At exit confirmation | Getting up is often normal but worth logging; it does not page a caregiver by itself. |
| 5 | Low-confidence exit → **MONITOR** | `exit_low_conf: 0.6` | Uncertain evidence should be reviewed, not escalated as an immediate alert. |
| 6 | Prolonged `SITTING_ON_BED` → **MONITOR** | `edge_sit_monitor_sec: 300 s` | A few minutes may be normal (resting or putting on slippers); very long sitting may indicate difficulty standing. |
| 7 | Continuous `UNKNOWN` → **MONITOR** | `unknown_monitor_sec: 30 s` | Persistent uncertainty needs dashboard visibility, but is not by itself proof of danger. |
| 8 | No rule matches → **NORMAL** | None | No configured condition currently warrants human attention. |
| — | Select day/night thresholds from the assumed video clock | `night_window: 22:00–06:00`; `video_start_clock: 02:00` | The clip has no clock metadata; these assumptions determine which time-of-day duration threshold applies. |

thresholds would be tuned per resident (mobility, medication, usual night routine) and per time of day.

Edge sitting is intentionally approximated using all `SITTING_ON_BED`
segments. Sitting at the mattress edge is not reliably distinguished from
sitting up in bed with the current features; this is a known limitation, not
a claim that those postures are equivalent.

`LYING_ON_FLOOR` is an additional state based on a horizontal torso and
`torso_angle >= state_scoring.floor_torso_angle_min_deg` together with
`kp_in_bed_frac` at or below `state_scoring.floor_bed_fraction_max`. It is
added to scoring and smoothing, but this video contains no real fall. The
floor-alert rule is validated with synthetic tests only; no fall was staged.
The generated `summary.json` includes a `final_decision` object with a decision level,
matched rule, and human-readable reason. Each bed-exit entry also records its
own MONITOR decision and note (including `night_exit`), even if a later return
makes the final video state NORMAL.

### Time-based decision evaluation

`decision_timeline` evaluates event and sustained-segment rules in time order,
instead of inferring alerts from end-of-video totals. An out-of-bed or
out-of-view alert is timestamped at `bed_exit.confirmed_time + threshold`,
the time it would have fired during monitoring—not at the later return time.
Threshold selection uses `video_start_clock` plus the current segment/event
timestamp, so an episode spanning the day/night boundary uses the threshold
active when the alert would fire.

The same stateful evaluation can run unchanged on a live stream by supplying
segments/events as they arrive and the current elapsed timestamp. The current
batch pipeline evaluates the completed video timeline, but its alert
timestamps represent the corresponding live decision times.
`summary.json` stores the chronological `decision_timeline` (including
MONITOR decisions) and a separate `alerts` list containing only ALERT
decisions. At the same timestamp, the highest-priority matching rule wins.

Each entry in `events.json` includes `decision` and `reason`. Confirmed bed
exits are MONITOR by default (or MONITOR with a low-confidence reason); a
confirmed bed return is NORMAL because the person is back in bed. Threshold
crossings are written to [`data/alerts.json`](data/alerts.json) with exactly
`time`, `level`, `rule`, and `reason`. `summary.json` also reports
`overall_decision`, the highest severity in the chronological decision
timeline and event decisions.

Synthetic alert-policy cases are in [`tests/test_alerts.py`](tests/test_alerts.py).
They cover return before threshold, day/night duration differences, out of
view, prolonged bed sitting, UNKNOWN duration, low-confidence exit, floor
lying, and horizontal lying on the bed. Run with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_alerts.py -q
```

### Synthetic scenarios: demonstrating all decision levels

The JSON timelines in [`scenarios/`](scenarios/) are explicitly synthetic;
they exercise alert logic and are not real footage, staged falls, or evidence
of clinical safety. Run them with the accelerated demo thresholds:

```powershell
.\.venv\Scripts\python.exe tools\run_scenarios.py
```

Expected demonstration results:

| Synthetic timeline | Decisions | Overall |
|---|---|---|
| Floor lying | ALERT at 3.0 s (`floor_lying`) | **ALERT** |
| Night out of bed | MONITOR at 0.0 s (`confirmed_bed_exit`); ALERT at 45.0 s (`out_of_bed_duration`) | **ALERT** |
| Night out of view | MONITOR at 0.0 s (`confirmed_bed_exit`); ALERT at 30.0 s (`out_of_view_duration`) | **ALERT** |

The demo uses a 3-second floor threshold, 45-second night out-of-bed
threshold, and 30-second night out-of-view threshold so these cases complete
quickly. Production-assumption settings are longer; neither set has been
clinically validated. Floor lying is untested on real footage, and edge
sitting remains an approximation of all `SITTING_ON_BED` segments.

## Bounded agent triage

The agent begins with a deterministic trigger inventory; this step does not
call a VLM. It creates cases for low-confidence or short segments, UNKNOWN
segments, lying segments with a mixed bed/floor keypoint signal, exit
candidates, and bed exits below the configured alert confidence. Overlapping
triggers are grouped into one time window so the same uncertainty does not
create duplicate cases.

Run the inventory against the existing timeline and pose features:

```powershell
.\.venv\Scripts\python.exe tools\triage_cases.py
```

The current video yields **8 cases**, within the requested 5–15 range. They
cover 0.0–11.8 s, 18.2–88.2 s, 150.6–156.4 s, 174.6–180.8 s,
190.8–199.8 s, 228.6–231.0 s, 314.8–347.2 s, and 359.6–378.2 s.
The command prints trigger reasons and configured ceilings: 6 steps per case,
2 VLM calls per case, 40 total VLM calls, and 6 frames per VLM call.
Candidate-exit triggers require an earlier in-bed segment, so ordinary
walking at the start of a clip does not become a false bed-exit case.

All thresholds and call budgets live under `agent` in
[`config.yaml`](config.yaml); the same agent settings are available in
[`config_demo.yaml`](config_demo.yaml). The target execution order is
state-history lookup, pose-feature lookup, wider temporal context, then a
budgeted VLM call only when cheaper evidence has not resolved the case. The
VLM executor and its call-rate measurement are not implemented in this
trigger-inventory step, so no “8% of segments” call rate is claimed yet.
Future evaluation should compute that percentage from actual call records.
