# Elder Monitor: Project Journey and Interview Notes

This document explains what was built, why each step was needed, what failed
during development, and how those failures were fixed. It is intended as an
interview preparation guide as well as a project handover document.

## 1. Project goal

The goal is to analyze an elder-care video and identify posture/location
changes such as:

- lying in bed
- sitting or standing near the chair
- walking
- leaving the camera view
- unknown or low-confidence states

The system uses video frames, YOLO pose estimation, ByteTrack, polygon zones,
normalized geometric features, and later rule-based or state-machine
classification.

The current input is [`data/clip.mp4`](data/clip.mp4).

## 2. Final pipeline

The pipeline is intentionally split into stages:

```text
config.yaml
    |
    v
Frame sampling --------------------------+
    |                                     |
    v                                     |
Pose detection and tracking               |
    |                                     |
    v                                     |
data/raw_pose.parquet                     |
    |                                     |
    v                                     |
Feature extraction                        |
    |                                     |
    +--> data/features.parquet
    |
    +--> data/feature_diagnostics.png
```

Important design principle: pose inference is expensive, so raw pose output is
persisted once. Feature calculations and classification can then be changed
without running the neural network again.

## 3. Configuration

All important thresholds and paths are in [`config.yaml`](config.yaml):

```yaml
video: data/clip.mp4
sample_fps: 5
pose_model: yolov8m-pose.pt
bed_polygon: [...]
chair_polygon: [...]
thresholds:
  lying_angle_deg: 60
  standing_hip_ratio: 0.55
  walk_speed: 0.25
  min_kp_conf: 0.3
  unknown_score: 0.45
min_segment_sec: 2.0
```

Keeping thresholds in configuration avoids hidden constants in the code. It
also makes experiments reproducible and allows an operator to tune behavior
without editing the pipeline.

## 4. Step-by-step implementation

### Step 1: Frame sampling

Implemented in [`src/frame_sampler.py`](src/frame_sampler.py).

The source video FPS is read from OpenCV. If the source is 30 FPS and the
configured sample rate is 5 FPS, approximately every sixth source frame is
saved for analysis:

```text
frame 0  -> t = 0.0 seconds
frame 6  -> t = 0.2 seconds
frame 12 -> t = 0.4 seconds
```

The timestamp is calculated from the original source frame index:

```python
t = frame_idx / video_fps
```

This is better than incrementing time by an assumed constant because it keeps
the analysis aligned with the actual video.

### Step 2: Bed and chair polygons

Implemented in [`tools/draw_bed.py`](tools/draw_bed.py).

The tool:

1. Opens a selected video frame.
2. Displays the frame in a fitted window.
3. Converts mouse coordinates back to original video coordinates.
4. Saves the selected polygon to `bed_polygon` or `chair_polygon`.
5. Saves an overlay image for visual verification.

Example commands:

```powershell
.\.venv\Scripts\python.exe tools\draw_bed.py `
  --zone bed `
  --video data\clip.mp4 `
  --frame-time 12

.\.venv\Scripts\python.exe tools\draw_bed.py `
  --zone chair `
  --video data\clip.mp4 `
  --frame-time 12
```

The polygon should cover the mattress surface, not the complete bed frame.
The chair polygon is optional but useful for location classification.

### Step 3: Raw pose extraction

Implemented in [`tools/extract_raw_pose.py`](tools/extract_raw_pose.py).

For each source frame:

1. YOLO pose inference is run with `model.track`.
2. ByteTrack maintains identities.
3. Only frames at the configured sample rate are persisted.
4. Each saved row contains:
   - `frame_idx`
   - `t`
   - `track_id`
   - `present`
   - `bbox`: `[x1, y1, x2, y2]`
   - `keypoints`: 17 `[x, y, confidence]` values

The output is [`data/raw_pose.parquet`](data/raw_pose.parquet).

Parquet was chosen because it stores typed tabular data efficiently and
supports nested list columns such as bounding boxes and keypoints.

Run:

```powershell
.\.venv\Scripts\python.exe tools\extract_raw_pose.py
```

The script refuses to overwrite the output unless this is explicit:

```powershell
.\.venv\Scripts\python.exe tools\extract_raw_pose.py --overwrite
```

### Step 4: Feature extraction

Implemented in [`tools/extract_features.py`](tools/extract_features.py).

The script reads the frozen raw Parquet and does not invoke YOLO again. It
uses the COCO keypoint indices:

| Body part | COCO indices |
|---|---:|
| Shoulders | 5, 6 |
| Hips | 11, 12 |
| Knees | 13, 14 |
| Ankles | 15, 16 |

Features written to [`data/features.parquet`](data/features.parquet):

| Feature | Meaning |
|---|---|
| `torso_angle` | Angle of shoulder midpoint to hip midpoint relative to vertical |
| `bbox_aspect` | Bounding-box width divided by height |
| `hip_in_bed` | Hip midpoint inside the bed polygon |
| `kp_in_bed_frac` | Fraction of visible keypoints inside the bed polygon |
| `hip_height_ratio` | Normalized ankle-to-hip vertical distance |
| `speed` | Hip displacement per second normalized by torso length |
| `vis` | Mean keypoint confidence |
| `present` | Whether a patient detection exists |

Confidence filtering uses `thresholds.min_kp_conf`. Distance-dependent values
are normalized using torso length or bounding-box height so a person farther
from the camera does not automatically appear slower or smaller.

Run:

```powershell
.\.venv\Scripts\python.exe tools\extract_features.py
```

The diagnostic plot is
[`data/feature_diagnostics.png`](data/feature_diagnostics.png).

## 5. Challenges encountered and fixes

### Challenge 1: Configured video did not exist

The initial configuration referred to `data/session1.mp4`, but the repository
contained `data/clip.mp4`.

**Why it happened:** the configuration came from the initial task template,
while the actual local asset had a different name.

**Fix:** changed the canonical input to `data/clip.mp4` and added a `--video`
override to the polygon tool.

**Prevention:**

- Validate that the configured path exists before opening the video.
- List available video files in the error message.
- Keep one canonical path in configuration.
- Avoid hardcoding different video names in separate scripts.

### Challenge 2: The selection window appeared zoomed

The source video was 1920x1080, which was larger than the available display
area.

**Why it happened:** OpenCV initially displayed the frame near its native
resolution.

**Fix:** the selector now fits the display to a maximum size and maps mouse
coordinates back to source-video coordinates.

**Prevention:** always separate display coordinates from source coordinates.
The saved polygon must use original video coordinates, not resized-window
coordinates.

### Challenge 3: The first frame did not show the objects clearly

The first frame was not always a useful frame for selecting the bed or chair.

**Fix:** added `--frame-time`, for example:

```powershell
.\.venv\Scripts\python.exe tools\draw_bed.py `
  --zone bed `
  --video data\clip.mp4 `
  --frame-time 12
```

**Prevention:** make frame selection timestamp-based and save an overlay for
human verification.

### Challenge 4: Parquet support was missing

Pandas was installed, but neither `pyarrow` nor `fastparquet` was available.

**Why it happened:** pandas does not include a Parquet engine by itself.

**Fix:** installed `pyarrow` and added it to [`requirements.txt`](requirements.txt).

**Prevention:** declare all required serialization engines in the dependency
manifest and run a small read/write smoke test after environment setup.

### Challenge 5: Direct script execution could not import `src`

Running `python tools\extract_raw_pose.py` initially raised:

```text
ModuleNotFoundError: No module named 'src'
```

**Why it happened:** when a file inside `tools` is executed directly, Python
places `tools` on `sys.path`, not necessarily the repository root.

**Fix:** the script resolves the project root from `__file__` and adds it to
`sys.path`.

**Prevention:** either package the project properly or consistently run modules
from the repository root. For a small script-based project, the explicit root
bootstrap is practical.

### Challenge 6: The initial Parquet showed only about 20 seconds

The first raw output had 2,007 sampled rows but only 100 `present=True` rows.
The selected track ID ended at approximately 19.8 seconds even though the
patient remained visible.

**Why it happened:** the first implementation ran tracking only on sparse 5
FPS frames and then required one ByteTrack ID to remain valid for the entire
video. ByteTrack can lose or change an ID when frames are skipped or the
person's appearance/pose changes.

**Fixes:**

1. Tracking now runs on every source frame so ByteTrack receives continuous
   temporal context.
2. Only the configured 5 FPS frames are written to Parquet.
3. Detections are retained even when `boxes.id` is unavailable.
4. The patient track selected from the first 30 seconds is preferred.
5. If that ID changes, the largest detected person box is used as the
   single-person fallback for that sampled frame.

**Result:** the final raw output contains 1,876 present rows out of 2,007,
approximately 93.5% coverage across the full 401-second video.

**Prevention:**

- Do not confuse an identity-tracker ID with a permanent human identity.
- Run tracking continuously, even if persistence is sampled.
- Measure coverage, not just file creation success.
- Inspect `present` counts and timestamp ranges after every extraction.
- For multi-person videos, replace the largest-box fallback with a stronger
  identity association method using appearance embeddings, spatial continuity,
  or a manually selected reference track.

### Challenge 7: Nested keypoints were returned as object arrays

Reading Parquet returned the keypoints as a NumPy object array containing
17 separate arrays rather than a direct numeric `(17, 3)` array.

**Fix:** normalized the value with:

```python
keypoints = np.stack(row["keypoints"]).astype(float)
```

**Prevention:** validate nested data shapes at the feature boundary and fail
with a clear error if 17 keypoints are not available.

## 6. Verification checklist

After raw extraction:

```powershell
.\.venv\Scripts\python.exe -c "import pandas as pd; d=pd.read_parquet('data/raw_pose.parquet'); print(len(d)); print(d.present.value_counts().to_dict()); print(d.t.min(), d.t.max())"
```

Expected checks:

- Number of rows equals the number of sampled frames.
- `t` spans the video duration.
- `present=True` coverage is plausible.
- Present rows contain a four-value bounding box.
- Present rows contain exactly 17 keypoints.

After feature extraction:

```powershell
.\.venv\Scripts\python.exe -c "import pandas as pd; d=pd.read_parquet('data/features.parquet'); print(d.columns.tolist()); print(d.t.min(), d.t.max())"
```

Then inspect
[`data/feature_diagnostics.png`](data/feature_diagnostics.png). If the
signals are noisy, do not immediately tune classification thresholds. First
check:

1. patient coverage
2. polygon coordinates
3. keypoint confidence filtering
4. identity association
5. timestamp continuity

## 7. Interview questions and strong answers

### Why did you save raw poses before extracting features?

Pose inference is expensive. Persisting raw detections creates a stable
intermediate dataset, so feature formulas and thresholds can be tuned quickly
without rerunning the neural model.

### Why use timestamps based on source FPS?

The source frame index and source FPS represent the actual video timeline. This
keeps speed calculations and event durations correct even when only every
Nth frame is analyzed.

### Why normalize speed by torso length?

Pixel displacement depends on distance from the camera. Dividing by torso
length converts movement into an approximate body-lengths-per-second measure,
which is more comparable across the video.

### Why use both `hip_in_bed` and `kp_in_bed_frac`?

The hip midpoint is a direct location signal, but one hip can be hidden or
poorly detected. The fraction of visible keypoints inside the polygon is a
more robust secondary signal.

### Why is `present=False` different from `UNKNOWN`?

`present=False` means the person was not detected. `UNKNOWN` should mean the
person was detected but the available features do not confidently distinguish
the posture. Separating these states makes monitoring and alert logic clearer.

### What is the main limitation of the current patient selection?

The current fallback assumes one person and chooses the largest detection when
the preferred tracking ID is unavailable. This is suitable for this video, but
not sufficient for a crowded scene. A production multi-person system would
need robust re-identification and explicit patient association.

### How would you improve production reliability?

- Add automated schema and coverage checks.
- Version the model and configuration with every output.
- Add confidence-aware smoothing and outlier rejection.
- Use a temporal state machine with minimum segment duration.
- Add tests for missing detections, ID switches, and malformed keypoints.
- Use person re-identification for multi-person scenes.
- Store processing metadata such as source hash, model hash, and run time.
- Monitor false positives and false negatives on labeled validation clips.

## 8. Current output summary

The current verified outputs are:

- [`data/raw_pose.parquet`](data/raw_pose.parquet): 2,007 sampled rows,
  1,876 present rows, approximately 93.5% coverage.
- [`data/features.parquet`](data/features.parquet): normalized feature table
  covering the complete 0.0 to 401.2 second timeline.
- [`data/feature_diagnostics.png`](data/feature_diagnostics.png): full-duration
  torso-angle and bed-occupancy diagnostic plot.

## Ground-truth evaluation

Ground-truth intervals are stored in [`data/gt.csv`](data/gt.csv) with
`start,end,state` columns. Re-run state scoring after changing rules:

```powershell
.\.venv\Scripts\python.exe tools\score_states.py --overwrite
```

When `gt.csv` is present, the command prints argmax accuracy and a confusion
matrix over frames whose timestamps fall inside labeled intervals. This is a
rough frame-level check; it does not replace temporal smoothing or segment-level
evaluation. Overlapping intervals should be corrected because the first
matching interval is used.

## Viterbi smoothing

The raw argmax state can flicker because each frame is classified
independently. [`tools/smooth_states.py`](tools/smooth_states.py) treats the
states as a hidden Markov chain:

```text
emission[t, state] = log(frame_score[t, state])
transition[a, b]   = log(P(state_t=b | state_(t-1)=a))
```

The Viterbi dynamic program keeps the best cumulative path and a backpointer
for every state. It then backtracks from the best final state. Self-transition
probability is approximately `0.98`, plausible movements receive small
probability, impossible jumps receive zero probability, and `UNKNOWN` remains
reachable from every state.

Run:

```powershell
.\.venv\Scripts\python.exe tools\smooth_states.py --overwrite
```

Output:
[`data/smoothed_states.parquet`](data/smoothed_states.parquet). The file
contains both the frame-level argmax and the smoothed state so flicker can be
measured directly.

## Timeline segments

[`tools/build_timeline.py`](tools/build_timeline.py) collapses consecutive
smoothed labels into segments:

```text
start, end, state, mean_confidence
```

It merges segments shorter than `min_segment_sec` into the longer neighboring
state, except for a short `STANDING` segment whose neighbors are
`SITTING_ON_BED` and `LYING_IN_BED`; that transition is intentionally retained
because it represents a brief stand. Segment boundaries use the source video
duration, not just the last sampled timestamp, so state totals add up to the
full clip duration.

Run:

```powershell
.\.venv\Scripts\python.exe tools\build_timeline.py --overwrite
```

The human-readable result is
[`data/timeline.txt`](data/timeline.txt), and the structured result is
[`data/timeline.parquet`](data/timeline.parquet).
