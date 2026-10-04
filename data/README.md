# Local data layout

Video and annotation files are private/local inputs and are intentionally not
committed. Place the configured input video at `data/raw/clip.mp4`. Optional
annotations belong in `data/raw/gt.csv` and `data/raw/gt_events.csv`.

Intermediate tables, rendered videos, plots, event JSON, and other generated
artifacts are written to `data/processed/`. Both data subfolders are ignored
by Git except for their instruction files. Keep original inputs and generated
outputs separate; do not put derived files in `data/raw/`.

After cloning, create/copy the input files and run the pipeline from the
repository root. Set `video` in the relevant config file if your source video
has another path or name.
