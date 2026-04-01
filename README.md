# Dodo CV Table Detector

<p align="center">
  <img src="https://media1.tenor.com/m/7mPAuCDLAc0AAAAC/kendallwantstouse-headphone-cat.gif" alt="headphone cat" width="220">
</p>

Lightweight Python tool for detecting the state of a single table in video using OpenCV.

The script analyzes a selected table region (`ROI`), builds a background model, tracks activity, and exports both annotated video and structured reports.

## Features

- Detects basic table states:
  - `empty`
  - `occupied`
  - `approach`
- Supports manual ROI input or interactive ROI selection
- Builds a static background from the first frames of the video
- Adapts the background gradually during `empty` periods
- Exports:
  - annotated output video
  - event timeline CSV
  - state segment CSV
  - JSON summary
  - text report

## Use Cases

- quick occupancy analysis for a single table
- estimating idle time between visitors
- CV prototyping without training a model
- baseline analytics for restaurant or cafe video footage

## Tech Stack

- Python 3.10+
- OpenCV
- NumPy
- Pandas

## Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

Interactive ROI selection:

```bash
python main.py --video path/to/video.mp4 --select-roi
```

Manual ROI:

```bash
python main.py --video path/to/video.mp4 --roi 200 150 240 180
```

Extended example:

```bash
python main.py ^
  --video path/to/video.mp4 ^
  --output result.mp4 ^
  --roi 200 150 240 180 ^
  --bg-frames 30 ^
  --threshold 0.04 ^
  --min-frames-for-change 5 ^
  --binary-threshold 25 ^
  --blur-size 5 ^
  --background-update-alpha 0.02
```

For macOS / Linux shells:

```bash
python main.py \
  --video path/to/video.mp4 \
  --output result.mp4 \
  --roi 200 150 240 180 \
  --bg-frames 30 \
  --threshold 0.04 \
  --min-frames-for-change 5 \
  --binary-threshold 25 \
  --blur-size 5 \
  --background-update-alpha 0.02
```

## CLI Arguments

- `--video` - input video path
- `--roi X Y W H` - ROI provided manually
- `--select-roi` - interactive ROI selection with mouse
- `--output` - output video name
- `--bg-frames` - number of initial frames used to build the background
- `--threshold` - motion ratio threshold for `occupied`
- `--min-frames-for-change` - number of consecutive frames required to confirm a state change
- `--binary-threshold` - threshold used for frame differencing binarization
- `--blur-size` - Gaussian blur kernel size for noise reduction
- `--background-update-alpha` - background adaptation rate during `empty`
- `--progress-every` - print progress every `N` frames

## Output Files

If `--output result.mp4` is provided, the script creates:

- `result.mp4`
- `result_events.csv`
- `result_segments.csv`
- `result_summary.json`
- `result_report.txt`

## Repository Layout

```text
.
├── main.py
├── requirements.txt
├── README.md
├── CHANGELOG.md
└── .gitignore
```

## Limitations

- This is a heuristic baseline, not a trained human detection model.
- Accuracy depends strongly on ROI quality.
- Strong lighting changes, reflections, shadows, or partial occlusions can reduce quality.
- Very large videos may take a while to process.
- The current implementation is focused on one table per run.

## Demo Media

Large source videos and generated outputs are intentionally not included in the repository.

Recommended options for showcasing results:

- add a short demo GIF to the README
- upload a short sample clip to the repository releases
- link full-resolution source videos from cloud storage

## Roadmap

- batch processing for multiple videos
- preview frame export with ROI overlay
- automatic ROI suggestions
- cleaner package/module structure
- lightweight tests for smoke validation

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
