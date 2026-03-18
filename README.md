# Vehicle Transit Tracker

A computer vision pipeline that tracks vehicles across two camera feeds — an entry camera and an exit camera — and measures how long each vehicle takes to transit between them. Built with YOLOv8 for detection/tracking and a ResNet-50 feature extractor for cross-camera re-identification.

---

## Overview

This project solves a non-trivial computer vision problem: **re-identifying the same vehicle across two completely separate camera feeds** with no shared tracking context. Each camera runs an independent YOLO tracker, and vehicles are matched between feeds using visual similarity of their appearance embeddings.

**Key capabilities:**
- Real-time vehicle detection and tracking within configurable regions of interest (ROI)
- Cross-camera vehicle re-identification using deep appearance embeddings
- Transit time calculation from entry to exit
- Annotated sample video output for visual verification of tracking quality
- CSV export of all matched entry/exit pairs with timestamps

---

## How It Works

```
Entry Camera Feed          Exit Camera Feed
       │                          │
  YOLOv8 Tracking            YOLOv8 Tracking
  (within ROI)               (within ROI)
       │                          │
  Crop N frames              Crop N frames
  per vehicle                per vehicle
       │                          │
  ResNet-50                  ResNet-50
  Embedding                  Embedding
  (averaged)                 (averaged)
       │                          │
       └──────────┬───────────────┘
                  │
         Cosine Similarity Match
         (within time window)
                  │
         results.csv
         [entry_id, exit_id, entry_time, exit_time, transit_seconds]
```

### Detection & Tracking
YOLOv8n (`yolov8n.pt`) runs on each frame within the configured ROI, tracking only vehicle classes (car, bus, truck — COCO classes 2, 5, 7). The built-in ByteTrack tracker assigns persistent IDs to each vehicle within each camera.

### Appearance Embedding
Once a vehicle has been visible for a configurable delay period and meets the minimum size threshold, `CROP_NUM` image crops are collected. Each crop is passed through a pretrained ResNet-50 (ImageNet weights, final FC layer replaced with identity) to produce a 2048-dimensional feature vector. The crops are averaged and L2-normalized to produce a single robust embedding per vehicle.

### Cross-Camera Matching
For each exit event, the pipeline performs a nearest-neighbor search over all entry events using **cosine similarity**. A match is only accepted if:
1. The similarity exceeds `SIM_THRESHOLD`
2. The transit time is positive and within `MATCH_TIMEOUT` seconds

Matched entries are removed from the pool to prevent double-matching.

---

## Tech Stack

| Component | Library |
|---|---|
| Object Detection & Tracking | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| Appearance Embedding | PyTorch + torchvision ResNet-50 |
| Video I/O & Annotation | OpenCV |
| Numerical Operations | NumPy |
| Apple Silicon Acceleration | PyTorch MPS backend (auto-detected) |

---

## Project Structure

```
yolov8-learning/
├── yolo.py              # Main pipeline (all logic)
├── requirements.txt     # Python dependencies
├── entry_file.mp4       # Input: entry camera footage (not tracked in git)
├── exit_file.mp4        # Input: exit camera footage (not tracked in git)
├── sample_entry.mp4     # Output: annotated entry camera preview
├── sample_exit.mp4      # Output: annotated exit camera preview
└── results.csv          # Output: matched transit times
```

---

## Setup

**Requirements:** Python 3.9+, `pip`

```bash
# 1. Clone the repo
git clone <repo-url>
cd yolov8-learning

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

On first run, `yolov8n.pt` will be downloaded automatically by the Ultralytics library (~6 MB).

---

## Usage

### 1. Add your video files

Place your footage in the project root:
```
entry_file.mp4   — the camera that vehicles pass first
exit_file.mp4    — the downstream camera
```

Both videos should cover overlapping real-world time (the pipeline matches by timestamp offset within the video).

### 2. Tune configuration (optional)

Open `yolo.py` and adjust the constants at the top of the file:

| Parameter | Default | Description |
|---|---|---|
| `SIM_THRESHOLD` | `0.7` | Cosine similarity cutoff for a valid match. Lower = more matches (less strict). Higher = fewer but more confident matches. |
| `MATCH_TIMEOUT` | `150` | Maximum transit time in seconds. Vehicles that take longer than this won't be matched. |
| `CROP_NUM` | `5` | Number of crops averaged to form each vehicle embedding. More = more robust, slower. |
| `CROP_DELAY_SEC` | `1` | Seconds to wait after a vehicle first appears before collecting crops. Avoids partial/occluded views. |
| `ENTRY_ROI` | `(0.0, 0.167, 1.0, 1.0)` | Region of interest as `(left, top, right, bottom)` in 0–1 fractions of frame size. Limits detection to a specific area. |
| `EXIT_ROI` | `(0.25, 0.0, 1.0, 1.0)` | Same format as `ENTRY_ROI` for the exit camera. |
| `EXIT_MIN_CAR_SIZE` | `3/8` | Minimum fraction of ROI area a vehicle must occupy to be logged on the exit camera. Filters out distant/small detections. |

### 3. Run

```bash
python yolo.py
```

The script will:
1. Generate `sample_entry.mp4` and `sample_exit.mp4` — 90-second annotated previews showing bounding boxes, track IDs, and collection state (red = waiting/too small, yellow = collecting crops, green = logged)
2. Process both full videos and match vehicles
3. Print a summary of matched, unmatched entry, and unmatched exit events
4. Write all matches to `results.csv`

**Expected runtime:** Several minutes for long videos. Processing is CPU-bound unless an Apple Silicon MPS device is detected, in which case embedding inference runs on GPU.

### 4. Inspect outputs

**`results.csv`** — one row per matched vehicle pair:
```
entry_track_id, exit_track_id, entry_time, exit_time, transit_seconds
3, 7, 12.40, 54.80, 42.40
```

**`sample_entry.mp4` / `sample_exit.mp4`** — review these first to verify your ROI is correct and that vehicles are being detected and tracked as expected before running the full pipeline.

---

## Tuning Guide

**Getting too few matches?**
- Lower `SIM_THRESHOLD` (e.g., `0.6`)
- Increase `MATCH_TIMEOUT` if the transit takes longer
- Check sample videos — if vehicles are mostly red/small, adjust the ROI or `EXIT_MIN_CAR_SIZE`

**Getting false matches (wrong vehicles paired)?**
- Raise `SIM_THRESHOLD` (e.g., `0.75–0.85`)
- Increase `CROP_NUM` for more robust embeddings
- Increase `CROP_DELAY_SEC` to avoid capturing vehicles at odd angles

**Vehicles not detected at all?**
- Review your ROI fractions — a common mistake is having the ROI exclude the area where vehicles actually appear
- Open `sample_entry.mp4` and look for the blue ROI boundary rectangle drawn on the frame

---

## Design Decisions

**Why average multiple crops instead of one?** A single frame may catch a vehicle at a bad angle, partially occluded, or in motion blur. Averaging `N` crops produces a more stable embedding that better represents the vehicle's overall appearance.

**Why ResNet-50 instead of a re-ID-specific model?** This project was built as a learning exercise with general-purpose pretrained weights. A purpose-built vehicle re-ID model (e.g., trained on VeRi-776) would likely improve match accuracy, especially for similar-looking vehicles.

**Why YOLOv8n (nano)?** Speed. The nano model is fast enough to process video in reasonable time on CPU/MPS without requiring a discrete GPU, at a modest cost to detection accuracy versus larger variants.