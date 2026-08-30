# Vehicle Counting and Speed Estimation

An automatic traffic survey tool that counts vehicles by class (LV/Light Vehicle, HV/Heavy Vehicle, MC/Motorcycle) and estimates their speed from traffic video, using [YOLO](https://github.com/ultralytics/ultralytics) for detection and tracking. Results are written to an Excel file broken down by configurable time interval (e.g. every 5, 15, or 30 minutes) — matching the format used in traditional manual traffic surveys.

## Motivation

Manual traffic counting (a person with a tally counter, sitting by the road) is slow, error-prone, and doesn't scale well across long survey periods. This project automates that process end-to-end: point it at a video, get back an Excel sheet with vehicle counts and speeds broken down by time interval, ready to drop into a traffic impact analysis report.

## Features

- Vehicle detection and tracking with YOLO (supports any Ultralytics YOLOv8–v10 weights).
- Per-vehicle speed estimation based on frame-to-frame displacement.
- Counts automatically split into configurable time intervals, each written as a new row in Excel as soon as the interval is reached.
- Interactive ROI picker (`tools/pick_roi.py`) — no need to hand-calculate pixel coordinates for a new video/camera angle.
- Automatic inference device selection: **CUDA** (NVIDIA) → **MPS** (Apple Silicon) → **CPU**, or force a specific one manually.
- Two playback modes: an accuracy-first mode (every frame processed, ideal for final data) and a real-time preview mode (synced to the video's original speed, some frames may be skipped).

## Installation

```bash
git clone https://github.com/<username>/vehicle-counting-speed-estimation.git
cd vehicle-counting-speed-estimation
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the YOLO weights you want to use (e.g. `yolov10x.pt`) and place them in the project root, or simply pass the name via `--model` — Ultralytics will download it automatically on first run.

## Usage

```bash
python main.py --video data/sample.mp4 --interval 15
```

More examples:

```bash
# Real-time preview (playback synced to the original video speed, frames may be skipped)
python main.py --video data/sample.mp4 --interval 5 --realtime

# Force a specific device & use a lighter model
python main.py --video data/sample.mp4 --model yolov10s.pt --device mps

# Headless, no preview window (faster, suitable for servers/batch jobs)
python main.py --video data/sample.mp4 --no-display
```

Run `python main.py --help` to see all available options.

### CLI options

| Option           | Default          | Description                                                                 |
|------------------|------------------|-------------------------------------------------------------------------------|
| `--video`        | `data/sample.mp4`| Path to the source video                                                     |
| `--output`       | `output.xlsx`    | Path to the Excel output file                                                |
| `--model`        | `yolov10x.pt`    | Path/name of the YOLO weights to use                                         |
| `--classes`      | `coco.txt`       | Path to the class names file                                                 |
| `--interval`     | `15`             | Excel logging interval, in minutes                                           |
| `--device`       | auto             | `cuda` / `mps` / `cpu`, defaults to auto-detect                              |
| `--realtime`     | off              | Sync playback to the video's real speed (frames may be skipped)              |
| `--no-display`   | off              | Run without a preview window                                                 |
| `--roi-file`     | none             | Path to a JSON file from `tools/pick_roi.py`; falls back to `DEFAULT_ROI` if omitted |
| `--hud-margin-x` | `0.05`           | On-screen counter position, distance from left edge (ratio of frame width)   |
| `--hud-margin-y` | `0.06`           | On-screen counter position, distance from top edge (ratio of frame height)   |
| `--hud-line-gap` | `0.06`           | Vertical spacing between the LV/HV/MC lines (ratio of frame height)          |
| `--hud-scale`    | `2.0`            | Font scale of the on-screen counters                                         |
| `--imgsz`        | `640`            | Input resolution for YOLO inference                                          |

## Calibration

Two things need to be adjusted for a different video/camera:

### 1. ROI (counting area)

Use the interactive `tools/pick_roi.py` tool — click the ROI points directly on the video frame, and the result is automatically saved to a JSON file:

```bash
python tools/pick_roi.py --video data/new_video.mp4 --output roi_new_video.json
```

Controls while the window is open:

| Key/Action | Function                    |
|------------|------------------------------|
| Left click | Add an ROI point             |
| `z`        | Undo the last point          |
| `r`        | Reset all points              |
| `s`        | Save to JSON file and quit    |
| `q`        | Quit without saving           |

Then use that file when running `main.py`:

```bash
python main.py --video data/new_video.mp4 --roi-file roi_new_video.json
```

If `--roi-file` is not provided, `main.py` falls back to the `DEFAULT_ROI` built into the source code — useful if you don't want to generate a JSON file every time and the camera angle hasn't changed.

### 2. Speed conversion

**`PIXEL_TO_METER_SCALE`** — the pixel-distance-to-real-speed conversion factor. This depends on the camera's angle and distance from the road, so it must be recalibrated per video (e.g. by measuring a real-world distance between two points on the road and comparing it to the pixel distance in the frame).

## Accuracy Mode vs. Preview Mode

- **Default (accuracy mode)**: every frame is processed, none are skipped. The preview window may appear slower than the video's real speed (because inference takes time), but that's purely cosmetic — counts and speeds are the most accurate this way. **Use this mode for actual survey data collection.**
- **`--realtime`**: playback is synced to the video's original speed by skipping frames whenever inference falls behind. Good for quick demos/previews, but may reduce counting accuracy since a vehicle could pass through the ROI on a skipped frame.

## Live / CCTV Streams

`main.py` currently targets pre-recorded video files. `cv2.VideoCapture()` itself does support RTSP/live streams (`cv2.VideoCapture("rtsp://...")`), but a few assumptions in the current code are specific to static files and would need to be adapted for continuous live monitoring: unreliable FPS reporting from some streams, interval timing based on wall-clock time instead of video time, threaded frame reading with automatic buffer/latency management, reconnect handling, and a loop that runs indefinitely instead of stopping at end-of-file. This is a natural next step for turning this from a periodic-survey tool into a continuous monitoring system.

## Output

The Excel file contains two sheets:

- **Survey Data** — one row per time interval: `No`, `Time Interval`, `LV`, `HV`, `MC`.
- **Speed** — one row per vehicle with a measured speed: `ID`, `Vehicle Type`, `Speed (km/h)`, `Time Interval`.

## Tech Stack

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — object detection & tracking
- [OpenCV](https://opencv.org/) — video I/O and frame processing
- [PyTorch](https://pytorch.org/) — inference backend (CUDA / Apple MPS / CPU)
- [openpyxl](https://openpyxl.readthedocs.io/) — Excel file generation (no Microsoft Office installation required)

## Possible Extensions

- Threaded live-stream ingestion (RTSP/CCTV) with automatic reconnect
- Web dashboard for uploading videos and viewing results without the CLI
- Multi-camera / multi-ROI support in a single run
- Automatic PDF report generation matching common traffic-survey report formats

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
