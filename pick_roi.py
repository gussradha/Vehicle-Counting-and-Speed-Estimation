"""
Pick ROI
========

Interactive helper to define ROI (Region of Interest) points by clicking on a
video frame, then saving them to a JSON file that can be passed straight to
main.py via --roi-file.

Controls:
    Left click  - add a point
    Press 'z'   - undo the last point
    Press 's'   - save the ROI to file and quit
    Press 'r'   - reset all points
    Press 'q'   - quit without saving

Usage example:
    python tools/pick_roi.py --video data/sample.mp4 --output roi.json
    python main.py --video data/sample.mp4 --roi-file roi.json
"""

from __future__ import annotations

import argparse
import json
import sys

import cv2
import numpy as np

FRAME_SIZE = (1020, 500)  # must match FRAME_SIZE in main.py
WINDOW_NAME = "Pick ROI - click points, 's' save, 'z' undo, 'r' reset, 'q' quit"

points: list[tuple[int, int]] = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point added: ({x}, {y})  -> {len(points)} point(s) total")


def draw_overlay(frame: np.ndarray) -> np.ndarray:
    overlay = frame.copy()
    for i, p in enumerate(points):
        cv2.circle(overlay, p, 4, (0, 255, 0), -1)
        cv2.putText(overlay, str(i), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if len(points) >= 2:
        cv2.polylines(overlay, [np.array(points, np.int32)], len(points) >= 3, (0, 0, 255), 2)
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactively pick ROI points from a video frame.")
    parser.add_argument("--video", required=True, help="Path to the source video")
    parser.add_argument("--output", default="roi.json", help="Path to the output JSON file")
    parser.add_argument(
        "--frame-index", type=int, default=0,
        help="Which frame to use as reference (default: the first frame)",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Could not open video: {args.video}")

    if args.frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        sys.exit("Failed to read a frame from the video.")

    frame = cv2.resize(frame, FRAME_SIZE)

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    print("Click the ROI points in order (clockwise or counter-clockwise, doesn't matter).")
    print("Press 's' to save, 'z' to undo, 'r' to reset, 'q' to quit without saving.\n")

    while True:
        cv2.imshow(WINDOW_NAME, draw_overlay(frame))
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            print("Quit without saving.")
            break
        elif key == ord("z") and points:
            removed = points.pop()
            print(f"Undo point: {removed}")
        elif key == ord("r"):
            points.clear()
            print("All points reset.")
        elif key == ord("s"):
            if len(points) < 3:
                print("At least 3 points are required to form an ROI area.")
                continue
            with open(args.output, "w") as f:
                json.dump(points, f, indent=2)
            print(f"ROI saved to {args.output}: {points}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
