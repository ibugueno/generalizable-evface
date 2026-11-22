#!/usr/bin/env python3
"""Create a lightweight FacEvent clip-window manifest without loading events.

FacEvent stores 640x640 face-crop videos beside HDF5 event streams under
``naturalistic/train/{batch}/{archive}``. The resulting CSV creates one row per
video window; event windows are expressed in microseconds because FacEvent's
HDF5 timestamps use microseconds.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-clips", type=int, default=-1)
    parser.add_argument("--max-windows", type=int, default=-1)
    args = parser.parse_args()
    if args.sequence_length <= 0 or args.stride <= 0:
        parser.error("sequence length and stride must be positive")

    videos = sorted(args.dataset_root.rglob("*_face640.mp4"))
    if args.max_clips >= 0:
        videos = videos[: args.max_clips]
    if not videos:
        raise SystemExit(f"no *_face640.mp4 files found under {args.dataset_root}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for video in tqdm(videos, desc="FacEvent clips"):
        event_path = video.with_name(video.name.replace("_face640.mp4", "_face640_events.h5"))
        if not event_path.is_file():
            continue
        capture = cv2.VideoCapture(str(video))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        capture.release()
        if frame_count <= 0:
            continue
        for start_frame in range(0, max(1, frame_count - args.sequence_length + 1), args.stride):
            end_frame = min(start_frame + args.sequence_length, frame_count)
            rows.append(
                {
                    "sample_id": f"{video.stem}:{start_frame}",
                    # Keep the mounted dataset path instead of resolving its
                    # symlink into Hugging Face's suffix-less cache blob.
                    # Absolute paths remain independent of the manifest
                    # directory while retaining .mp4/.h5 for loaders.
                    "rgb_path": str(video),
                    "event_path": str(event_path),
                    "frame_start": start_frame,
                    "frame_count": end_frame - start_frame,
                    "event_start": start_frame * 1_000_000.0 / fps,
                    "event_end": end_frame * 1_000_000.0 / fps,
                    "event_height": 640,
                    "event_width": 640,
                    "timestamp_unit": "us",
                }
            )
            if args.max_windows >= 0 and len(rows) >= args.max_windows:
                break
        if args.max_windows >= 0 and len(rows) >= args.max_windows:
            break

    if not rows:
        raise SystemExit("no synchronized FacEvent video/event pairs found")
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} windows to {args.output}")


if __name__ == "__main__":
    main()
