#!/usr/bin/env python3
"""Create FPS-aligned RGB/grayscale/event pairs from FacEvent clips.

For video frame ``i`` the event interval is exactly ``[i/fps, (i+1)/fps)``.
The HDF5 reader slices that interval directly, so the complete event stream is
never loaded into RAM. By default this script writes manifests only and keeps
RGB data in the original MP4 and event targets on demand. Optional export flags
materialize RGB/grayscale frames or representation arrays under the persistent
output directory.

Typical server workflow:

  python scripts/preprocess_facevent.py \
      --dataset-root /app/data/FacEvent/naturalistic/train \
      --output-root /app/persistent/facevent_preprocessed \
      --representations acc sae tbr lads \
      --sequence-length 8 --max-clips 1 --max-frames 128

To materialize a small inspection export, add ``--write-event-representations
--write-event-png --export-rgb-frames --export-gray-frames``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from facial_analysis.representations import REPRESENTATIONS, make_representation


def _bound(dataset, value: float) -> int:
    low, high = 0, int(dataset.shape[0])
    while low < high:
        middle = (low + high) // 2
        if float(dataset[middle]) < value:
            low = middle + 1
        else:
            high = middle
    return low


class EventWindowReader:
    """Random-access FacEvent HDF5 reader that only materializes one window."""

    def __init__(self, path: Path):
        self.path = path
        self.handle = h5py.File(path, "r")
        group = self.handle["events"] if "events" in self.handle else self.handle
        self.t = group["t"]
        self.x = group["x"]
        self.y = group["y"]
        self.p = group["p"]
        self.height = int(group.attrs.get("height", self.handle.attrs.get("height", 640)))
        self.width = int(group.attrs.get("width", self.handle.attrs.get("width", 640)))

    def read(self, start_us: float, end_us: float) -> np.ndarray:
        left = _bound(self.t, start_us)
        right = _bound(self.t, end_us)
        if right <= left:
            return np.empty((0, 4), dtype=np.float32)
        return np.column_stack((
            np.asarray(self.t[left:right]), np.asarray(self.x[left:right]),
            np.asarray(self.y[left:right]), np.asarray(self.p[left:right]),
        )).astype(np.float32, copy=False)

    def close(self) -> None:
        self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.close()


def _clip_key(video: Path, root: Path) -> str:
    return str(video.relative_to(root).with_suffix(""))


def _output_path(root: Path, category: str, representation: str, clip_key: str, frame_index: int, suffix: str) -> Path:
    path = root / category
    if representation:
        path /= representation
    path /= clip_key
    path.mkdir(parents=True, exist_ok=True)
    return path / f"frame_{frame_index:06d}{suffix}"


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _preview(array: np.ndarray) -> np.ndarray:
    image = array[..., 0] if array.ndim == 3 else array
    return ((np.clip(image, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--representations", nargs="+", choices=sorted(REPRESENTATIONS), default=["acc"])
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1, help="frame stride for generated rows")
    parser.add_argument("--fps", type=float, help="override video FPS; default reads it from each MP4")
    parser.add_argument("--output-size", type=int, default=224)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-clips", type=int, default=-1)
    parser.add_argument("--max-frames", type=int, default=-1, help="global frame limit across all clips")
    parser.add_argument("--tbr-bins", type=int, default=8)
    parser.add_argument("--lads-tau-us", type=float)
    parser.add_argument("--lads-rate-alpha", type=float, default=1.0)
    parser.add_argument("--write-event-representations", action="store_true")
    parser.add_argument("--write-event-png", action="store_true")
    parser.add_argument("--export-rgb-frames", action="store_true")
    parser.add_argument("--export-gray-frames", action="store_true")
    args = parser.parse_args()
    if args.sequence_length <= 0 or args.stride <= 0 or args.output_size <= 0:
        parser.error("sequence length, stride and output size must be positive")

    videos = sorted(args.dataset_root.rglob("*_face640.mp4"))
    if args.max_clips >= 0:
        videos = videos[: args.max_clips]
    if not videos:
        raise SystemExit(f"no *_face640.mp4 files found under {args.dataset_root}")

    pair_rows: list[dict] = []
    reconstruction_rows: list[dict] = []
    processed_frames = 0
    for video in tqdm(videos, desc="FacEvent clips"):
        event_path = video.with_name(video.name.replace("_face640.mp4", "_face640_events.h5"))
        if not event_path.is_file():
            print(f"[SKIP] missing events: {event_path}")
            continue
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            print(f"[SKIP] cannot open video: {video}")
            continue
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = args.fps or float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        frame_start = min(max(0, args.start_frame), max(0, frame_count - 1))
        frame_stop = frame_count if args.max_frames < 0 else min(frame_count, frame_start + args.max_frames - processed_frames)
        clip_key = _clip_key(video, args.dataset_root)
        with EventWindowReader(event_path) as reader:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
            for frame_index in range(frame_start, frame_stop):
                ok, bgr = capture.read()
                if not ok:
                    break
                start_us = frame_index * 1_000_000.0 / fps
                end_us = (frame_index + 1) * 1_000_000.0 / fps
                events = reader.read(start_us, end_us)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb_resized = cv2.resize(rgb, (args.output_size, args.output_size), interpolation=cv2.INTER_AREA)
                gray_resized = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2GRAY)

                rgb_path = ""
                gray_path = ""
                if args.export_rgb_frames:
                    path = _output_path(args.output_root, "rgb_frames", "", clip_key, frame_index, ".png")
                    cv2.imwrite(str(path), cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR))
                    rgb_path = str(path.resolve())
                if args.export_gray_frames:
                    path = _output_path(args.output_root, "gray_frames", "", clip_key, frame_index, ".png")
                    cv2.imwrite(str(path), gray_resized)
                    gray_path = str(path.resolve())

                for representation in args.representations:
                    kwargs = {"time_start": start_us, "time_end": end_us}
                    if representation == "tbr":
                        kwargs["bins"] = args.tbr_bins
                    if representation == "lads":
                        kwargs["tau"] = args.lads_tau_us
                        kwargs["rate_alpha"] = args.lads_rate_alpha
                    target_path = ""
                    if args.write_event_representations or args.write_event_png:
                        target = make_representation(representation, events, reader.height, reader.width, **kwargs)
                        target = cv2.resize(target[..., 0], (args.output_size, args.output_size), interpolation=cv2.INTER_AREA)[..., None]
                        path = _output_path(args.output_root, "event_representations", representation, clip_key, frame_index, ".npy")
                        if args.write_event_representations:
                            np.save(path, target.astype(np.float32))
                            target_path = str(path.resolve())
                        if args.write_event_png:
                            cv2.imwrite(str(path.with_suffix(".png")), _preview(target))

                    common = {
                        "sample_id": f"{clip_key}:{frame_index}:{representation}",
                        "clip_key": clip_key,
                        # Do not resolve the input symlink. FacEvent downloaded
                        # through Hugging Face uses suffix-less cache blobs,
                        # while the mounted dataset path retains .mp4/.h5 and
                        # lets downstream loaders identify the file format.
                        "rgb_path": rgb_path or str(video),
                        "gray_path": gray_path,
                        "event_path": target_path or str(event_path),
                        "target_path": target_path,
                        "representation": representation,
                        "frame_index": frame_index,
                        "frame_fps": fps,
                        "event_start": start_us,
                        "event_end": end_us,
                        "event_height": reader.height,
                        "event_width": reader.width,
                        "timestamp_unit": "us",
                        "num_events": len(events),
                        "frame_start": frame_index,
                        "frame_count": 1,
                    }
                    pair_rows.append(common)
                    context_start = max(frame_start, frame_index - args.sequence_length + 1)
                    reconstruction_row = dict(common)
                    reconstruction_row["sample_id"] = f"{clip_key}:{frame_index}:{representation}:sequence"
                    reconstruction_row["rgb_path"] = str(video)
                    reconstruction_row["frame_start"] = context_start
                    reconstruction_row["frame_count"] = frame_index - context_start + 1
                    reconstruction_rows.append(reconstruction_row)
                processed_frames += 1
                if args.max_frames >= 0 and processed_frames >= args.max_frames:
                    break
        capture.release()
        if args.max_frames >= 0 and processed_frames >= args.max_frames:
            break

    if not pair_rows:
        raise SystemExit("no FacEvent pairs were produced")
    _write_csv(args.output_root / "pairs.csv", pair_rows)
    _write_csv(args.output_root / "reconstruction_manifest.csv", reconstruction_rows)
    for representation in args.representations:
        pair_subset = [row for row in pair_rows if row["representation"] == representation]
        reconstruction_subset = [row for row in reconstruction_rows if row["representation"] == representation]
        _write_csv(args.output_root / f"pairs_{representation}.csv", pair_subset)
        _write_csv(args.output_root / f"reconstruction_manifest_{representation}.csv", reconstruction_subset)
    metadata = {
        "dataset_root": str(args.dataset_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "representations": args.representations,
        "fps_window": "[frame_index / fps, (frame_index + 1) / fps)",
        "processed_frames": processed_frames,
        "pairs": len(pair_rows),
        "write_event_representations": args.write_event_representations,
        "export_rgb_frames": args.export_rgb_frames,
        "export_gray_frames": args.export_gray_frames,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "preprocessing.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {len(pair_rows)} pair rows and {len(reconstruction_rows)} reconstruction rows")


if __name__ == "__main__":
    main()
