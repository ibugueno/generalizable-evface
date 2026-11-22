#!/usr/bin/env python3
"""Generate ACC, SAE, TBR and/or LADS files from event streams.

Examples inside the container:

    python scripts/generate_representations.py \
      --input-root /app/data/events --output-root /app/persistent/representations \
      --height 640 --width 640 --timestamp-unit us --representations acc sae tbr lads
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from tqdm import tqdm

from facial_analysis.events import SUPPORTED_SUFFIXES, convert_time, load_events, select_window, timestamp_scale
from facial_analysis.representations import REPRESENTATIONS, make_representation


def _visualization(array: np.ndarray) -> np.ndarray:
    image = array[..., 0] if array.ndim == 3 else array
    image = ((np.clip(image, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8)
    return image


def _iter_inputs(root: Path, recursive: bool):
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="one event file")
    parser.add_argument("--input-root", type=Path, help="directory containing event files")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--representations", nargs="+", choices=sorted(REPRESENTATIONS), default=sorted(REPRESENTATIONS))
    parser.add_argument("--timestamp-unit", choices=("s", "ms", "us", "ns"), default="s")
    parser.add_argument("--event-layout", default="txyP", help="column order for matrix files, e.g. txyP or xy tP")
    parser.add_argument("--time-start", type=float)
    parser.add_argument("--time-end", type=float)
    parser.add_argument("--tbr-bins", type=int, default=8)
    parser.add_argument("--lads-tau", type=float)
    parser.add_argument("--lads-rate-alpha", type=float, default=1.0)
    parser.add_argument("--save-png", action="store_true")
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    if bool(args.input) == bool(args.input_root):
        parser.error("provide exactly one of --input or --input-root")
    paths = [args.input] if args.input else _iter_inputs(args.input_root, args.recursive)
    if not paths:
        raise SystemExit("no event files found")
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    for path in tqdm(paths, desc="representations"):
        raw_start = args.time_start
        raw_end = args.time_end
        if path.suffix.lower() in {".h5", ".hdf5"} and raw_start is None and raw_end is None:
            raise RuntimeError(
                f"refusing to materialize an unbounded HDF5 event stream: {path}; "
                "provide --time-start/--time-end or use the FacEvent reconstruction manifest"
            )
        events = load_events(path, args.event_layout, time_start=raw_start, time_end=raw_end)
        events = convert_time(events, args.timestamp_unit)
        scale = timestamp_scale(args.timestamp_unit)
        start = args.time_start * scale if args.time_start is not None else None
        end = args.time_end * scale if args.time_end is not None else None
        selected = select_window(events, start, end)
        relative = path.name if args.input else path.relative_to(args.input_root)
        stem = relative.with_suffix("")
        outputs = {}
        for name in args.representations:
            kwargs = {"time_start": start, "time_end": end}
            if name == "tbr":
                kwargs["bins"] = args.tbr_bins
            if name == "lads":
                kwargs["tau"] = args.lads_tau
                kwargs["rate_alpha"] = args.lads_rate_alpha
            image = make_representation(name, selected, args.height, args.width, **kwargs)
            output = args.output_root / name / stem
            output = output.with_suffix(".npy")
            output.parent.mkdir(parents=True, exist_ok=True)
            np.save(output, image.astype(np.float32))
            if args.save_png:
                cv2.imwrite(str(output.with_suffix(".png")), _visualization(image))
            outputs[name] = str(output)
        manifest.append({"input": str(path), "outputs": outputs, "events": int(len(selected))})

    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
