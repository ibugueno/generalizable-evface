#!/usr/bin/env python3
"""Write reconstructed grayscale representations for a manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from facial_analysis.dataset import FacEventReconstructionDataset
from facial_analysis.model import RGBToEventReconstruction, build_rgb_teacher, infer_teacher_channels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no GPU is visible")
        torch.cuda.set_device(device)
    teacher = build_rgb_teacher(
        config["teacher_type"], checkpoint=config.get("teacher_checkpoint"),
        faceptor_factory=config.get("faceptor_factory"),
    ).to(device)
    model = RGBToEventReconstruction(teacher, infer_teacher_channels(teacher, config["image_size"]), config["image_size"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = FacEventReconstructionDataset(
        args.manifest, image_size=config["image_size"], sequence_length=config["sequence_length"],
        representation=config["representation"], event_height=config["event_height"], event_width=config["event_width"],
        timestamp_unit=config["timestamp_unit"], event_layout=config["event_layout"], tbr_bins=config["tbr_bins"],
        lads_tau=config["lads_tau"], lads_rate_alpha=config["lads_rate_alpha"], limit=args.limit,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="reconstruct"):
            prediction = model(batch["rgb"].to(device)).cpu().numpy()
            for sample_id, value in zip(batch["sample_id"], prediction[:, 0]):
                safe_id = str(sample_id).replace("/", "_").replace(":", "_")
                npy_path = args.output_dir / f"{safe_id}.npy"
                png_path = args.output_dir / f"{safe_id}.png"
                np.save(npy_path, value[..., None].astype(np.float32))
                preview = ((np.clip(value, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8)
                cv2.imwrite(str(png_path), preview)
                records.append({"sample_id": str(sample_id), "npy": str(npy_path), "png": str(png_path)})
    (args.output_dir / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
