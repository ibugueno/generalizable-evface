#!/usr/bin/env python3
"""Train the first RGB-to-event reconstruction baseline on one visible GPU."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from facial_analysis.dataset import FacEventReconstructionDataset
from facial_analysis.model import RGBToEventReconstruction, build_rgb_teacher, infer_teacher_channels


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest")
    parser.add_argument("--output-dir")
    parser.add_argument("--representation", choices=("acc", "sae", "tbr", "lads"))
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--event-height", type=int)
    parser.add_argument("--event-width", type=int)
    parser.add_argument("--timestamp-unit", choices=("s", "ms", "us", "ns"))
    parser.add_argument("--event-layout")
    parser.add_argument("--tbr-bins", type=int)
    parser.add_argument("--lads-tau", type=float)
    parser.add_argument("--lads-rate-alpha", type=float)
    parser.add_argument("--teacher-type", choices=("faceptor", "resnet18", "tiny"))
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--faceptor-factory")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--l1-weight", type=float)
    parser.add_argument("--mse-weight", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-steps-per-epoch", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume")
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=None)
    return parser


def _load_config(args: argparse.Namespace) -> dict:
    config = {}
    if args.config:
        with args.config.open() as handle:
            config = yaml.safe_load(handle) or {}
    mapping = {
        "manifest": "manifest",
        "output_dir": "output_dir",
        "representation": "representation",
        "image_size": "image_size",
        "sequence_length": "sequence_length",
        "event_height": "event_height",
        "event_width": "event_width",
        "timestamp_unit": "timestamp_unit",
        "event_layout": "event_layout",
        "tbr_bins": "tbr_bins",
        "lads_tau": "lads_tau",
        "lads_rate_alpha": "lads_rate_alpha",
        "teacher_type": "teacher_type",
        "teacher_checkpoint": "teacher_checkpoint",
        "faceptor_factory": "faceptor_factory",
        "epochs": "epochs",
        "batch_size": "batch_size",
        "num_workers": "num_workers",
        "learning_rate": "learning_rate",
        "weight_decay": "weight_decay",
        "l1_weight": "l1_weight",
        "mse_weight": "mse_weight",
        "limit": "limit",
        "max_steps_per_epoch": "max_steps_per_epoch",
        "seed": "seed",
        "amp": "amp",
    }
    # A nested training section is accepted for natural YAML organization.
    merged = dict(config)
    merged.update(config.get("training", {}))
    values = {}
    for argument, key in mapping.items():
        value = getattr(args, argument)
        values[key] = value if value is not None else merged.get(key)
    defaults = {
        "representation": "acc", "image_size": 128, "sequence_length": 8,
        "event_height": 640, "event_width": 640, "timestamp_unit": "s",
        "event_layout": "txyP", "tbr_bins": 8, "lads_rate_alpha": 1.0,
        "teacher_type": "faceptor", "epochs": 10, "batch_size": 4,
        "num_workers": 4, "learning_rate": 1e-4, "weight_decay": 0.0,
        "l1_weight": 1.0, "mse_weight": 0.0, "limit": None,
        "max_steps_per_epoch": -1, "seed": 44, "amp": True,
    }
    for key, value in defaults.items():
        values[key] = values[key] if values[key] is not None else value
    if not values["manifest"] or not values["output_dir"]:
        raise SystemExit("--manifest and --output-dir are required (or must be present in --config)")
    return values


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_checkpoint(path: Path, epoch: int, model, optimizer, scaler, loss: float, config: dict):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "train_loss": loss,
        "config": config,
    }, path)


def main() -> None:
    args = _parser().parse_args()
    config = _load_config(args)
    _seed_everything(config["seed"])
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no GPU is visible")
        torch.cuda.set_device(device)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps({**config, "device": str(device)}, indent=2) + "\n")

    dataset = FacEventReconstructionDataset(
        config["manifest"], image_size=config["image_size"], sequence_length=config["sequence_length"],
        representation=config["representation"], event_height=config["event_height"], event_width=config["event_width"],
        timestamp_unit=config["timestamp_unit"], event_layout=config["event_layout"], tbr_bins=config["tbr_bins"],
        lads_tau=config["lads_tau"], lads_rate_alpha=config["lads_rate_alpha"], limit=config["limit"],
    )
    loader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"],
        pin_memory=device.type == "cuda", persistent_workers=config["num_workers"] > 0,
    )

    teacher = build_rgb_teacher(
        config["teacher_type"], checkpoint=config["teacher_checkpoint"], faceptor_factory=config["faceptor_factory"],
    ).to(device)
    teacher_channels = infer_teacher_channels(teacher, config["image_size"])
    model = RGBToEventReconstruction(teacher, teacher_channels, config["image_size"]).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config["amp"] and device.type == "cuda"))

    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_loss = float(checkpoint.get("train_loss", best_loss))

    metrics_path = output_dir / "train_metrics.csv"
    write_header = not metrics_path.exists() or start_epoch == 0
    with metrics_path.open("a", newline="") as metrics_handle:
        metrics_writer = csv.DictWriter(metrics_handle, fieldnames=["epoch", "train_loss", "samples"])
        if write_header:
            metrics_writer.writeheader()
        for epoch in range(start_epoch, config["epochs"]):
            model.train()
            total_loss = 0.0
            samples = 0
            progress = tqdm(loader, desc=f"epoch {epoch + 1}/{config['epochs']}")
            for step, batch in enumerate(progress):
                if config["max_steps_per_epoch"] >= 0 and step >= config["max_steps_per_epoch"]:
                    break
                rgb = batch["rgb"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    prediction = model(rgb)
                    loss = config["l1_weight"] * F.l1_loss(prediction, target)
                    if config["mse_weight"]:
                        loss = loss + config["mse_weight"] * F.mse_loss(prediction, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                batch_size = rgb.shape[0]
                samples += batch_size
                total_loss += float(loss.detach()) * batch_size
                progress.set_postfix(loss=f"{float(loss.detach()):.5f}")
            if samples == 0:
                raise RuntimeError("no training batches were processed")
            epoch_loss = total_loss / samples
            metrics_writer.writerow({"epoch": epoch, "train_loss": epoch_loss, "samples": samples})
            metrics_handle.flush()
            _save_checkpoint(output_dir / "last.pt", epoch, model, optimizer, scaler, epoch_loss, config)
            if epoch_loss <= best_loss:
                best_loss = epoch_loss
                _save_checkpoint(output_dir / "best.pt", epoch, model, optimizer, scaler, epoch_loss, config)
            print(f"epoch={epoch} train_loss={epoch_loss:.6f} samples={samples}")


if __name__ == "__main__":
    main()
