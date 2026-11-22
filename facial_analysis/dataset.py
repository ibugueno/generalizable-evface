"""FacEvent-compatible synchronized RGB/video and event dataset."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .events import convert_time, load_events, select_window, timestamp_scale
from .representations import make_representation


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _natural_key(path: Path):
    return [int(part) if part.isdigit() else part.lower() for part in __import__("re").split(r"(\d+)", path.name)]


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read RGB image {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_video(path: Path, frame_start: int = 0, frame_count: int | None = None) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open RGB video {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_start))
    images: list[np.ndarray] = []
    maximum = frame_count if frame_count is not None else 2**31 - 1
    try:
        for _ in range(maximum):
            ok, frame = capture.read()
            if not ok:
                break
            images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not images:
        raise RuntimeError(f"RGB video returned no frames: {path} starting at {frame_start}")
    return images


def _load_rgb_sequence(path: Path, frame_start: int = 0, frame_count: int | None = None) -> list[np.ndarray]:
    if path.is_dir():
        frames = sorted((item for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES), key=_natural_key)
        frames = frames[frame_start : None if frame_count is None else frame_start + frame_count]
        if not frames:
            raise RuntimeError(f"RGB frame directory returned no frames: {path}")
        return [_read_image(item) for item in frames]
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return [_read_image(path)]
    if path.suffix.lower() in VIDEO_SUFFIXES:
        return _read_video(path, frame_start, frame_count)
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            key = next((candidate for candidate in ("frames", "images", "rgb") if candidate in archive.files), archive.files[0])
            array = archive[key]
    else:
        # Hugging Face's local dataset layout may expose a downloaded video as
        # a suffix-less cache blob when a manifest was generated through
        # ``Path.resolve()``. OpenCV can identify the MP4 from its contents,
        # so keep old manifests usable while new manifests preserve the
        # original extension.
        if path.is_file():
            capture = cv2.VideoCapture(str(path))
            is_video = capture.isOpened()
            capture.release()
            if is_video:
                return _read_video(path, frame_start, frame_count)
        raise ValueError(f"unsupported RGB input {path}")
    if array.ndim == 3:
        array = array[None]
    array = array[frame_start : None if frame_count is None else frame_start + frame_count]
    if not len(array):
        raise RuntimeError(f"RGB array returned no frames: {path}")
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        raise ValueError(f"RGB array must have three channels, got {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array * 255.0 if np.max(array) <= 1.0 else array, 0, 255).astype(np.uint8)
    return [frame for frame in array]


def _prepare_rgb(frames: list[np.ndarray], image_size: int, sequence_length: int) -> torch.Tensor:
    if len(frames) >= sequence_length:
        indexes = np.linspace(0, len(frames) - 1, sequence_length).round().astype(np.int64)
        frames = [frames[index] for index in indexes]
    else:
        frames = frames + [frames[-1]] * (sequence_length - len(frames))
    processed = []
    for frame in frames:
        frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        processed.append(torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0)
    return torch.stack(processed)


def _load_target(path: Path, image_size: int) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        target = np.load(path, allow_pickle=False)
        target = np.asarray(target)
        if target.ndim == 3 and target.shape[-1] == 1:
            target = target[..., 0]
        if target.ndim != 2:
            raise ValueError(f"target array must be HxW or HxWx1, got {target.shape}")
        target = cv2.resize(target.astype(np.float32), (image_size, image_size), interpolation=cv2.INTER_AREA)
        return np.clip(target, -1.0, 1.0)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"could not read target {path}")
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 127.5 - 1.0


class FacEventReconstructionDataset(Dataset):
    """One manifest row is a synchronized RGB clip and event time window.

    The manifest is intentionally lightweight. Required columns are
    ``rgb_path`` and ``event_path``. Optional columns are ``frame_start``,
    ``frame_count``, ``event_start``, ``event_end``, ``event_height`` and
    ``event_width``. A ``target_path`` column can point to a pre-generated
    representation; otherwise the representation is generated on demand.
    """

    def __init__(
        self,
        manifest: str | Path,
        *,
        image_size: int,
        sequence_length: int,
        representation: str,
        event_height: int,
        event_width: int,
        timestamp_unit: str = "s",
        event_layout: str = "txyP",
        tbr_bins: int = 8,
        lads_tau: float | None = None,
        lads_rate_alpha: float = 1.0,
        limit: int | None = None,
    ):
        self.manifest = Path(manifest).resolve()
        self.base = self.manifest.parent
        with self.manifest.open(newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if limit is not None and limit >= 0:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f"manifest contains no rows: {self.manifest}")
        self.image_size = image_size
        self.sequence_length = sequence_length
        self.representation = representation
        self.event_height = event_height
        self.event_width = event_width
        self.timestamp_unit = timestamp_unit
        self.event_layout = event_layout
        self.tbr_bins = tbr_bins
        self.lads_tau = lads_tau
        self.lads_rate_alpha = lads_rate_alpha

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _number(row: dict[str, str], key: str, default: Any = None):
        value = row.get(key, "")
        if value in (None, ""):
            return default
        return float(value)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        rgb_value = row.get("rgb_path") or row.get("rgb") or row.get("video_path")
        event_value = row.get("event_path") or row.get("events")
        target_value = row.get("target_path", "")
        if not rgb_value or (not event_value and not target_value):
            raise ValueError("manifest rows require rgb_path and either event_path or target_path")
        rgb_path = _resolve(self.base, rgb_value)
        event_path = _resolve(self.base, event_value) if event_value else None
        frame_start = int(self._number(row, "frame_start", 0))
        frame_count = self._number(row, "frame_count")
        frame_count = int(frame_count) if frame_count is not None else None
        rgb = _prepare_rgb(_load_rgb_sequence(rgb_path, frame_start, frame_count), self.image_size, self.sequence_length)

        if target_value:
            target = _load_target(_resolve(self.base, target_value), self.image_size)
        else:
            row_time_unit = row.get("timestamp_unit") or self.timestamp_unit
            raw_event_start = self._number(row, "event_start")
            raw_event_end = self._number(row, "event_end")
            # HDF5 readers use these raw units to slice the huge FacEvent
            # stream before it reaches memory.
            events = load_events(
                event_path, self.event_layout,
                time_start=raw_event_start, time_end=raw_event_end,
            )
            events = convert_time(events, row_time_unit)
            time_scale = timestamp_scale(row_time_unit)
            event_start = raw_event_start
            event_end = raw_event_end
            if event_start is not None:
                event_start *= time_scale
            if event_end is not None:
                event_end *= time_scale
            events = select_window(events, event_start, event_end)
            height = int(self._number(row, "event_height", self.event_height))
            width = int(self._number(row, "event_width", self.event_width))
            kwargs = {"time_start": event_start, "time_end": event_end}
            if self.representation == "tbr":
                kwargs["bins"] = self.tbr_bins
            if self.representation == "lads":
                kwargs["tau"] = self.lads_tau
                kwargs["rate_alpha"] = self.lads_rate_alpha
            target = make_representation(self.representation, events, height, width, **kwargs)
            target = cv2.resize(target[..., 0], (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        return {
            "rgb": rgb,
            "target": torch.from_numpy(target.copy()).unsqueeze(0).float(),
            "sample_id": row.get("sample_id", str(index)),
        }
