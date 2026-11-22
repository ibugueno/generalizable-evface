"""Single-channel grayscale event representations.

All functions return ``float32`` arrays with shape ``(H, W, 1)``. Values are
kept in ``[-1, 1]`` so that the reconstruction target has one common range.
The definitions are intentionally explicit and configurable; they are a first
baseline, not a claim that every paper's implementation is identical.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _validate(events: np.ndarray, height: int, width: int) -> np.ndarray:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError(f"expected events with shape (N, 4), got {events.shape}")
    if events.size == 0:
        return events.astype(np.float32, copy=False)
    valid = (
        np.isfinite(events).all(axis=1)
        & (events[:, 1] >= 0)
        & (events[:, 1] < width)
        & (events[:, 2] >= 0)
        & (events[:, 2] < height)
    )
    return events[valid]


def _finish(image: np.ndarray) -> np.ndarray:
    image = np.nan_to_num(image.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(image, -1.0, 1.0)[..., None]


def _time_bounds(events: np.ndarray, time_start: float | None, time_end: float | None) -> tuple[float, float]:
    if events.size == 0:
        start = 0.0 if time_start is None else float(time_start)
        end = start + 1.0 if time_end is None else float(time_end)
    else:
        start = float(events[:, 0].min()) if time_start is None else float(time_start)
        end = float(events[:, 0].max()) if time_end is None else float(time_end)
    if end <= start:
        end = start + 1.0
    return start, end


def _normalize_signed(image: np.ndarray, scale: float | None = None) -> np.ndarray:
    if scale is None:
        scale = float(np.percentile(np.abs(image), 99.0)) if np.any(image) else 1.0
    return image / max(float(scale), 1e-6)


def accumulated_events(
    events: np.ndarray,
    height: int,
    width: int,
    *,
    time_start: float | None = None,
    time_end: float | None = None,
) -> np.ndarray:
    """ACC: signed polarity accumulation, robustly normalized to ``[-1, 1]``."""

    events = _validate(events, height, width)
    image = np.zeros((height, width), dtype=np.float32)
    if len(events):
        np.add.at(image, (events[:, 2].astype(np.int64), events[:, 1].astype(np.int64)), events[:, 3])
    return _finish(_normalize_signed(image))


def surface_of_active_events(
    events: np.ndarray,
    height: int,
    width: int,
    *,
    time_start: float | None = None,
    time_end: float | None = None,
) -> np.ndarray:
    """SAE: signed recency of the most recent event at each pixel.

    A recent positive event is close to ``+1`` and a recent negative event is
    close to ``-1``. Empty pixels are zero. This is a compact single-channel
    variant; polarity-separated SAE planes are intentionally out of scope for
    the common ``H x W x 1`` interface in this first implementation.
    """

    events = _validate(events, height, width)
    start, end = _time_bounds(events, time_start, time_end)
    last_time = np.full((height, width), -np.inf, dtype=np.float32)
    last_polarity = np.zeros((height, width), dtype=np.float32)
    if len(events):
        for timestamp, x, y, polarity in events:
            xi, yi = int(x), int(y)
            if timestamp >= last_time[yi, xi]:
                last_time[yi, xi] = timestamp
                last_polarity[yi, xi] = polarity
    valid = np.isfinite(last_time)
    surface = np.zeros((height, width), dtype=np.float32)
    age = (end - last_time[valid]) / max(end - start, 1e-6)
    surface[valid] = last_polarity[valid] * (1.0 - np.maximum(age, 0.0))
    return _finish(surface)


def temporal_binary_representation(
    events: np.ndarray,
    height: int,
    width: int,
    *,
    bins: int = 8,
    time_start: float | None = None,
    time_end: float | None = None,
) -> np.ndarray:
    """TBR: compact temporal activity bits into one scalar channel.

    Each bit indicates whether any event was observed at a pixel in the
    corresponding temporal interval. The integer bit mask is divided by its
    maximum possible value, yielding a grayscale value in ``[0, 1]``. This
    keeps the representation single-channel and makes its bin count explicit.
    """

    if not 1 <= bins <= 31:
        raise ValueError("TBR bins must be between 1 and 31")
    events = _validate(events, height, width)
    start, end = _time_bounds(events, time_start, time_end)
    result = np.zeros((height, width), dtype=np.uint32)
    if len(events):
        indices = np.floor((events[:, 0] - start) * bins / (end - start)).astype(np.int64)
        indices = np.clip(indices, 0, bins - 1)
        ys = events[:, 2].astype(np.int64)
        xs = events[:, 1].astype(np.int64)
        for y, x, index in zip(ys, xs, indices):
            result[y, x] |= np.uint32(1 << int(index))
    return _finish(result.astype(np.float32) / float((1 << bins) - 1))


def locally_adaptive_decay_surface(
    events: np.ndarray,
    height: int,
    width: int,
    *,
    tau: float | None = None,
    rate_alpha: float = 1.0,
    time_start: float | None = None,
    time_end: float | None = None,
) -> np.ndarray:
    """Event-rate LADS baseline with a per-pixel decay constant.

    The LADS paper describes several local-dynamics strategies. This first
    implementation uses the event-rate strategy: pixels with higher local
    event rates receive shorter decay constants, while quiet pixels retain
    their last event for longer. The method is exposed as a separate function
    so a Laplacian or spectral strategy can replace it later without changing
    dataset or model code.
    """

    events = _validate(events, height, width)
    start, end = _time_bounds(events, time_start, time_end)
    duration = max(end - start, 1e-6)
    if tau is None:
        tau = duration / 5.0
    if tau <= 0:
        raise ValueError("LADS tau must be positive")
    last_time = np.full((height, width), -np.inf, dtype=np.float32)
    last_polarity = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    if len(events):
        for timestamp, x, y, polarity in events:
            xi, yi = int(x), int(y)
            counts[yi, xi] += 1.0
            if timestamp >= last_time[yi, xi]:
                last_time[yi, xi] = timestamp
                last_polarity[yi, xi] = polarity

    positive_rates = counts[counts > 0] / duration
    reference_rate = float(np.median(positive_rates)) if len(positive_rates) else 1.0
    relative_rate = np.zeros_like(counts)
    relative_rate[counts > 0] = (counts[counts > 0] / duration) / max(reference_rate, 1e-6)
    relative_rate = np.clip(relative_rate, 0.25, 4.0)
    local_tau = float(tau) / (1.0 + max(rate_alpha, 0.0) * np.maximum(relative_rate - 1.0, 0.0))
    age = np.maximum(0.0, end - last_time)
    surface = np.where(np.isfinite(last_time), last_polarity * np.exp(-age / np.maximum(local_tau, 1e-6)), 0.0)
    return _finish(surface)


REPRESENTATIONS: dict[str, Callable[..., np.ndarray]] = {
    "acc": accumulated_events,
    "sae": surface_of_active_events,
    "tbr": temporal_binary_representation,
    "lads": locally_adaptive_decay_surface,
}


def make_representation(name: str, events: np.ndarray, height: int, width: int, **kwargs) -> np.ndarray:
    try:
        function = REPRESENTATIONS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown representation {name!r}; choose from {sorted(REPRESENTATIONS)}") from exc
    return function(events, height, width, **kwargs)
