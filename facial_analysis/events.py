"""Loading and canonicalizing common event-stream file formats.

The canonical representation used by this project is an ``(N, 4)`` float32
array with columns ``[t, x, y, polarity]``. Time is kept in the units stored in
the file; callers can convert it explicitly with ``timestamp_unit``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np


SUPPORTED_SUFFIXES = {".npy", ".npz", ".csv", ".txt", ".h5", ".hdf5"}


def _first_present(mapping: dict, names: Iterable[str]):
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _canonicalize(t, x, y, p) -> np.ndarray:
    arrays = [np.asarray(value).reshape(-1) for value in (t, x, y, p)]
    if len({len(value) for value in arrays}) != 1:
        raise ValueError("event columns have different lengths")
    if len(arrays[0]) == 0:
        return np.empty((0, 4), dtype=np.float32)

    result = np.column_stack(arrays).astype(np.float32, copy=False)
    # Accept both {0, 1} and {-1, +1} polarity conventions.
    result[:, 3] = np.where(result[:, 3] > 0, 1.0, -1.0)
    order = np.argsort(result[:, 0], kind="stable")
    return result[order]


def _from_matrix(matrix: np.ndarray, layout: str) -> np.ndarray:
    matrix = np.asarray(matrix)
    if matrix.dtype.names:
        fields = {name.lower(): matrix[name] for name in matrix.dtype.names}
        t = _first_present(fields, ("t", "ts", "time", "timestamp"))
        x = _first_present(fields, ("x", "x_pos", "x_coordinate"))
        y = _first_present(fields, ("y", "y_pos", "y_coordinate"))
        p = _first_present(fields, ("p", "polarity", "pol", "sign"))
        if any(value is None for value in (t, x, y, p)):
            raise ValueError(f"structured event array is missing one of t/x/y/p: {matrix.dtype.names}")
        return _canonicalize(t, x, y, p)

    if matrix.ndim != 2 or matrix.shape[1] < 4:
        raise ValueError(f"expected an event matrix with at least four columns, got {matrix.shape}")
    layout = layout.replace(" ", "").lower()
    if len(layout) != 4 or set(layout) != {"t", "x", "y", "p"}:
        raise ValueError("layout must be a permutation of txyP, for example txyP or xy tP without spaces")
    columns = {symbol: matrix[index] for index, symbol in enumerate(layout)}
    return _canonicalize(columns["t"], columns["x"], columns["y"], columns["p"])


def _load_npz(path: Path, layout: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        keys = {key.lower(): key for key in archive.files}
        t = _first_present({key: archive[value] for key, value in keys.items()}, ("t", "ts", "time", "timestamp"))
        x = _first_present({key: archive[value] for key, value in keys.items()}, ("x", "x_pos"))
        y = _first_present({key: archive[value] for key, value in keys.items()}, ("y", "y_pos"))
        p = _first_present({key: archive[value] for key, value in keys.items()}, ("p", "polarity", "pol", "sign"))
        if all(value is not None for value in (t, x, y, p)):
            return _canonicalize(t, x, y, p)
        if len(archive.files) == 1:
            return _from_matrix(archive[archive.files[0]], layout)
        raise ValueError(f"could not identify t/x/y/p arrays in {path}; keys={archive.files}")


def _h5_bound(dataset, value: float, *, right: bool = False) -> int:
    """Binary-search a sorted HDF5 timestamp dataset without reading it all."""

    low, high = 0, int(dataset.shape[0])
    while low < high:
        middle = (low + high) // 2
        current = float(dataset[middle])
        if current < value or (right and current == value):
            low = middle + 1
        else:
            high = middle
    return low


def _load_h5(path: Path, layout: str, time_start: float | None = None, time_end: float | None = None) -> np.ndarray:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - only used outside the image
        raise RuntimeError("h5py is required to load HDF5 event files") from exc

    with h5py.File(path, "r") as handle:
        datasets = {}

        def visit(name, node):
            if hasattr(node, "shape") and len(node.shape) > 0:
                datasets[name.lower()] = node

        handle.visititems(visit)
        by_leaf = {name.rsplit("/", 1)[-1]: value for name, value in datasets.items()}
        t = _first_present(by_leaf, ("t", "ts", "time", "timestamp"))
        x = _first_present(by_leaf, ("x", "x_pos"))
        y = _first_present(by_leaf, ("y", "y_pos"))
        p = _first_present(by_leaf, ("p", "polarity", "pol", "sign"))
        if all(value is not None for value in (t, x, y, p)):
            start_index = 0 if time_start is None else _h5_bound(t, time_start)
            end_index = int(t.shape[0]) if time_end is None else _h5_bound(t, time_end)
            if end_index <= start_index:
                return np.empty((0, 4), dtype=np.float32)
            return _canonicalize(
                t[start_index:end_index], x[start_index:end_index],
                y[start_index:end_index], p[start_index:end_index],
            )
        # Some exporters store one dataset with shape (N, 4).
        for dataset in datasets.values():
            value = np.asarray(dataset)
            if value.ndim == 2 and value.shape[1] >= 4:
                return _from_matrix(value, layout)
    raise ValueError(f"could not identify an event dataset in {path}")


def _load_text(path: Path, layout: str) -> np.ndarray:
    # Handle named CSV columns first. For plain text, NumPy accepts whitespace
    # and an explicitly supplied comma delimiter below.
    with path.open("r", newline="") as handle:
        first_line = handle.readline()
    header = [part.strip().lower() for part in first_line.replace(";", ",").split(",")]
    named = {name: index for index, name in enumerate(header)}
    aliases = {
        "t": ("t", "ts", "time", "timestamp"),
        "x": ("x", "x_pos", "x_coordinate"),
        "y": ("y", "y_pos", "y_coordinate"),
        "p": ("p", "polarity", "pol", "sign"),
    }
    if all(any(alias in named for alias in names) for names in aliases.values()):
        rows = np.genfromtxt(path, delimiter="," if "," in first_line else None, names=True)
        fields = {name.lower(): rows[name] for name in rows.dtype.names or ()}
        return _canonicalize(
            _first_present(fields, aliases["t"]),
            _first_present(fields, aliases["x"]),
            _first_present(fields, aliases["y"]),
            _first_present(fields, aliases["p"]),
        )
    matrix = np.loadtxt(path, delimiter="," if "," in first_line else None, ndmin=2)
    return _from_matrix(matrix, layout)


def load_events(
    path: str | Path,
    layout: str = "txyP",
    *,
    time_start: float | None = None,
    time_end: float | None = None,
) -> np.ndarray:
    """Load an event stream into sorted ``[t, x, y, polarity]`` rows."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return _load_npz(path, layout)
    if suffix == ".npy":
        return _from_matrix(np.load(path, allow_pickle=False), layout)
    if suffix in {".csv", ".txt"}:
        return _load_text(path, layout)
    if suffix in {".h5", ".hdf5"}:
        return _load_h5(path, layout, time_start, time_end)
    raise ValueError(f"unsupported event file extension {suffix}; supported={sorted(SUPPORTED_SUFFIXES)}")


def timestamp_scale(unit: str) -> float:
    scales = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
    if unit not in scales:
        raise ValueError(f"timestamp unit must be one of {sorted(scales)}, got {unit!r}")
    return scales[unit]


def select_window(events: np.ndarray, start: float | None = None, end: float | None = None) -> np.ndarray:
    """Select a half-open time window ``[start, end)``."""

    if events.size == 0:
        return events
    mask = np.ones(len(events), dtype=bool)
    if start is not None:
        mask &= events[:, 0] >= start
    if end is not None:
        mask &= events[:, 0] < end
    return events[mask]


def convert_time(events: np.ndarray, unit: str) -> np.ndarray:
    result = events.copy()
    result[:, 0] *= timestamp_scale(unit)
    return result
