"""Past-only sensor windows built before deduplicating source observations."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forge.data.datasets import FEATURES, merge_group, sensor_matrix


def causal_windows(frame, center, scale, *, window=32, max_gap_seconds=2):
    """Return N x sensors x window values and availability for one recording.

    Timestamps only delimit contiguous segments. The endpoint is included and
    no later row enters its window. Unavailable rows contain zeros.
    """
    for name in ("experiment_id", "_source"):
        if name in frame and frame[name].nunique(dropna=False) != 1:
            raise ValueError("Windows accept one source recording at a time.")
    values = sensor_matrix(frame)
    center, scale = np.asarray(center), np.asarray(scale)
    if (
        center.shape != (len(FEATURES),)
        or scale.shape != center.shape
        or not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise ValueError("A finite center and positive scale are required per sensor.")
    if not isinstance(window, int) or window < 1 or max_gap_seconds <= 0:
        raise ValueError("Invalid window or gap limit.")
    times = pd.DatetimeIndex(frame.datetime)
    if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Windows require unique increasing timestamps.")
    normalized = ((values - center) / scale).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("Nonfinite normalized sensor values.")
    gaps = np.r_[True, np.diff(times.as_unit("ns").asi8) / 1e9 > max_gap_seconds]
    starts = np.flatnonzero(gaps)
    x = np.zeros((len(frame), len(FEATURES), window), dtype=np.float32)
    ready = np.zeros(len(frame), dtype=bool)
    for start, stop in zip(starts, np.r_[starts[1:], len(frame)], strict=True):
        if stop - start >= window:
            first = start + window - 1
            x[first:stop] = np.lib.stride_tricks.sliding_window_view(
                normalized[start:stop], window, axis=0
            )
            ready[first:stop] = True
    return x, ready


@dataclass
class WindowData:
    x: np.ndarray
    y: np.ndarray
    rows: pd.DataFrame
    audit: dict


def prepare_windows(streams, center, scale, *, window=32, max_gap_seconds=2):
    """Deduplicate endpoints per leakage group after building original context.

    First source order owns shared observations, including unavailable context.
    Conflicting labels are excluded, matching the existing merge_group policy.
    """
    chunks, by_group = [], {}
    offset = 0
    for source, frame in streams.items():
        if frame._group.nunique() != 1:
            raise ValueError("A source recording must belong to one leakage group.")
        if not frame.anomaly.isin([0, 1]).all():
            raise ValueError("Source anomaly targets must be binary.")
        x, ready = causal_windows(
            frame, center, scale, window=window, max_gap_seconds=max_gap_seconds
        )
        rows = frame.copy()
        rows["_source"] = source
        rows["_window_index"] = np.arange(len(frame)) + offset
        rows["_ready"] = ready
        offset += len(frame)
        chunks.append(x)
        by_group.setdefault(frame._group.iloc[0], []).append(rows)
    if not chunks:
        raise ValueError("No source recordings provided.")
    retained, audit = [], {}
    for group, frames in by_group.items():
        rows, audit[group] = merge_group(frames)
        retained.append(rows.loc[rows._ready & rows.anomaly.ge(0)])
    rows = pd.concat(retained, ignore_index=True)
    all_x = np.concatenate(chunks)
    return WindowData(
        all_x[rows._window_index.to_numpy(dtype=int)],
        rows.anomaly.to_numpy(dtype=np.float32),
        rows,
        audit,
    )
