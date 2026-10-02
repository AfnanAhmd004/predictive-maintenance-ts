"""Maintenance-oriented metrics: warning lead time and false alarms."""
from __future__ import annotations

import numpy as np


def debounce(alarms: np.ndarray, k: int = 3) -> np.ndarray:
    """Raise an alarm only after k consecutive anomalous hours (reduces nuisance alarms)."""
    run, out = 0, np.zeros_like(alarms, dtype=bool)
    for i, a in enumerate(alarms):
        run = run + 1 if a else 0
        out[i] = run >= k
    return out


def lead_times(alarms: np.ndarray, failure_idx, horizon: int = 200) -> list[float]:
    """Hours between the first alarm in the `horizon` hours before each failure and the failure (NaN if missed)."""
    out = []
    for f in failure_idx:
        window = alarms[max(0, f - horizon) : f]
        hits = np.flatnonzero(window)
        out.append(float(len(window) - hits[0]) if len(hits) else np.nan)
    return out


def false_alarms_per_week(alarms: np.ndarray, failure_idx, horizon: int = 200) -> float:
    mask = np.ones(len(alarms), dtype=bool)
    for f in failure_idx:
        mask[max(0, f - horizon) : f + 24] = False  # exclude pre-failure and repair periods
    rising = np.flatnonzero(alarms[mask][1:] & ~alarms[mask][:-1])
    return len(rising) / (mask.sum() / (24 * 7))
