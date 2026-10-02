"""Synthetic multi-sensor pump data with gradual degradation before each failure."""
from __future__ import annotations

import numpy as np
import pandas as pd

SENSORS = ["vibration_rms", "bearing_temp", "discharge_pressure", "flow_rate", "motor_current"]


def pump_run(n_hours: int = 3000, failures=(1400, 2600), degradation_hours: int = 150, seed: int = 0) -> pd.DataFrame:
    """Hourly readings. Normal behaviour has daily load cycles and correlated sensors;
    before each failure a fault develops (rising vibration and temperature, falling flow),
    then the pump is repaired and returns to normal."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours)
    load = 0.7 + 0.2 * np.sin(2 * np.pi * t / 24) + 0.05 * rng.standard_normal(n_hours)
    flow = 100 * load + rng.normal(0, 2, n_hours)
    pressure = 5 + 2 * load + rng.normal(0, 0.1, n_hours)
    current = 20 + 15 * load + rng.normal(0, 0.5, n_hours)
    vib = 1.0 + 0.3 * load + rng.normal(0, 0.05, n_hours)
    temp = 45 + 10 * load + rng.normal(0, 0.7, n_hours)
    health = np.zeros(n_hours)  # 0 = healthy, rising to 1 at failure
    for f in failures:
        start = f - degradation_hours
        ramp = np.clip((t - start) / degradation_hours, 0, 1) ** 2
        ramp[t > f] = 0
        health = np.maximum(health, ramp)
    vib += 1.2 * health + 0.3 * health * rng.standard_normal(n_hours)
    temp += 12 * health
    flow -= 15 * health
    current += 4 * health
    df = pd.DataFrame(dict(zip(SENSORS, [vib, temp, pressure, flow, current])),
                      index=pd.date_range("2025-01-01", periods=n_hours, freq="h"))
    df["health"] = health
    df["failure"] = 0
    df.loc[df.index[list(failures)], "failure"] = 1
    return df
