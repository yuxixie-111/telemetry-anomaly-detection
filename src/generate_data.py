from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd

from src.utils import AnomalyEvent, seed_everything, events_to_point_labels



def generate_telemetry_series(
    n: int = 5000,
    seed: int = 42,
    trend_slope: float = 0.0005,
    seasonality_amp: float = 0.8,
    seasonality_period: int = 200,
    noise_sigma: float = 0.8,
    n_spikes: int = 8,
    spike_mag: Tuple[float, float] = (6.0, 10.0),
    n_steps: int = 2,
    step_mag: Tuple[float, float] = (2.0, 5.0),
    n_drifts: int = 2,
    drift_total: Tuple[float, float] = (3.0, 6.0),
    drift_len: Tuple[int, int] = (250, 600),
) -> Tuple[pd.DataFrame, List[AnomalyEvent]]:
    """
    Creates a synthetic telemetry time series with labeled anomaly events.
    Signal = baseline + trend + seasonality + noise + injected anomalies.

    Returns:
      df with columns: t, x, y_true
      events list (kind,start,end)
    """
    rng = seed_everything(seed)
    t = np.arange(n)

    baseline = 0.0
    trend = trend_slope * t
    season = seasonality_amp * np.sin(2 * np.pi * t / seasonality_period)
    noise = rng.normal(0.0, noise_sigma, size=n)

    x = baseline + trend + season + noise
    events: List[AnomalyEvent] = []

    # Inject spikes (single-point)
    for _ in range(n_spikes):
        idx = int(rng.integers(50, n - 50))
        mag = float(rng.uniform(spike_mag[0], spike_mag[1]))
        sign = float(rng.choice([-1.0, 1.0]))
        x[idx] += sign * mag
        events.append(AnomalyEvent(kind="spike", start=idx, end=idx))

    # Inject step changes (persistent from start to end)
    for _ in range(n_steps):
        s = int(rng.integers(200, n - 800))
        e = int(rng.integers(s + 200, min(n - 1, s + 1200)))
        mag = float(rng.uniform(step_mag[0], step_mag[1]))
        sign = float(rng.choice([-1.0, 1.0]))
        x[s:e + 1] += sign * mag
        events.append(AnomalyEvent(kind="step", start=s, end=e))

    # Inject drifts (ramp up/down)
    for _ in range(n_drifts):
        length = int(rng.integers(drift_len[0], drift_len[1]))
        s = int(rng.integers(200, n - length - 200))
        e = s + length
        total = float(rng.uniform(drift_total[0], drift_total[1]))
        sign = float(rng.choice([-1.0, 1.0]))
        ramp = np.linspace(0.0, sign * total, num=length + 1)
        x[s:e + 1] += ramp
        events.append(AnomalyEvent(kind="drift", start=s, end=e))

    # Build labels
    y_true = events_to_point_labels(n, events)

    df = pd.DataFrame({"t": t, "x": x, "y_true": y_true})
    # Keep events sorted (nice for reading)
    events = sorted(events, key=lambda ev: ev.start)
    return df, events
