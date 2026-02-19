from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

from statsmodels.tsa.seasonal import STL


def rolling_zscore_detector(x: np.ndarray, window: int = 80, z_thresh: float = 3.5) -> np.ndarray:
    """
    Rolling z-score detector: flags points whose |z| > threshold.
    Uses rolling mean/std on a trailing window (causal).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    y = np.zeros(n, dtype=int)
    if window < 5:
        raise ValueError("window too small")

    for i in range(window, n):
        hist = x[i - window:i]
        mu = float(np.mean(hist))
        sd = float(np.std(hist) + 1e-8)
        z = (x[i] - mu) / sd
        if abs(z) > z_thresh:
            y[i] = 1
    return y


def ewma_detector(x: np.ndarray, alpha: float = 0.05, k: float = 3.0, warmup: int = 200) -> np.ndarray:
    """
    EWMA control chart style detector.
    Maintains EWMA mean; flags when deviation exceeds k * sigma_est.
    sigma_est from rolling MAD of residuals (robust).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    y = np.zeros(n, dtype=int)

    ewma = x[0]
    resids = []

    for i in range(1, n):
        ewma = alpha * x[i] + (1 - alpha) * ewma
        resid = x[i] - ewma
        resids.append(resid)

        if i < warmup:
            continue

        # robust scale estimate on recent residuals
        recent = np.array(resids[-warmup:], dtype=float)
        scale = float(median_abs_deviation(recent, scale="normal") + 1e-8)
        if abs(resid) > k * scale:
            y[i] = 1

    return y


def cusum_detector(x: np.ndarray, drift: float = 0.01, h: float = 6.0, warmup: int = 200) -> np.ndarray:
    """
    Two-sided CUSUM on standardized residuals relative to rolling robust center/scale.
    Good for step/drift detection.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    y = np.zeros(n, dtype=int)

    g_pos = 0.0
    g_neg = 0.0

    for i in range(n):
        if i < warmup:
            continue
        hist = x[i - warmup:i]
        center = float(np.median(hist))
        scale = float(median_abs_deviation(hist, scale="normal") + 1e-8)
        z = (x[i] - center) / scale

        g_pos = max(0.0, g_pos + z - drift)
        g_neg = min(0.0, g_neg + z + drift)

        if g_pos > h or abs(g_neg) > h:
            y[i] = 1
            # reset after signal (common in charts)
            g_pos, g_neg = 0.0, 0.0

    return y


def stl_residual_detector(x: np.ndarray, period: int = 200, mad_thresh: float = 4.0) -> np.ndarray:
    """
    STL decomposition -> residual -> robust threshold via MAD.
    Helpful when signal has seasonality/trend.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    y = np.zeros(n, dtype=int)

    # STL expects a pandas series
    s = pd.Series(x)
    stl = STL(s, period=period, robust=True)
    res = stl.fit().resid.values

    med = float(np.median(res))
    scale = float(median_abs_deviation(res, scale="normal") + 1e-8)
    z = (res - med) / scale
    y[np.abs(z) > mad_thresh] = 1
    return y


def run_all_detectors(x: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "rolling_z": rolling_zscore_detector(x, window=80, z_thresh=3.5),
        "ewma": ewma_detector(x, alpha=0.05, k=3.0, warmup=200),
        "cusum": cusum_detector(x, drift=0.05, h=6.0, warmup=200),
        "stl_resid": stl_residual_detector(x, period=200, mad_thresh=4.0),
    }
