from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np


@dataclass
class AnomalyEvent:
    kind: str
    start: int
    end: int  # inclusive


def seed_everything(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def events_to_point_labels(n: int, events: List[AnomalyEvent]) -> np.ndarray:
    y = np.zeros(n, dtype=int)
    for ev in events:
        s = max(0, ev.start)
        e = min(n - 1, ev.end)
        if s <= e:
            y[s:e + 1] = 1
    return y


def point_labels_to_events(y: np.ndarray) -> List[Tuple[int, int]]:
    """
    Convert 0/1 point labels to contiguous intervals [start,end] inclusive.
    """
    y = np.asarray(y).astype(int)
    n = len(y)
    events: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        if y[i] == 1:
            s = i
            while i < n and y[i] == 1:
                i += 1
            e = i - 1
            events.append((s, e))
        else:
            i += 1
    return events


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den != 0 else 0.0


def compute_prf(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def detection_delay(events_true: List[AnomalyEvent], y_pred: np.ndarray) -> Dict[str, Any]:
    """
    For each ground-truth event, find first detection index within [start,end].
    Delay = first_detect - start. If never detected, delay = None.
    Return avg delay over detected events + detection rate.
    """
    y_pred = np.asarray(y_pred).astype(int)
    delays = []
    detected = 0
    for ev in events_true:
        s, e = ev.start, ev.end
        s = max(0, s)
        e = min(len(y_pred) - 1, e)
        hit_idxs = np.where(y_pred[s:e + 1] == 1)[0]
        if len(hit_idxs) > 0:
            first = s + int(hit_idxs[0])
            delays.append(first - s)
            detected += 1
        else:
            delays.append(None)

    detected_delays = [d for d in delays if d is not None]
    avg_delay = float(np.mean(detected_delays)) if detected_delays else None
    detection_rate = safe_div(detected, len(events_true)) if events_true else 0.0
    return {"avg_delay": avg_delay, "detection_rate": detection_rate, "per_event_delay": delays}
