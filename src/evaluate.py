from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from .utils import AnomalyEvent, compute_prf, detection_delay


def evaluate_detectors(
    y_true: np.ndarray,
    preds: Dict[str, np.ndarray],
    events_true: List[AnomalyEvent],
) -> pd.DataFrame:
    rows = []
    for name, y_pred in preds.items():
        prf = compute_prf(y_true, y_pred)
        delay = detection_delay(events_true, y_pred)
        rows.append({
            "method": name,
            "precision": prf["precision"],
            "recall": prf["recall"],
            "f1": prf["f1"],
            "fp": prf["fp"],
            "fn": prf["fn"],
            "detection_rate": delay["detection_rate"],
            "avg_delay": delay["avg_delay"] if delay["avg_delay"] is not None else np.nan,
        })
    df = pd.DataFrame(rows).sort_values(by="f1", ascending=False).reset_index(drop=True)
    return df
