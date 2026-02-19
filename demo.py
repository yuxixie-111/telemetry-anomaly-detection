from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

from src.generate_data import generate_telemetry_series
from src.detectors import run_all_detectors
from src.evaluate import evaluate_detectors


def plot_series_with_labels(t, x, y_true, title: str):
    plt.figure(figsize=(12, 4))
    plt.plot(t, x, linewidth=1)
    idx = np.where(y_true == 1)[0]
    if len(idx) > 0:
        plt.scatter(t[idx], x[idx], s=8)
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("telemetry")
    plt.tight_layout()


def plot_detections_overlay(t, x, y_true, preds, title: str):
    plt.figure(figsize=(12, 4))
    plt.plot(t, x, linewidth=1, label="signal")

    # ground truth (light markers)
    gt = np.where(y_true == 1)[0]
    if len(gt) > 0:
        plt.scatter(t[gt], x[gt], s=10, label="ground truth")

    # overlay detections (mark at top)
    ymax = float(np.max(x))
    y0 = ymax + 0.5
    offset = 0.25
    for i, (name, y_pred) in enumerate(preds.items()):
        hit = np.where(y_pred == 1)[0]
        if len(hit) > 0:
            plt.scatter(t[hit], np.full_like(hit, y0 + i * offset, dtype=float), s=10, label=name)

    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("telemetry / detection markers")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()


def main():
    df, events = generate_telemetry_series(n=5000, seed=7)
    t = df["t"].values
    x = df["x"].values
    y_true = df["y_true"].values

    preds = run_all_detectors(x)
    report = evaluate_detectors(y_true, preds, events)
    print("\n=== Detector comparison ===")
    print(report.to_string(index=False))

    # Make figures folder
    out_dir = os.path.join("outputs", "figures")
    os.makedirs(out_dir, exist_ok=True)

    plot_series_with_labels(t, x, y_true, "Synthetic Telemetry with Ground-Truth Anomalies")
    plt.savefig(os.path.join(out_dir, "example_series.png"), dpi=180)

    plot_detections_overlay(t, x, y_true, preds, "Detections Overlay (Markers Above Signal)")
    plt.savefig(os.path.join(out_dir, "detections_overlay.png"), dpi=180)

    print(f"\nSaved figures to {out_dir}/")


if __name__ == "__main__":
    main()
