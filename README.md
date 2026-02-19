# Telemetry Time-Series Anomaly Detection (Mini Project)

This repository demonstrates an engineering-style approach to anomaly detection for telemetry time series, motivated by autonomy validation workflows.  
Goal: detect anomalies (spikes, step changes, and drifts) and quantify detection quality and delay.

## What this project includes
- **Synthetic telemetry generator** with ground-truth anomaly intervals:
  - Spike (sensor glitch)
  - Step change (mode shift / fault)
  - Drift (model mismatch / calibration drift)
- Multiple **interpretable detectors**:
  - Rolling Z-score
  - EWMA control chart
  - CUSUM change detection
  - (Optional) STL residual detector
- **Evaluation**:
  - Precision / Recall / F1 (point-wise)
  - Detection rate (event-wise)
  - Average detection delay (time to first detection within each event)

## Quickstart
```bash
pip install -r requirements.txt
python demo.py
