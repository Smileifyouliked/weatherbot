Project: Post-processing model that predicts daily max temperature at one ASOS weather station, outputting a calibrated probability distribution, not a point estimate.

Hard rules:

1. Never use data timestamped after the forecast issue time. Every feature carries explicit valid_time and issue_time. Any feature where issue_time > forecast_time is a bug.
2. No random train/test splits ever. Walk-forward only.
3. No accuracy metrics. Use CRPS, Brier at thresholds, and PIT histograms.
4. Every model is compared against three baselines: raw ensemble mean, climatology, and persistence. Report all four together or the result is meaningless.
5. Cache every API pull to disk as parquet. Never re-download in a loop.
6. Store all times as UTC internally. Convert to station-local only at the resolution boundary.
7. If a step fails, stop and report. Never fall back to synthetic or filled data silently.
