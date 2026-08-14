Project: Post-processing model that predicts daily max temperature at one ASOS weather station, outputting a calibrated probability distribution, not a point estimate.

Hard rules:

1. Never use data timestamped after the forecast issue time. Every feature carries explicit valid_time and issue_time. Any feature where issue_time > forecast_time is a bug.
2. No random train/test splits ever. Walk-forward only.
3. No accuracy metrics. Use CRPS, Brier at thresholds, and PIT histograms.
4. Every model is compared against four baselines: raw deterministic ECMWF with climatological residual sigma, climatology, persistence, and NBM. Report all five together or the result is meaningless. NBM is the operational bar this project has to clear, so a gain over the other three means little without it. It comes from the NBM station bulletins on NOAA's S3 archive (noaa-nbm-grib2-pds), which supply both a max temperature (TXN) and its standard deviation (XND), giving a genuine probabilistic baseline. api.weather.gov cannot be used for this: its gridpoints endpoint serves only the current forecast, with no date parameter and no archive. The raw ensemble mean was dropped as a baseline because it cannot be built over the training period: Open-Meteo's ensemble archive is about four days deep, so no historical ensemble spread exists for this station. The deterministic ECMWF forecast supplies the mean and a trailing residual std supplies the spread in its place.
5. Cache every API pull to disk as parquet. Never re-download in a loop.
6. Store all times as UTC internally. Convert to station-local only at the resolution boundary.
7. If a step fails, stop and report. Never fall back to synthetic or filled data silently.
