# Data sources and their provenance limits

Probed 2026-08-13 against Open-Meteo and the Iowa Environmental Mesonet, for
KNYC (New York Central Park, 40.7790 / -73.9693).

## Summary

| Source | `issue_time` provenance | Archive depth (probed) | Range per request |
|---|---|---|---|
| Single Runs — `ecmwf_ifs` (9 km) | **confirmed, server-validated** | 2024-03-14 onward | one run |
| Single Runs — `gfs_seamless` | confirmed, server-validated | 2026-04-02 onward | one run |
| Ensemble — `ecmwf_ifs025` (51 members) | **none** | ~4 days | window, mostly null |
| Previous Runs | implicit — lead offset only | GFS t2m ~2021-03 | date range |
| Historical Forecast | **none** — best-available splice | 2021-01-01 onward | date range |
| IEM ASOS observations | n/a (truth) | full record | date range |

## Why the backbone is ECMWF Single Runs

The Single Runs API takes an explicit `run=` initialisation time and validates it
server-side. A run that does not exist returns HTTP 400 naming the resolved
model, rather than quietly substituting best-available data:

```
run=2026-08-10T00:00 -> HTTP 200
run=2026-08-10T03:00 -> HTTP 400 "The requested model run is not available.
                                  Model: ncep_gfs025, run: 2026-08-10T03:00Z"
```

That failure mode is what makes hard rule 1 enforceable: if a value comes back,
it provably came from the run we named.

**Caveat, and the reason `issue_time` is written from the request side:** the
success response body contains no run identifier at all. Its complete shape is

```
latitude, longitude, generationtime_ms, utc_offset_seconds,
timezone, timezone_abbreviation, elevation,
hourly_units: {time, temperature_2m}
hourly: {time[], temperature_2m[]}
```

So a cached response, divorced from the request that produced it, has
unrecoverable provenance. `ingest_forecast.py` records `issue_time` from the
request parameter and sets `issue_time_confirmed = True` only there.

`ecmwf_ifs` archives 00Z and 12Z runs (06Z and 18Z return "not available"), 168
hours per run. We pull 00Z only, giving exactly one issue_time per calendar day
and leads of +1 to +7 days.

## Why the ensemble is live-only and quarantined

The ensemble endpoint accepts `start_date`/`end_date`, but the accepted range and
the populated range are not the same thing:

```
start_date=2026-05-12&end_date=2026-08-13  -> HTTP 200, 31 gfs025 columns
    2256 hours requested
      96 hours with data (2026-08-10 -> 2026-08-13)
    2160 hours null
temperature_2m_spread over the same window -> 0 non-null hours
```

A request outside that window returns `HTTP 400: start_date is out of allowed
range from 2026-05-12 to 2026-09-17`. So the **real ensemble archive is ~4 days**,
and a naive range pull writes a parquet that is 96% null at HTTP 200 — precisely
the silent-fill failure hard rule 7 exists to catch.

The endpoint also exposes no run identifier, and its past days are spliced across
runs. Ensemble rows therefore carry `issue_time = NULL` and
`issue_time_confirmed = False`; `verify_data.py` reports them as unusable
rule-1 features. They accumulate forward from today so that real ensemble spread
becomes available for future evaluation.

## Why Historical Forecast is not used

It reaches 2021 but is a best-available splice with no recoverable issue_time.
Same station, same valid times, 2023-01-01:

```
valid_time(UTC)   hist-fcst   prev_d1  prev_d2  prev_d3  prev_d4  prev_d5
2023-01-01T00:00      1.9      3.1      2.0      1.2      2.9      3.0
2023-01-01T02:00      1.6      3.2      1.4      1.1      2.6      2.9
2023-01-01T05:00      2.6      3.0      2.1      1.1      2.7      2.6
```

It tracks no single lead consistently and the spread across leads is ~2 °C. Its
effective issue_time can be *after* the forecast_time we would assign, which
violates hard rule 1 in a way no downstream check can detect.

## The target variable is the reported daily max, not max-of-hourly

`ingest_asos.py` pulls two different things from IEM:

- `cgi-bin/request/asos.py` — hourly METAR temperatures, stored in
  `raw_hourly.parquet` as `source='asos_obs'`. Features and QC.
- `api/1/daily.json` — the station's **reported** daily maximum, stored in
  `daily.parquet` as `source='asos_daily_official'`. This is the target.

They are not the same quantity. Over 1311 days at KNYC:

```
max-of-hourly minus reported max:  mean -0.04 C, MAE 0.12 C
  16.1% of days differ by >0.5 C
   3.6% of days differ by >1.0 C
  worst: -1.11 C / +3.33 C
```

The mean is near zero, so the difference is easy to miss in aggregate, but the
per-day error is large next to the skill differences CRPS is meant to resolve.
Two causes: hourly sampling misses the true peak between reports, and rounding
:51 reports to the hour pushed late-evening observations across the local-day
boundary. The rounding is gone — `valid_time` now keeps the true report time,
and observations join model output at the daily level rather than on an hourly
grid.

Rows where IEM flags `tmpf_est` (network-estimated rather than measured) are
excluded rather than accepted as truth.

## Unreachable runs

Some individual archive runs hang server-side rather than returning an error —
`ecmwf_ifs` at `2024-03-17T00:00` is a reproducible example. These are
distinguished from real archive gaps:

- **archive gap** — server returns HTTP 400 "not available". Recorded, reported.
- **unreachable** — every attempt timed out. Recorded, reported separately.

Neither is filled. If more than `MAX_UNREACHABLE_FRACTION` (5%) of runs are
unreachable the backfill aborts, on the grounds that a systemic outage is not an
archive gap and should not be quietly absorbed into the dataset.

## Consequence for hard rule 4

The raw-ensemble-mean baseline cannot be constructed historically — the ensemble
archive is 4 days deep. Until live ensemble data accumulates, the comparable
baselines are the raw deterministic forecast, climatology, and persistence, all
of which are derivable from the current caches. This is a real gap against the
four-baseline requirement and should not be papered over.
