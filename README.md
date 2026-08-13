# weatherbot

Post-processing model for daily maximum temperature at **KNYC** (New York
Central Park), producing a calibrated probability distribution rather than a
point estimate.

This repository currently covers data ingestion and verification only. No model
is fitted yet.

## Layout

```
config.py            station, sources, ranges, cache paths
wxio.py              HTTP with retry, parquet cache, daily derivation
ingest_asos.py       observations: hourly METAR + reported daily max
ingest_forecast.py   forecasts: ECMWF deterministic backbone + ensemble
verify_data.py       checks both caches against the project's hard rules
DATA_SOURCES.md      what each endpoint can and cannot prove about provenance
```

## Running

```bash
pip install pandas pyarrow requests

python3 ingest_asos.py        # observations, 2023-01-01 onward
python3 ingest_forecast.py    # ECMWF runs 2024-03-14 onward, + live ensemble
python3 verify_data.py        # must exit 0 before anything downstream
```

Both ingestion modules are resumable: runs already in the cache are skipped, so
re-running never re-downloads (hard rule 5). `ingest_forecast.py` takes
`--start`, `--end`, `--limit`, `--skip-backbone` and `--skip-ensemble`.

## The two caches

`data/raw_hourly.parquet` — every hourly value ever pulled, long format, one row
per (source, model, member, issue_time, valid_time). All timestamps UTC.

`data/daily.parquet` — station-local daily maxima: the reported observed max
(the target), the hourly-derived max (QC), and forecast maxima per run and
member with `lead_days`.

Both are gitignored; they are rebuildable local state.

## Provenance, in short

`issue_time` is trustworthy only where `issue_time_confirmed` is true. That
holds for the ECMWF deterministic backbone, where the run is named in the
request and validated server-side. It does **not** hold for the ensemble, whose
endpoint exposes no run identifier — those rows carry a null `issue_time` and
are reported by `verify_data.py` as unusable under hard rule 1.

Read `DATA_SOURCES.md` before adding a source. The short version: the ensemble
archive is about four days deep despite advertising a 92-day window, and the
Historical Forecast API reaches 2021 but cannot tell you which run produced a
value.
