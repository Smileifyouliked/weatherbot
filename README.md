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
pip install pandas pyarrow requests matplotlib

./make_cache.sh          # rebuild or resume every cache, then verify
./make_cache.sh --fresh  # delete data/ first, then rebuild from nothing
```

That runs the ingestion modules in dependency order and finishes with
`verify_data.py`, which must exit 0 before anything downstream is trustworthy.
Any failing step stops the script, so a partial cache is never left looking
complete (hard rule 7).

To run a single stage instead:

```bash
python3 ingest_asos.py        # observations, 2023-01-01 onward
python3 ingest_forecast.py    # ECMWF runs 2024-03-14 onward, + live ensemble
python3 ingest_nbm.py         # NBM day-ahead max + spread
python3 verify_data.py
```

All ingestion modules are resumable: anything already cached is skipped, so
re-running never re-downloads (hard rule 5). `ingest_forecast.py` takes
`--start`, `--end`, `--limit`, `--workers`, `--timeout`, `--skip-backbone` and
`--skip-ensemble`; `ingest_nbm.py` takes `--start`, `--end`, `--workers` and
`--refresh`.

### Runtime

| Stage | Cold (`--fresh`), estimated | Warm re-run, measured |
|---|---|---|
| Observations | ~25 s | 9 s |
| ECMWF backbone, main pass | ~30 min | 1 m 25 s |
| ECMWF backbone, straggler retry | ~13 min | 1 m 36 s |
| ECMWF ensemble | ~5 s | 4 s |
| NBM bulletins | ~2 min | 3 s |
| Verify | ~5 s | 3 s |
| **Total** | **~45–55 min** | **3 m 20 s** |

Warm figures are from an actual run on a populated cache. Cold figures are
extrapolated from observed per-request rates rather than timed end to end.

The backbone dominates, and not because of volume: the Single Runs archive
serves roughly 0.4–0.5 runs/second, and a minority of runs hang server-side
rather than returning an error. The straggler pass exists because those hangs
are sporadic — a slower, more patient second pass recovers nearly all of them.

A warm run is not free either. The backbone still retries the six runs that are
permanently absent from the archive (2025-08-05 … 2025-08-09 and 2026-06-23),
which is essentially all of the three minutes above. They are re-attempted
rather than blacklisted because an archive gap can be backfilled upstream later.

Observations are pulled in a single request covering the whole range, and NBM
fetches ~21 MB per cycle at ~20 MB/s, so neither is a bottleneck.

## The two caches

`data/raw_hourly.parquet` — every hourly value ever pulled, long format, one row
per (source, model, member, issue_time, valid_time). All timestamps UTC.

`data/daily.parquet` — station-local daily maxima: the reported observed max
(the target), the hourly-derived max (QC), NBM's day-ahead max with its
published spread, and forecast maxima per run and member with `lead_days`.

Both are committed (about 1.6 MB together) so the published numbers can be
reproduced without rebuilding, and `./make_cache.sh` regenerates them from the
sources at any time.

### Cache snapshot policy

The committed parquet files are a **point-in-time snapshot, refreshed when
results are published — not on every pull.**

The reason is churn, not size. `raw_hourly.parquet` is rewritten whole on every
ingestion, so each pull produces an entirely new ~1.5 MB blob rather than a
delta. Committing on every run would add roughly half a gigabyte of history per
year for data that `./make_cache.sh` can rebuild from scratch. So:

- **Do** refresh the snapshot when the numbers in a report or PR change, in the
  same commit as those numbers, so the data and the results stay in step.
- **Don't** commit it after a routine daily pull. Leave the working tree dirty,
  or `git checkout -- data/` to discard.

Git LFS is deliberately not used. It exists to keep large blobs out of pack
files, and at 1.6 MB it would add a clone-time dependency — a missing LFS client
turns the parquet files into unusable pointer stubs — for no benefit. Revisit
only if the snapshot passes ~50 MB.

Rows are keyed so that re-running is a no-op rather than an append. Forecast
rows whose run cannot be identified (the ensemble) are additionally keyed on
their pull date, so successive pulls accumulate as distinct observations of the
forecast; observations are keyed on valid time alone, since an observation is
the same fact whenever it was fetched.

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
