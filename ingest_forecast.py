"""Ingest model forecasts for the target station from Open-Meteo.

Two streams, with deliberately different provenance guarantees:

  backbone  ECMWF IFS 9 km via the Single Runs API. The run is named in the
            request (run=YYYY-MM-DDTHH:MM) and validated server-side -- an
            unavailable run returns HTTP 400 rather than substituting
            best-available data. issue_time is therefore trustworthy, and is
            recorded from the request because the response body does not
            contain it.

  ensemble  ECMWF IFS 0.25 deg, 51 members. The ensemble endpoint exposes no
            run identifier at all, and past days are spliced across runs, so
            issue_time is left null and issue_time_confirmed is False. These
            rows are quarantined by verify_data.py and must not be used as
            rule-1 features until provenance can be established.

Usage:
    python3 ingest_forecast.py [--start ...] [--end ...] [--skip-backbone]
                              [--skip-ensemble] [--limit N]
"""

from __future__ import annotations

import argparse
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd

import config
import wxio


def _hourly_frame(payload: dict, key: str) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise wxio.SourceError(f"response had no hourly block: {list(payload)[:8]}")
    if key not in hourly:
        raise wxio.SourceError(f"response missing {key!r}; got {list(hourly)[:8]}")
    return pd.DataFrame(hourly)


# --- Deterministic backbone --------------------------------------------------

def run_times(start: date, end: date) -> list[datetime]:
    out = []
    day = start
    while day <= end:
        for hh in config.BACKBONE_RUN_HOURS:
            out.append(datetime(day.year, day.month, day.day, hh, tzinfo=timezone.utc))
        day += timedelta(days=1)
    return out


def fetch_run(run: datetime) -> pd.DataFrame | None:
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "hourly": config.VARIABLE,
        "models": config.BACKBONE_MODEL,
        "run": run.strftime("%Y-%m-%dT%H:%M"),
    }
    payload = wxio.get_json(config.SINGLE_RUNS_URL, params)
    if payload is None:
        return None  # archive gap

    h = _hourly_frame(payload, config.VARIABLE)
    valid = pd.to_datetime(h["time"], utc=True)
    value = pd.to_numeric(h[config.VARIABLE], errors="coerce")
    keep = value.notna()
    if not keep.any():
        raise wxio.SourceError(f"run {run:%Y-%m-%dT%H:%M} returned all-null values")

    return pd.DataFrame(
        {
            "source": "single_runs",
            "model": config.BACKBONE_MODEL,
            "member": pd.NA,
            "issue_time": run,          # from the request; validated server-side
            "issue_time_confirmed": True,
            "valid_time": valid[keep],
            "variable": config.VARIABLE,
            "value_c": value[keep],
            "fetched_at": wxio.utcnow(),
        }
    )


def ingest_backbone(start: date, end: date, limit: int | None) -> None:
    wanted = run_times(start, end)
    have = wxio.cached_issue_times("single_runs", config.BACKBONE_MODEL)
    todo = [r for r in wanted if pd.Timestamp(r) not in have]
    cached = len(wanted) - len(todo)
    if limit:
        todo = todo[:limit]

    print(f"backbone {config.BACKBONE_MODEL}: {len(wanted)} runs in range, "
          f"{cached} already cached, {len(todo)} to fetch"
          + (f" (limited to {limit})" if limit else ""))

    gaps, unreachable, batch, added = [], [], [], 0
    t0 = time.time()
    for i, run in enumerate(todo, 1):
        t = time.time()
        try:
            frame = fetch_run(run)
        except wxio.SourceError as exc:
            # The server hung or errored repeatedly on this specific run. That
            # is a hole in the archive, not a reason to fabricate a value; it is
            # recorded and reported, and a systemic rate is caught below.
            unreachable.append((run, str(exc)[:80]))
            frame = None
            print(f"  unreachable: {run:%Y-%m-%d} ({str(exc)[:60]})", flush=True)
        else:
            if frame is None:
                gaps.append(run)
        dt = time.time() - t
        if frame is not None:
            batch.append(frame)
        if dt > 10:
            print(f"  slow: {run:%Y-%m-%d} took {dt:.0f}s", flush=True)
        time.sleep(config.REQUEST_SPACING)

        if i % 25 == 0 or i == len(todo):
            rate = i / max(time.time() - t0, 1e-9)
            eta = (len(todo) - i) / rate / 60
            print(f"  {i}/{len(todo)} runs | {rate:.1f}/s | ETA {eta:.0f}m | "
                  f"{len(gaps)} gaps | {len(unreachable)} unreachable", flush=True)

        # Flush periodically so a long backfill is resumable (hard rule 5).
        if len(batch) >= 50 or i == len(todo):
            if batch:
                added += wxio.merge_raw(pd.concat(batch, ignore_index=True))
                batch = []
                print(f"    flushed -> +{added} rows total", flush=True)

    if gaps:
        print(f"  archive gaps -- server reported {len(gaps)} runs not available:")
        for g in gaps[:10]:
            print(f"    {g:%Y-%m-%dT%H:%M}Z")
        if len(gaps) > 10:
            print(f"    ... and {len(gaps) - 10} more")

    if unreachable:
        print(f"  unreachable -- {len(unreachable)} runs failed every attempt:")
        for run, why in unreachable[:10]:
            print(f"    {run:%Y-%m-%dT%H:%M}Z  {why}")
        if len(unreachable) > 10:
            print(f"    ... and {len(unreachable) - 10} more")
        frac = len(unreachable) / max(len(todo), 1)
        if frac > config.MAX_UNREACHABLE_FRACTION:
            raise wxio.SourceError(
                f"{frac:.1%} of runs unreachable (limit "
                f"{config.MAX_UNREACHABLE_FRACTION:.0%}); treating as a source outage "
                f"rather than an archive gap -- stopping"
            )


# --- Ensemble ----------------------------------------------------------------

def ingest_ensemble() -> None:
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "hourly": config.VARIABLE,
        "models": config.ENSEMBLE_MODEL,
        "past_days": config.ENSEMBLE_PAST_DAYS,
        "forecast_days": config.ENSEMBLE_FORECAST_DAYS,
    }
    print(f"ensemble {config.ENSEMBLE_MODEL}: 1 request "
          f"(past {config.ENSEMBLE_PAST_DAYS}d + {config.ENSEMBLE_FORECAST_DAYS}d forecast)")
    payload = wxio.get_json(config.ENSEMBLE_URL, params)
    if payload is None:
        raise wxio.SourceError("ensemble endpoint reported no available run")

    h = _hourly_frame(payload, config.VARIABLE)
    valid = pd.to_datetime(h["time"], utc=True)
    fetched = wxio.utcnow()

    cols = [c for c in h.columns if c.startswith(config.VARIABLE)]
    frames = []
    for col in cols:
        suffix = col[len(config.VARIABLE):]
        member = 0 if suffix == "" else int(suffix.replace("_member", ""))
        value = pd.to_numeric(h[col], errors="coerce")
        keep = value.notna()
        if not keep.any():
            continue
        frames.append(
            pd.DataFrame(
                {
                    "source": "ensemble",
                    "model": config.ENSEMBLE_MODEL,
                    "member": member,
                    "issue_time": pd.NaT,        # endpoint exposes no run id
                    "issue_time_confirmed": False,
                    "valid_time": valid[keep],
                    "variable": config.VARIABLE,
                    "value_c": value[keep],
                    "fetched_at": fetched,
                }
            )
        )

    if not frames:
        raise wxio.SourceError(
            f"ensemble returned {len(cols)} member columns but every one was null"
        )

    hours = int(frames[0].shape[0])
    added = wxio.merge_raw(pd.concat(frames, ignore_index=True))
    print(f"  {len(frames)} members with data, {hours} hours each | +{added} rows")
    print("  issue_time left NULL: this endpoint does not identify the run")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=date.fromisoformat, default=config.BACKBONE_START)
    ap.add_argument("--end", type=date.fromisoformat,
                    default=date.today() - timedelta(days=1))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-backbone", action="store_true")
    ap.add_argument("--skip-ensemble", action="store_true")
    args = ap.parse_args()

    if not args.skip_backbone:
        ingest_backbone(args.start, args.end, args.limit)
    if not args.skip_ensemble:
        ingest_ensemble()

    daily = wxio.build_daily()
    print(f"daily.parquet rebuilt: {len(daily)} rows "
          f"({daily.attrs['dropped_short_days']} short days dropped)")


if __name__ == "__main__":
    main()
