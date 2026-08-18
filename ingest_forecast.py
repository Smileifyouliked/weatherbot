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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import config
import wxio

# A run still absent from the archive after this long is a hole that will not
# fill. Shorter than this and it is simply not published yet -- today's run is
# absent for the first several hours by definition.
GAP_SETTLED_DAYS = 7


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
    """One request per run, carrying every predictor in FORECAST_VARIABLES."""
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "hourly": ",".join(config.FORECAST_VARIABLES),
        "models": config.BACKBONE_MODEL,
        "run": run.strftime("%Y-%m-%dT%H:%M"),
    }
    payload = wxio.get_json(config.SINGLE_RUNS_URL, params)
    if payload is None:
        return None  # archive gap

    h = _hourly_frame(payload, config.VARIABLE)
    valid = pd.to_datetime(h["time"], utc=True)
    fetched = wxio.utcnow()

    # Some archived runs are partial: 2025-08-07 returns wind and dewpoint but
    # no temperature or cloud. Without the target variable the run is unusable
    # whatever else it holds, and the response is deterministic, so this is an
    # archive gap rather than a transport failure. Classifying it as unreachable
    # made these runs trip the outage guard on every warm-cache pull, where the
    # denominator is only the handful still outstanding.
    if not pd.to_numeric(h.get(config.VARIABLE), errors="coerce").notna().any():
        return None

    frames = []
    for var in config.FORECAST_VARIABLES:
        if var not in h.columns:
            raise wxio.SourceError(f"run {run:%Y-%m-%dT%H:%M} response missing {var!r}")
        value = pd.to_numeric(h[var], errors="coerce")
        keep = value.notna()
        if not keep.any():
            # The target variable is present but this predictor is not, so the
            # model does not carry it at all (e.g. pressure-level fields on
            # ecmwf_ifs). That is a configuration error, not an archive gap.
            raise wxio.SourceError(
                f"run {run:%Y-%m-%dT%H:%M} has {config.VARIABLE} but returned "
                f"all-null values for {var!r}; this model does not carry it")
        frames.append(
            pd.DataFrame(
                {
                    "source": "single_runs",
                    "model": config.BACKBONE_MODEL,
                    "member": pd.NA,
                    "issue_time": run,   # from the request; validated server-side
                    "issue_time_confirmed": True,
                    "valid_time": valid[keep],
                    "variable": var,
                    "value": value[keep],
                    "fetched_at": fetched,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def gap_path() -> Path:
    return config.DATA_DIR / "archive_gaps.parquet"


def load_gaps() -> dict[pd.Timestamp, pd.Timestamp]:
    """Runs the server has said it does not have, and when it last said so."""
    p = gap_path()
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    return dict(zip(pd.to_datetime(df["run"], utc=True),
                    pd.to_datetime(df["last_checked"], utc=True)))


def save_gaps(gaps: dict[pd.Timestamp, pd.Timestamp]) -> None:
    if not gaps:
        return
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"run": list(gaps), "last_checked": list(gaps.values())}) \
        .sort_values("run").to_parquet(gap_path(), index=False)


def settled_gap(run: datetime, now: pd.Timestamp) -> bool:
    """Whether a known-absent run is old enough to stop asking about.

    A run missing from the archive an hour after issue is simply not published
    yet and must stay retryable -- that is the normal state of today's run, and
    blacklisting it would mean never fetching the only run that matters. A run
    still missing after GAP_SETTLED_DAYS is a hole in the archive that will not
    fill, and re-asking three times per cron tick, forever, spends the daily API
    budget on runs that do not exist.
    """
    return (now - pd.to_datetime(run, utc=True)) > pd.Timedelta(days=GAP_SETTLED_DAYS)


def ingest_backbone(start: date, end: date, limit: int | None,
                    retry_gaps: bool = False) -> None:
    wanted = run_times(start, end)
    have = wxio.cached_issue_times("single_runs", config.BACKBONE_MODEL,
                                   config.FORECAST_VARIABLES)
    todo = [r for r in wanted if pd.Timestamp(r) not in have]
    cached = len(wanted) - len(todo)

    known_gaps = {} if retry_gaps else load_gaps()
    now = pd.Timestamp.now(tz="UTC")
    skipped = [r for r in todo
               if pd.to_datetime(r, utc=True) in known_gaps and settled_gap(r, now)]
    todo = [r for r in todo if r not in set(skipped)]

    if limit:
        todo = todo[:limit]

    print(f"backbone {config.BACKBONE_MODEL}: {len(wanted)} runs in range, "
          f"{cached} already cached, {len(todo)} to fetch"
          + (f", {len(skipped)} known archive gaps skipped" if skipped else "")
          + (f" (limited to {limit})" if limit else ""))

    # Individual archive reads are slow (often 20-60s for older dates), so these
    # run concurrently. Only the fetch is parallel; merging into the parquet
    # cache stays on the main thread.
    gaps, unreachable, batch, added = [], [], [], 0
    t0 = time.time()
    done = 0

    def work(run: datetime):
        try:
            return run, fetch_run(run), None
        except wxio.SourceError as exc:
            return run, None, exc

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = {pool.submit(work, r): r for r in todo}
        for fut in as_completed(futures):
            run, frame, exc = fut.result()
            done += 1
            if exc is not None:
                # The server hung or errored repeatedly on this specific run.
                # That is a hole in the archive, not a reason to fabricate a
                # value; it is recorded, and a systemic rate is caught below.
                unreachable.append((run, str(exc)[:80]))
                print(f"  unreachable: {run:%Y-%m-%d} ({str(exc)[:60]})", flush=True)
            elif frame is None:
                gaps.append(run)
            else:
                batch.append(frame)

            if done % 25 == 0 or done == len(todo):
                rate = done / max(time.time() - t0, 1e-9)
                eta = (len(todo) - done) / max(rate, 1e-9) / 60
                print(f"  {done}/{len(todo)} runs | {rate:.1f}/s | ETA {eta:.0f}m | "
                      f"{len(gaps)} gaps | {len(unreachable)} unreachable", flush=True)

            # Flush periodically so a long backfill is resumable (hard rule 5).
            if len(batch) >= 100 or done == len(todo):
                if batch:
                    added += wxio.merge_raw(pd.concat(batch, ignore_index=True))
                    batch = []
                    print(f"    flushed -> +{added} rows total", flush=True)

    # Recorded so a settled hole stops being re-asked on every cron tick. Only
    # gaps are recorded, never transport failures: an unreachable run may well
    # be there, and writing it off would lose real data.
    if gaps or skipped:
        known_gaps.update({pd.to_datetime(g, utc=True): now for g in gaps})
        for g in skipped:
            known_gaps.setdefault(pd.to_datetime(g, utc=True), now)
        save_gaps(known_gaps)

    if gaps:
        settled = [g for g in gaps if settled_gap(g, now)]
        print(f"  archive gaps -- server reported {len(gaps)} runs not available:")
        for g in gaps[:10]:
            print(f"    {g:%Y-%m-%dT%H:%M}Z"
                  + ("" if settled_gap(g, now) else "  (recent -- will retry)"))
        if len(gaps) > 10:
            print(f"    ... and {len(gaps) - 10} more")
        if settled:
            print(f"  {len(settled)} of these are older than {GAP_SETTLED_DAYS}d "
                  f"and will be skipped from now on ({gap_path().name}); "
                  f"--retry-gaps forces a recheck")

    if unreachable:
        print(f"  unreachable -- {len(unreachable)} runs failed every attempt:")
        for run, why in unreachable[:10]:
            print(f"    {run:%Y-%m-%dT%H:%M}Z  {why}")
        if len(unreachable) > 10:
            print(f"    ... and {len(unreachable) - 10} more")
        frac = len(unreachable) / max(len(todo), 1)
        # The fraction only means something over a reasonable number of
        # attempts. On an incremental run of a few outstanding days, one
        # timeout is already 25% and would abort a healthy pull.
        if len(todo) >= config.MIN_RUNS_FOR_OUTAGE_CHECK and frac > config.MAX_UNREACHABLE_FRACTION:
            print(f"  {frac:.1%} unreachable, above the "
                  f"{config.MAX_UNREACHABLE_FRACTION:.0%} limit. These runs are not "
                  f"cached, so re-running retries exactly them; try "
                  f"--workers 2 --timeout 90 before treating this as an outage.")
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
                    "value": value[keep],
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
    config.add_station_arg(ap)
    ap.add_argument("--start", type=date.fromisoformat, default=config.BACKBONE_START)
    ap.add_argument("--end", type=date.fromisoformat,
                    default=date.today() - timedelta(days=1))
    ap.add_argument("--limit", type=int, default=None)
    # Stragglers that time out under concurrency usually succeed on a slower,
    # more patient pass: --workers 2 --timeout 90
    ap.add_argument("--workers", type=int, default=config.MAX_WORKERS)
    ap.add_argument("--timeout", type=int, default=config.HTTP_TIMEOUT)
    ap.add_argument("--skip-backbone", action="store_true")
    ap.add_argument("--skip-ensemble", action="store_true")
    ap.add_argument("--retry-gaps", action="store_true",
                    help="re-attempt runs previously reported absent")
    args = ap.parse_args()
    config.use_station(args.station)
    config.MAX_WORKERS = args.workers
    config.HTTP_TIMEOUT = args.timeout

    if not args.skip_backbone:
        ingest_backbone(args.start, args.end, args.limit, args.retry_gaps)
    if not args.skip_ensemble:
        ingest_ensemble()

    daily = wxio.build_daily()
    print(f"daily.parquet rebuilt: {len(daily)} rows "
          f"({daily.attrs['dropped_short_days']} short days dropped)")


if __name__ == "__main__":
    main()
