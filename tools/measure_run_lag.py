"""Measure how long after issue an ECMWF 00Z/12Z run becomes retrievable.

The publication lag of the Open-Meteo Single Runs archive sets when a forecast
for today can first be used, which in turn sets the useful market-polling
window. It cannot be reconstructed after the fact -- the API says whether a run
is available now, not when it became available -- so it has to be sampled by
watching pending runs and recording first sight.

Appends one row per run to data/run_lag.parquet:
    model, run (UTC), first_seen (UTC), lag_hours, polls

Usage:
    python3 tools/measure_run_lag.py [--hours 24] [--interval 300]
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

OUT = config.DATA_DIR / "run_lag.parquet"
URL = "https://single-runs-api.open-meteo.com/v1/forecast"
MODEL = "ecmwf_ifs"
RUN_HOURS = (0, 12)          # the only cycles this model archives


def available(run: datetime) -> bool | None:
    """True/False, or None when the check itself failed and proves nothing."""
    try:
        r = requests.get(URL, params={
            "latitude": config.LATITUDE, "longitude": config.LONGITUDE,
            "hourly": "temperature_2m", "models": MODEL,
            "run": run.strftime("%Y-%m-%dT%H:%M")}, timeout=60)
    except requests.RequestException:
        return None
    if r.status_code == 200:
        return True
    if r.status_code == 400 and "not available" in r.text:
        return False
    return None


def pending_runs(now: datetime, back_hours: int = 20) -> list[datetime]:
    """Recent cycles that could plausibly still be unpublished."""
    out = []
    t = now.replace(minute=0, second=0, microsecond=0)
    for h in range(back_hours + 1):
        c = t - timedelta(hours=h)
        if c.hour in RUN_HOURS and c <= now:
            out.append(c)
    return sorted(set(out))


def load() -> pd.DataFrame:
    if OUT.exists():
        return pd.read_parquet(OUT)
    return pd.DataFrame(columns=["model", "run", "first_seen", "lag_hours", "polls"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--interval", type=int, default=300, help="seconds between sweeps")
    args = ap.parse_args()

    done = load()
    seen = set(pd.to_datetime(done["run"], utc=True)) if len(done) else set()
    polls: dict[datetime, int] = {}
    deadline = datetime.now(timezone.utc) + timedelta(hours=args.hours)
    print(f"watching {MODEL} 00Z/12Z runs until {deadline:%Y-%m-%d %H:%M}Z, "
          f"every {args.interval}s; {len(seen)} already recorded", flush=True)

    while datetime.now(timezone.utc) < deadline:
        now = datetime.now(timezone.utc)
        for run in pending_runs(now):
            if pd.Timestamp(run) in seen:
                continue
            ok = available(run)
            polls[run] = polls.get(run, 0) + 1
            if ok is None:
                continue
            if ok:
                lag = (now - run).total_seconds() / 3600.0
                # Only a run we watched from unavailable to available gives a
                # true first-seen. One already published when we started only
                # bounds the lag from above, so it is recorded with polls=0 and
                # excluded from the summary.
                row = {"model": MODEL, "run": pd.Timestamp(run),
                       "first_seen": pd.Timestamp(now), "lag_hours": lag,
                       "polls": polls.get(run, 0) if polls.get(run, 0) > 1 else 0}
                done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
                seen.add(pd.Timestamp(run))
                config.DATA_DIR.mkdir(parents=True, exist_ok=True)
                done.to_parquet(OUT, index=False)
                kind = "measured" if row["polls"] else "bound only"
                print(f"  {run:%Y-%m-%d %H}Z available at {now:%H:%M}Z "
                      f"-> {lag:.2f}h ({kind})", flush=True)
        time.sleep(args.interval)

    print("done", flush=True)


if __name__ == "__main__":
    main()
