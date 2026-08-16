"""Ingest NWS/NBM day-ahead max temperature for the target station.

Source is the NBM station bulletin archive on NOAA's public S3 bucket
(noaa-nbm-grib2-pds), not api.weather.gov. The api.weather.gov gridpoints
endpoint serves only the current forecast -- no date parameter, no archive, and
no percentiles -- so it cannot support a historical comparison at all.

The NBE ("extended") bulletin carries, per station:

    FHR    24  36| 48  60| ...     forecast hour
    TXN    89  74| 88  74| ...     max/min temperature, degrees F
    XND     1   1|  2   1| ...     standard deviation of TXN

Day alignment, verified against observations rather than assumed: for the 00Z
cycle on date D, the TXN entry at FHR 24 (valid D+1 00Z, i.e. the 12-hour
daytime period ending 20:00 local on D) is the maximum for **local day D**.
Checked against reported maxima at KNYC for 2025-08-12..14, which matched
89/88/89 exactly; the alternative reading (using the column's printed day label)
disagreed with observations on most days.

That makes the 00Z cycle on D a lead +1d forecast for local day D issued at
D 00:00 UTC -- the same issue time and lead as the ECMWF backbone, so the two
are directly comparable.

Usage:
    python3 ingest_nbm.py [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--workers N]
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

import config
import wxio

BUCKET = "https://noaa-nbm-grib2-pds.s3.amazonaws.com"
CYCLE_HOUR = 0            # matches the ECMWF backbone run hour
TARGET_FHR = 24           # lead +1d for a 00Z cycle
SOURCE = "nbm"
MODEL = "nbm_nbe"

# XND is printed as whole degrees F and is occasionally 0, which would make the
# predictive distribution degenerate. Floored rather than dropped, and the floor
# is reported so it is never silent.
SIGMA_FLOOR_F = 0.5


def bulletin_url(cycle: datetime) -> str:
    return (f"{BUCKET}/blend.{cycle:%Y%m%d}/{cycle:%H}/text/"
            f"blend_nbetx.t{cycle:%H}z")


def extract_station_block(body: bytes, station: str) -> list[str] | None:
    """Pull one station's block out of a ~21 MB multi-station bulletin."""
    marker = re.compile(rb"^\s*" + station.encode() + rb"\s+NBM\b", re.M)
    m = marker.search(body)
    if not m:
        return None
    chunk = body[m.start(): m.start() + 4000].decode("ascii", "replace")
    return chunk.splitlines()


def parse_block(lines: list[str]) -> tuple[float, float] | None:
    """Return (tmax_F, sigma_F) at TARGET_FHR, or None if absent.

    Columns are read by character position off the FHR row, so the parser does
    not depend on field widths that differ between bulletin versions (V4.3 and
    V5.0 both appear across the archive).
    """
    rows = {}
    for line in lines:
        key = line[:5].strip()
        if key in ("FHR", "TXN", "XND", "TMP") and key not in rows:
            rows[key] = line
        if len(rows) == 4:
            break
    if not {"FHR", "TXN", "XND"} <= set(rows):
        return None

    # Values are right-aligned in fixed-width fields, so a three-digit reading
    # (102 F) starts one character left of where a two-digit forecast hour (24)
    # starts. Slicing the FHR digit span alone would read "102" as "02".
    # Each field is therefore taken from the end of the previous field up to the
    # end of this one, then stripped of padding and the "|" day separators.
    ends = [mm.end() for mm in re.finditer(r"\d+", rows["FHR"][4:])]
    if not ends:
        return None
    fields, prev = [], 0
    for e in ends:
        fields.append((prev, e))
        prev = e

    def cell(line: str, field: tuple[int, int]) -> float | None:
        text = line[4:][field[0]: field[1]].strip(" |")
        return float(text) if re.fullmatch(r"-?\d+", text) else None

    for field in fields:
        if cell(rows["FHR"], field) != TARGET_FHR:
            continue
        tmax = cell(rows["TXN"], field)
        sigma = cell(rows["XND"], field)
        if tmax is None:
            return None

        # Column-alignment guard: the daytime maximum cannot sit below the
        # instantaneous temperature forecast in the same column. A misparse
        # produces a plausible-looking small number rather than an error, so
        # this is checked rather than assumed.
        tmp = cell(rows.get("TMP", ""), field) if "TMP" in rows else None
        if tmp is not None and tmax < tmp - 5.0:
            raise wxio.SourceError(
                f"TXN {tmax:.0f}F below TMP {tmp:.0f}F in the same column -- "
                f"column alignment is wrong, refusing to cache")

        return tmax, (sigma if sigma is not None else float("nan"))
    return None


def fetch_cycle(cycle: datetime) -> tuple[datetime, tuple[float, float] | None, str | None]:
    url = bulletin_url(cycle)
    last = None
    for attempt in range(config.HTTP_RETRIES):
        try:
            r = requests.get(url, timeout=config.HTTP_TIMEOUT)
            if r.status_code == 404:
                return cycle, None, "no bulletin for this cycle"
            if r.status_code == 200:
                block = extract_station_block(r.content, config.STATION_ICAO)
                if block is None:
                    return cycle, None, f"{config.STATION_ICAO} absent from bulletin"
                try:
                    parsed = parse_block(block)
                except wxio.SourceError as exc:
                    return cycle, None, str(exc)
                if parsed is None:
                    return cycle, None, f"no FHR {TARGET_FHR} row"
                return cycle, parsed, None
            last = f"HTTP {r.status_code}"
        except requests.RequestException as exc:
            last = f"{type(exc).__name__}"
        if attempt < config.HTTP_RETRIES - 1:
            import time
            time.sleep(config.HTTP_BACKOFF * (2 ** attempt))
    return cycle, None, f"failed after {config.HTTP_RETRIES} attempts: {last}"


def cached_cycles() -> set[pd.Timestamp]:
    if not config.DAILY.exists():
        return set()
    d = pd.read_parquet(config.DAILY)
    d = d[d["source"] == SOURCE]
    if d.empty:
        return set()
    return set(pd.to_datetime(d["issue_time"], utc=True).unique())


def main() -> None:
    ap = argparse.ArgumentParser()
    config.add_station_arg(ap)
    ap.add_argument("--start", type=date.fromisoformat,
                    default=date(2025, 8, 12))
    ap.add_argument("--end", type=date.fromisoformat,
                    default=date.today() - timedelta(days=2))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--refresh", action="store_true",
                    help="refetch cycles already cached (use after a parser fix)")
    args = ap.parse_args()
    config.use_station(args.station)

    cycles = []
    day = args.start
    while day <= args.end:
        cycles.append(datetime(day.year, day.month, day.day, CYCLE_HOUR,
                               tzinfo=timezone.utc))
        day += timedelta(days=1)

    have = set() if args.refresh else cached_cycles()
    todo = [c for c in cycles if pd.Timestamp(c) not in have]
    print(f"NBM {MODEL} @ {config.STATION_ICAO}: {len(cycles)} cycles in range, "
          f"{len(cycles) - len(todo)} cached, {len(todo)} to fetch")

    got, missing = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_cycle, c) for c in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            cycle, parsed, why = fut.result()
            if parsed is None:
                missing.append((cycle, why))
            else:
                tmax_f, sigma_f = parsed
                got.append({"issue_time": pd.Timestamp(cycle),
                            "local_date": pd.Timestamp(cycle.date()),
                            "tmax_f": tmax_f, "sigma_f": sigma_f})
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} cycles | {len(got)} parsed | "
                      f"{len(missing)} missing", flush=True)

    if missing:
        print(f"  {len(missing)} cycles without a usable forecast:")
        for c, why in missing[:8]:
            print(f"    {c:%Y-%m-%d}  {why}")
        if len(missing) > 8:
            print(f"    ... and {len(missing) - 8} more")

    if not got:
        # Nothing new is only a failure when there was also nothing cached. On a
        # warm cache the outstanding cycles are the ones with no bulletin at
        # all, and re-running should be a no-op rather than an error.
        if have:
            print(f"  nothing new to add; {len(have)} cycles already cached")
            daily = wxio.build_daily()
            n = int((daily["source"] == SOURCE).sum())
            print(f"  daily.parquet rebuilt: {len(daily)} rows, {n} NBM days")
            return
        raise wxio.SourceError("no NBM forecasts parsed and nothing cached")

    df = pd.DataFrame(got).sort_values("local_date")
    floored = int((df["sigma_f"].fillna(0) < SIGMA_FLOOR_F).sum())
    nan_sigma = int(df["sigma_f"].isna().sum())
    df["sigma_f"] = df["sigma_f"].fillna(SIGMA_FLOOR_F).clip(lower=SIGMA_FLOOR_F)
    print(f"  parsed {len(df)} cycles; sigma floored at {SIGMA_FLOOR_F} F on "
          f"{floored} days ({nan_sigma} had no XND value)")

    rows = pd.DataFrame({
        "source": SOURCE,
        "model": MODEL,
        "member": pd.NA,
        "issue_time": df["issue_time"].values,
        # The cycle is named in the object path and the bulletin header repeats
        # it, so the run behind each value is identifiable -- unlike the
        # ensemble endpoint.
        "issue_time_confirmed": True,
        "local_date": df["local_date"].values,
        "tmax_c": (df["tmax_f"].values - 32.0) * 5.0 / 9.0,
        "tmax_sigma_c": df["sigma_f"].values * 5.0 / 9.0,
        "n_hours": pd.NA,
        "lead_days": 1,
    })

    daily = wxio.build_daily(extra=rows)
    n = int((daily["source"] == SOURCE).sum())
    print(f"  daily.parquet rebuilt: {len(daily)} rows, {n} NBM days "
          f"({df['local_date'].min():%Y-%m-%d} -> {df['local_date'].max():%Y-%m-%d})")


if __name__ == "__main__":
    main()
