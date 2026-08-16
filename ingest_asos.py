"""Ingest ASOS hourly observations for the target station from the Iowa
Environmental Mesonet.

These are the truth values. They carry no issue_time -- an observation is not a
forecast -- so issue_time is left null and issue_time_confirmed is False. Nothing
downstream may treat an observation row as a feature with provenance.

Usage:
    python3 ingest_asos.py [--start YYYY-MM-DD] [--end YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import io
from datetime import date, timedelta

import pandas as pd

import config
import wxio


def fetch_asos(start: date, end: date) -> pd.DataFrame:
    """One request for the whole range -- IEM serves multi-year spans directly."""
    params = {
        "station": config.STATION_IEM_ID,
        "data": "tmpf",
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": end.year, "month2": end.month, "day2": end.day,
        "tz": "UTC",
        "format": "onlycomma",
        "latlon": "no",
        "missing": "empty",
        "trace": "empty",
        "report_type": 3,  # routine hourly METAR only, no specials
    }
    print(f"  GET ASOS {config.STATION_ICAO} {start} -> {end}")
    text = wxio.get_text(config.IEM_ASOS_URL, params)

    df = pd.read_csv(io.StringIO(text))
    if df.empty or "valid" not in df.columns:
        raise wxio.SourceError(
            f"ASOS response for {config.STATION_ICAO} had no usable rows; "
            f"first 200 chars: {text[:200]!r}"
        )
    return df


def to_raw(df: pd.DataFrame) -> pd.DataFrame:
    fetched = wxio.utcnow()

    valid = pd.to_datetime(df["valid"], utc=True, errors="coerce")
    tmpf = pd.to_numeric(df["tmpf"], errors="coerce")

    bad_time = int(valid.isna().sum())
    if bad_time:
        raise wxio.SourceError(f"{bad_time} ASOS rows had unparseable timestamps")

    # Drop missing observations rather than filling them (hard rule 7).
    missing = int(tmpf.isna().sum())
    keep = tmpf.notna()
    print(f"  {len(df)} reports, {missing} missing tmpf dropped, {int(keep.sum())} kept")

    out = pd.DataFrame(
        {
            "source": "asos_obs",
            "model": "",
            "member": pd.NA,
            "issue_time": pd.NaT,
            "issue_time_confirmed": False,
            "valid_time": valid[keep],
            "variable": config.VARIABLE,
            "value": (tmpf[keep] - 32.0) * 5.0 / 9.0,
            "fetched_at": fetched,
        }
    )
    # Keep the true report time. Rounding :51 reports to the hour pushes late
    # evening observations across the local-day boundary, which corrupts the
    # daily max. Observations and model output are joined at the daily level,
    # so they do not need to share an hourly grid.
    out = out.drop_duplicates(subset=["valid_time"], keep="last")
    return out


def fetch_official_daily(start: date, end: date) -> pd.DataFrame:
    """The station's reported daily maximum -- the model's target variable.

    Max-of-hourly-reports is not the same quantity: compared against the
    reported max over 1311 days at KNYC it differs by >0.5 C on 16% of days and
    >1.0 C on 3.6%, which is large next to the skill differences CRPS is meant
    to resolve. The reported value is the target; the hourly series is kept for
    features and QC.
    """
    params = {"station": config.STATION_IEM_ID, "network": config.IEM_NETWORK,
              "sdate": start.isoformat(), "edate": end.isoformat()}
    print(f"  GET official daily max {start} -> {end}")
    payload = wxio.get_json(config.IEM_DAILY_URL, params)
    if payload is None or "data" not in payload:
        raise wxio.SourceError(f"IEM daily API returned no data block for {config.STATION_ICAO}")

    df = pd.DataFrame(payload["data"])
    df["local_date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df[(df["local_date"] >= start) & (df["local_date"] <= end)]

    # tmpf_est marks a value the network estimated rather than measured. Those
    # are excluded, not silently accepted as truth (hard rule 7).
    est = int(df["tmpf_est"].fillna(False).astype(bool).sum())
    df = df[~df["tmpf_est"].fillna(False).astype(bool)]
    missing = int(df["max_tmpf"].isna().sum())
    df = df.dropna(subset=["max_tmpf"])
    print(f"  {len(df)} reported daily maxima ({est} estimated, {missing} missing, excluded)")

    return pd.DataFrame(
        {
            "source": "asos_daily_official",
            "model": "",
            "member": pd.NA,
            "issue_time": pd.NaT,
            "issue_time_confirmed": False,
            "local_date": df["local_date"].values,
            "tmax_c": (df["max_tmpf"].astype(float).values - 32.0) * 5.0 / 9.0,
            "n_hours": pd.NA,
            "lead_days": pd.NA,
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    config.add_station_arg(ap)
    ap.add_argument("--start", type=date.fromisoformat, default=config.OBS_START)
    ap.add_argument("--end", type=date.fromisoformat,
                    default=date.today() - timedelta(days=1))
    args = ap.parse_args()
    config.use_station(args.station)

    print(f"ASOS ingestion: {config.STATION_ICAO} ({config.STATION_IEM_ID})")
    raw = to_raw(fetch_asos(args.start, args.end))
    added = wxio.merge_raw(raw)
    print(f"  raw_hourly.parquet += {added} rows")

    official = fetch_official_daily(args.start, args.end)

    daily = wxio.build_daily(official=official)
    hourly_days = int((daily["source"] == "asos_obs").sum())
    target_days = int((daily["source"] == "asos_daily_official").sum())
    print(f"  daily.parquet rebuilt: {len(daily)} rows")
    print(f"    target (reported daily max) : {target_days} days")
    print(f"    hourly-derived max (QC)     : {hourly_days} days")
    print(f"  dropped {daily.attrs['dropped_short_days']} short days (<{config.MIN_HOURS_PER_DAY}h)")


if __name__ == "__main__":
    main()
