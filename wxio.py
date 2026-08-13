"""Shared HTTP, parquet cache and daily derivation.

Two parquet files only:
  data/raw_hourly.parquet  every hourly value ever pulled, long format
  data/daily.parquet       derived station-local daily max

raw_hourly schema
  source                'asos_obs' | 'single_runs' | 'ensemble'
  model                 '' for observations, else the Open-Meteo model id
  member                <NA> for deterministic and observations, else 0..N
                        (0 = control)
  issue_time            UTC model run init. <NA> when the source does not
                        expose one -- never guessed.
  issue_time_confirmed  True only when the run was named in the request and
                        validated server-side.
  valid_time            UTC timestamp the value applies to
  variable              e.g. 'temperature_2m'
  value_c               degrees Celsius
  fetched_at            UTC time of the pull; upper bound on issue_time
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
import requests

import config

RAW_COLUMNS = [
    "source",
    "model",
    "member",
    "issue_time",
    "issue_time_confirmed",
    "valid_time",
    "variable",
    "value_c",
    "fetched_at",
]

# (model, issue_time) pairs identify a deterministic pull; observations and the
# ensemble are keyed on the rest of the row.
IDENTITY = ["source", "model", "member", "issue_time", "valid_time", "variable"]

# Rows whose issue_time is null (the ensemble) would otherwise all collapse onto
# each other, because every pull carries NaT for the same valid_times. Keying
# those on the pull date instead lets successive pulls accumulate as distinct
# observations of the forecast, without inventing an issue_time we cannot know.
DEDUP_KEY = IDENTITY + ["_pull_date"]


class SourceError(RuntimeError):
    """A pull failed in a way that must stop the run (hard rule 7)."""


def utcnow() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def get_json(url: str, params: dict) -> dict:
    """GET with retry. Distinguishes a real gap from a transient failure.

    Returns None when the server says the requested model run does not exist --
    that is a genuine gap in the archive, not an error to retry. Any other
    failure raises after HTTP_RETRIES attempts rather than returning partial or
    substituted data.
    """
    last = None
    for attempt in range(config.HTTP_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=config.HTTP_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 400:
                reason = ""
                try:
                    reason = r.json().get("reason", "")
                except ValueError:
                    reason = r.text[:200]
                if "not available" in reason:
                    return None  # archive gap, caller records it
                raise SourceError(f"{url} rejected request: {reason}")
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.RequestException as exc:
            last = f"{type(exc).__name__}: {exc}"
        if attempt < config.HTTP_RETRIES - 1:
            time.sleep(config.HTTP_BACKOFF * (2**attempt))
    raise SourceError(f"{url} failed after {config.HTTP_RETRIES} attempts: {last}")


def get_text(url: str, params: dict) -> str:
    last = None
    for attempt in range(config.HTTP_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=config.HTTP_TIMEOUT)
            if r.status_code == 200:
                return r.text
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.RequestException as exc:
            last = f"{type(exc).__name__}: {exc}"
        if attempt < config.HTTP_RETRIES - 1:
            time.sleep(config.HTTP_BACKOFF * (2**attempt))
    raise SourceError(f"{url} failed after {config.HTTP_RETRIES} attempts: {last}")


def empty_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": pd.Series(dtype="string"),
            "model": pd.Series(dtype="string"),
            "member": pd.Series(dtype="Int16"),
            "issue_time": pd.Series(dtype="datetime64[us, UTC]"),
            "issue_time_confirmed": pd.Series(dtype="boolean"),
            "valid_time": pd.Series(dtype="datetime64[us, UTC]"),
            "variable": pd.Series(dtype="string"),
            "value_c": pd.Series(dtype="float64"),
            "fetched_at": pd.Series(dtype="datetime64[us, UTC]"),
        }
    )


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reindex(columns=RAW_COLUMNS)
    df["source"] = df["source"].astype("string")
    df["model"] = df["model"].astype("string")
    df["member"] = df["member"].astype("Int16")
    df["issue_time_confirmed"] = df["issue_time_confirmed"].astype("boolean")
    df["variable"] = df["variable"].astype("string")
    df["value_c"] = df["value_c"].astype("float64")
    for col in ("issue_time", "valid_time", "fetched_at"):
        s = pd.to_datetime(df[col], utc=True, errors="raise")
        df[col] = s.astype("datetime64[us, UTC]")
    return df


def read_raw() -> pd.DataFrame:
    if not config.RAW_HOURLY.exists():
        return empty_raw()
    return _coerce(pd.read_parquet(config.RAW_HOURLY))


def check_rule_1(df: pd.DataFrame) -> None:
    """No value may come from a run issued after the time it describes."""
    have = df["issue_time"].notna()
    bad = df[have & (df["issue_time"] > df["valid_time"])]
    if len(bad):
        raise SourceError(
            f"hard rule 1 violation: {len(bad)} rows with issue_time > valid_time, "
            f"first: {bad.iloc[0].to_dict()}"
        )


def merge_raw(new: pd.DataFrame) -> int:
    """Append to the raw cache, de-duplicating on IDENTITY. Returns rows added."""
    if new.empty:
        return 0
    new = _coerce(new)
    check_rule_1(new)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    old = read_raw()
    combined = pd.concat([old, new], ignore_index=True)
    # Later pull wins for an identical key, so a re-pull of a known run refreshes
    # rather than duplicates. Rows without an issue_time are additionally keyed
    # on their pull date (see DEDUP_KEY).
    combined["_pull_date"] = combined["fetched_at"].dt.floor("D")
    combined.loc[combined["issue_time"].notna(), "_pull_date"] = pd.NaT
    combined = combined.drop_duplicates(subset=DEDUP_KEY, keep="last")
    combined = combined.drop(columns="_pull_date")
    combined = combined.sort_values(["source", "model", "issue_time", "valid_time", "member"])
    combined = combined.reset_index(drop=True)
    combined.to_parquet(config.RAW_HOURLY, index=False)
    return len(combined) - len(old)


def cached_issue_times(source: str, model: str) -> set[pd.Timestamp]:
    """Issue times already on disk, so a re-run never re-downloads (rule 5)."""
    df = read_raw()
    if df.empty:
        return set()
    sel = df[(df["source"] == source) & (df["model"] == model)]
    return set(sel["issue_time"].dropna().unique())


# Daily sources that are authoritative rather than derived from raw_hourly, and
# so must survive a rebuild of daily.parquet.
AUTHORITATIVE_DAILY = ("asos_daily_official",)


def build_daily(official: pd.DataFrame | None = None) -> pd.DataFrame:
    """Derive station-local daily max from the raw hourly cache.

    This is the one place UTC is converted to station-local time (rule 6).
    Days with fewer than MIN_HOURS_PER_DAY values are dropped and counted --
    never interpolated or filled.

    `official` carries station daily maxima that are reported rather than
    derived (see ingest_asos.fetch_official_daily). When it is None, any such
    rows already in daily.parquet are preserved so that rebuilding from a
    forecast pull does not silently drop the target variable.
    """
    raw = read_raw()
    if raw.empty:
        raise SourceError("raw_hourly.parquet is empty; run the ingestion modules first")

    df = raw[raw["variable"] == config.VARIABLE].copy()
    local = df["valid_time"].dt.tz_convert(config.STATION_TZ)
    df["local_date"] = local.dt.date

    # Rows without an issue_time are separated by pull date for the same reason
    # merge_raw keys on it: otherwise successive ensemble pulls collapse into a
    # single day and the max is taken across unrelated runs.
    df["pull_date"] = df["fetched_at"].dt.floor("D")
    df.loc[df["issue_time"].notna(), "pull_date"] = pd.NaT

    keys = ["source", "model", "member", "issue_time", "issue_time_confirmed",
            "pull_date", "local_date"]
    grouped = df.groupby(keys, dropna=False, observed=True)
    daily = grouped["value_c"].agg(tmax_c="max", n_hours="count").reset_index()

    short = int((daily["n_hours"] < config.MIN_HOURS_PER_DAY).sum())
    daily = daily[daily["n_hours"] >= config.MIN_HOURS_PER_DAY].copy()

    # lead_days is measured in station-local days between the run and the target
    # day, and is only meaningful where issue_time is confirmed.
    issue_local = pd.to_datetime(daily["issue_time"], utc=True).dt.tz_convert(config.STATION_TZ)
    lead = pd.to_datetime(daily["local_date"]) - pd.to_datetime(issue_local.dt.date)
    daily["lead_days"] = lead.dt.days.astype("Int16")

    # Re-attach reported (non-derived) daily rows.
    if official is None and config.DAILY.exists():
        prev = pd.read_parquet(config.DAILY)
        official = prev[prev["source"].isin(AUTHORITATIVE_DAILY)]
    if official is not None and len(official):
        official = official.copy()
        if "pull_date" not in official.columns:
            official["pull_date"] = pd.NaT
        daily = pd.concat([daily, official], ignore_index=True)
        daily = daily.drop_duplicates(
            subset=["source", "model", "member", "issue_time", "pull_date", "local_date"],
            keep="last",
        )

    daily["n_hours"] = daily["n_hours"].astype("Int32")
    daily = daily.sort_values(["source", "model", "local_date", "issue_time", "member"])
    daily = daily.reset_index(drop=True)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(config.DAILY, index=False)
    # Set last: concat above does not carry attrs through.
    daily.attrs["dropped_short_days"] = short
    return daily
