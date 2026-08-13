"""Hourly ASOS observations for KNYC, and the daily-max target derived from them.

Source: Iowa Environmental Mesonet ASOS archive (mesonet.agron.iastate.edu).

Time handling follows rule 6. Everything the archive returns is requested and
stored in UTC. The only place a local timezone appears is `daily_max`, where the
midnight-to-midnight window that defines the prediction target is by definition a
station-local concept. That function is the resolution boundary.

Two parquet caches are written (rule 5):

  data/raw/asos/KNYC_<year>.parquet   one chunk per calendar year, exactly as
                                      pulled, so the derived series can be
                                      rebuilt without re-hitting the archive
  data/obs.parquet                    the daily max target series

Nothing here fills, interpolates, or synthesizes a missing value. Days with
sparse observations are reported with the counts that produced them and left for
the caller to reject (rule 7).
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

STATION = "KNYC"
STATION_TZ = ZoneInfo("America/New_York")

# Central Park, NY. Checked against the coordinates the archive reports on every
# pull; a disagreement beyond COORD_TOLERANCE_DEG is a hard error rather than a
# silent drift onto a different site.
STATION_LAT = 40.7789
STATION_LON = -73.9692
COORD_TOLERANCE_DEG = 0.05

ARCHIVE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
START_YEAR = 2019

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw" / "asos"
OBS_PATH = REPO_ROOT / "data" / "obs.parquet"

# The archive writes "M" for missing. "T" is a trace amount and never appears in
# the temperature columns, so it is deliberately not treated as null here.
NULL_TOKENS = ["M", ""]

REQUEST_TIMEOUT = 120
MAX_ATTEMPTS = 4

# A day of hourly METARs should carry ~24 observations; KNYC usually reports more
# because of specials. Below this the daily max is not trustworthy as a target.
MIN_OBS_PER_DAY = 18


class IngestError(RuntimeError):
    """Raised when a pull or a validation fails. Never caught to substitute data."""


def is_policy_denial(exc: Exception) -> bool:
    """True for an egress-policy refusal rather than a flaky network.

    The agent proxy answers a blocked CONNECT with 403 (407 when unauthorised),
    which requests surfaces as a ProxyError rather than an HTTP status. Retrying
    those is pointless: the answer will not change, and four backoffs only delay
    the report.
    """
    if not isinstance(exc, requests.exceptions.ProxyError):
        return False
    return "403" in str(exc) or "407" in str(exc)


def _get(params: dict) -> str:
    """One archive request, with retries for transport failures only.

    A non-200 response or a policy denial is surfaced immediately. Retrying a
    refusal would only turn a clear failure into a slow one.
    """
    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = requests.get(ARCHIVE_URL, params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            if is_policy_denial(exc):
                raise IngestError(
                    f"egress policy blocked {ARCHIVE_URL} (proxy answered 403/407). "
                    f"This host is not reachable from this session; it is not a "
                    f"transient error and was not retried."
                ) from exc
            last_error = exc
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise IngestError(
                f"ASOS archive unreachable after {MAX_ATTEMPTS} attempts: {exc}"
            ) from exc
        if response.status_code != 200:
            raise IngestError(
                f"ASOS archive returned HTTP {response.status_code} for "
                f"{params.get('year1')}-{params.get('year2')}: {response.text[:300]}"
            )
        return response.text
    raise IngestError(f"ASOS archive unreachable: {last_error}")


def _parse(payload: str, year: int) -> pd.DataFrame:
    """Parse one CSV chunk into a UTC-indexed frame, validating as we go."""
    frame = pd.read_csv(
        io.StringIO(payload),
        na_values=NULL_TOKENS,
        keep_default_na=True,
    )
    if frame.empty:
        raise IngestError(f"ASOS archive returned no rows for {year}")

    required = {"station", "valid", "tmpf", "lat", "lon"}
    missing = required - set(frame.columns)
    if missing:
        raise IngestError(
            f"ASOS response for {year} is missing expected columns: {sorted(missing)}; "
            f"got {sorted(frame.columns)}"
        )

    stations = set(frame["station"].dropna().unique())
    if stations != {STATION}:
        raise IngestError(
            f"ASOS response for {year} contains unexpected stations: {sorted(stations)}"
        )

    # tz=UTC was requested, so the timestamps are UTC wall time without an offset.
    frame["valid_time"] = pd.to_datetime(frame["valid"], utc=True, errors="coerce")
    if frame["valid_time"].isna().any():
        bad = int(frame["valid_time"].isna().sum())
        raise IngestError(f"{bad} unparseable timestamps in the {year} ASOS chunk")

    lat = float(frame["lat"].dropna().median())
    lon = float(frame["lon"].dropna().median())
    if (
        abs(lat - STATION_LAT) > COORD_TOLERANCE_DEG
        or abs(lon - STATION_LON) > COORD_TOLERANCE_DEG
    ):
        raise IngestError(
            f"{STATION} coordinates from the archive ({lat:.4f}, {lon:.4f}) disagree "
            f"with the configured ({STATION_LAT}, {STATION_LON}) beyond "
            f"{COORD_TOLERANCE_DEG} deg"
        )

    frame["temp_c"] = (pd.to_numeric(frame["tmpf"], errors="coerce") - 32.0) * 5.0 / 9.0

    out = (
        frame.loc[:, ["valid_time", "temp_c", "tmpf"]]
        .sort_values("valid_time")
        .drop_duplicates(subset="valid_time", keep="last")
        .reset_index(drop=True)
    )
    out.insert(0, "station", STATION)
    return out


def _chunk_path(year: int) -> Path:
    return RAW_DIR / f"{STATION}_{year}.parquet"


def fetch_hourly(
    start_year: int = START_YEAR,
    end_date: dt.date | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Hourly observations from `start_year` through `end_date`, UTC throughout.

    Cached one chunk per year. A cached complete year is never re-requested; the
    current year is always refetched because it is still growing (rule 5 forbids
    re-downloading what is already on disk, not keeping an open year current).
    """
    end_date = end_date or dt.datetime.now(dt.timezone.utc).date()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    this_year = end_date.year
    chunks: list[pd.DataFrame] = []

    for year in range(start_year, this_year + 1):
        path = _chunk_path(year)
        year_is_closed = year < this_year
        if path.exists() and year_is_closed and not refresh:
            chunks.append(pd.read_parquet(path))
            continue

        chunk_end = min(dt.date(year, 12, 31), end_date) + dt.timedelta(days=1)
        params = {
            "station": STATION,
            "data": "tmpf",
            "year1": year,
            "month1": 1,
            "day1": 1,
            "year2": chunk_end.year,
            "month2": chunk_end.month,
            "day2": chunk_end.day,
            "tz": "UTC",
            "format": "onlycomma",
            "latlon": "yes",
            "missing": "M",
            "trace": "T",
            "report_type": [3, 4],
        }
        print(f"[obs] fetching {STATION} {year}", file=sys.stderr)
        chunk = _parse(_get(params), year)
        chunk.to_parquet(path, index=False)
        chunks.append(chunk)

    hourly = (
        pd.concat(chunks, ignore_index=True)
        .sort_values("valid_time")
        .drop_duplicates(subset="valid_time", keep="last")
        .reset_index(drop=True)
    )
    return hourly


def daily_max(hourly: pd.DataFrame) -> pd.DataFrame:
    """Daily max temperature over station-local midnight-to-midnight windows.

    This is the resolution boundary named in rule 6: the UTC instants come in,
    the local calendar day is applied here and only here, and the window bounds
    go back out as UTC so nothing downstream has to reason about local time.
    """
    if hourly.empty:
        raise IngestError("cannot derive a daily target from an empty hourly frame")

    local = hourly["valid_time"].dt.tz_convert(STATION_TZ)
    work = hourly.assign(local_date=local.dt.date)

    grouped = work.groupby("local_date", sort=True)
    daily = grouped.agg(
        temp_max_c=("temp_c", "max"),
        n_obs=("temp_c", "count"),
        first_obs_utc=("valid_time", "min"),
        last_obs_utc=("valid_time", "max"),
    ).reset_index()

    # Window bounds, carried explicitly so no downstream step re-derives them.
    #
    # Both edges are localised from a calendar date rather than derived by adding
    # 24h to the start. A tz-aware Timedelta add is absolute-time arithmetic, so
    # on the two DST days a year it would push the end an hour past (or short of)
    # the next local midnight and overlap the neighbouring day's window. Local
    # midnight never falls inside a US transition, so both edges are unambiguous.
    local_dates = pd.to_datetime(daily["local_date"])
    starts = local_dates.dt.tz_localize(STATION_TZ)
    ends = (local_dates + pd.Timedelta(days=1)).dt.tz_localize(STATION_TZ)
    daily["window_start_utc"] = starts.dt.tz_convert("UTC")
    daily["window_end_utc"] = ends.dt.tz_convert("UTC")

    # The target is only knowable once its window has closed. Recording this
    # keeps rule 1 checkable at join time instead of assumed.
    daily["valid_time"] = daily["window_end_utc"]

    daily["sparse"] = daily["n_obs"] < MIN_OBS_PER_DAY
    daily.insert(0, "station", STATION)

    ordered = [
        "station",
        "local_date",
        "temp_max_c",
        "n_obs",
        "sparse",
        "window_start_utc",
        "window_end_utc",
        "valid_time",
        "first_obs_utc",
        "last_obs_utc",
    ]
    return daily.loc[:, ordered]


def build(refresh: bool = False) -> pd.DataFrame:
    hourly = fetch_hourly(refresh=refresh)
    daily = daily_max(hourly)

    OBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(OBS_PATH, index=False)

    sparse = int(daily["sparse"].sum())
    nulls = int(daily["temp_max_c"].isna().sum())
    print(
        f"[obs] {len(daily)} daily rows "
        f"{daily['local_date'].min()} to {daily['local_date'].max()} "
        f"-> {OBS_PATH.relative_to(REPO_ROOT)}",
        file=sys.stderr,
    )
    if sparse or nulls:
        print(
            f"[obs] flagged {sparse} sparse days (<{MIN_OBS_PER_DAY} obs) and "
            f"{nulls} days with no usable temperature. Left in place, not filled.",
            file=sys.stderr,
        )
    return daily


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-pull every year instead of reading closed years from cache",
    )
    args = parser.parse_args()
    try:
        build(refresh=args.refresh)
    except IngestError as exc:
        print(f"[obs] FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
