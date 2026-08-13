"""Ensemble forecasts for the KNYC coordinates from the Open-Meteo ensemble API.

Every member is kept separately. Nothing here averages, and nothing downstream
should have to guess which member a value came from, so the frame is long:

    station model member variable issue_time valid_time lead_hours value

`issue_time` and `valid_time` are both explicit UTC columns on every row, and
`lead_hours` is their difference (rule 1). One request covers one forecast cycle,
so `issue_time` is a property of the request rather than something inferred from
the response afterwards.

Two parquet caches are written (rule 5):

  data/raw/openmeteo/<model>_<run>.parquet   one chunk per model per cycle
  data/fcst.parquet                          all chunks concatenated

CAVEAT worth reading before this output is trusted for training. The ensemble
endpoint serves an archived date range; it does not state which model run the
values came from. This module labels a cycle with the `issue_time` it requested,
which is correct only if the API honours the run implied by that request. Until
that is confirmed against a live response, `issue_time` here is an assertion
about the request, not a fact read back from the API. `check_no_leakage` enforces
the arithmetic that follows from it, which is a weaker guarantee than the
provenance rule 1 actually wants.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from obs import (
    STATION,
    STATION_LAT,
    STATION_LON,
    IngestError,
    is_policy_denial,
)

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"

VARIABLES = [
    "temperature_2m",
    "cloud_cover",
    "wind_speed_10m",
    "dew_point_2m",
]

DEFAULT_MODELS = ["gfs025"]

# Hours past the run time to keep. Day-2/day-3 leads are the interesting range
# for a daily-max post-processor; going further mostly adds rows.
HORIZON_HOURS = 120

# UTC hour of the model cycle each daily request is labelled with.
DEFAULT_RUN_HOUR = 0

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw" / "openmeteo"
FCST_PATH = REPO_ROOT / "data" / "fcst.parquet"

REQUEST_TIMEOUT = 120
MAX_ATTEMPTS = 4
# Open-Meteo's free tier is rate limited; one cycle per request means a long
# backfill is many requests, so they are spaced rather than bursted.
SLEEP_BETWEEN_REQUESTS = 1.0

MEMBER_RE = re.compile(r"^(?P<variable>.+)_member(?P<member>\d+)$")


def _get(params: dict) -> dict:
    """One ensemble request. Transport failures retry; refusals surface at once."""
    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = requests.get(
                ENSEMBLE_URL, params=params, timeout=REQUEST_TIMEOUT
            )
        except requests.RequestException as exc:
            if is_policy_denial(exc):
                raise IngestError(
                    f"egress policy blocked {ENSEMBLE_URL} (proxy answered 403/407). "
                    f"This host is not reachable from this session; it is not a "
                    f"transient error and was not retried."
                ) from exc
            last_error = exc
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise IngestError(
                f"Open-Meteo unreachable after {MAX_ATTEMPTS} attempts: {exc}"
            ) from exc
        if response.status_code != 200:
            raise IngestError(
                f"Open-Meteo returned HTTP {response.status_code} for "
                f"{params.get('start_date')}: {response.text[:300]}"
            )
        return response.json()
    raise IngestError(f"Open-Meteo unreachable: {last_error}")


def _parse(payload: dict, model: str, issue_time: pd.Timestamp) -> pd.DataFrame:
    """Melt one cycle's response into long rows, one per member per hour."""
    if "hourly" not in payload:
        reason = payload.get("reason", "no 'hourly' block in response")
        raise IngestError(f"Open-Meteo response for {model} {issue_time}: {reason}")

    hourly = payload["hourly"]
    if "time" not in hourly:
        raise IngestError(f"Open-Meteo response for {model} {issue_time} has no time axis")

    valid_time = pd.to_datetime(pd.Series(hourly["time"]), utc=True, errors="coerce")
    if valid_time.isna().any():
        raise IngestError(f"unparseable timestamps for {model} {issue_time}")

    lat = float(payload.get("latitude", "nan"))
    lon = float(payload.get("longitude", "nan"))
    if abs(lat - STATION_LAT) > 0.5 or abs(lon - STATION_LON) > 0.5:
        raise IngestError(
            f"Open-Meteo served grid point ({lat}, {lon}), too far from "
            f"{STATION} ({STATION_LAT}, {STATION_LON})"
        )

    records: list[pd.DataFrame] = []
    seen_variables: set[str] = set()

    for key, values in hourly.items():
        if key == "time":
            continue
        match = MEMBER_RE.match(key)
        if match:
            variable = match.group("variable")
            member = int(match.group("member"))
        else:
            # The unsuffixed series is the control run.
            variable = key
            member = 0
        if variable not in VARIABLES:
            continue
        if len(values) != len(valid_time):
            raise IngestError(
                f"{key} for {model} {issue_time} has {len(values)} values against "
                f"{len(valid_time)} timestamps"
            )
        seen_variables.add(variable)
        records.append(
            pd.DataFrame(
                {
                    "valid_time": valid_time,
                    "variable": variable,
                    "member": member,
                    "value": pd.to_numeric(pd.Series(values), errors="coerce"),
                }
            )
        )

    absent = set(VARIABLES) - seen_variables
    if absent:
        raise IngestError(
            f"Open-Meteo returned no data for {sorted(absent)} "
            f"({model} {issue_time}); requested {VARIABLES}"
        )
    if not records:
        raise IngestError(f"no member series parsed for {model} {issue_time}")

    frame = pd.concat(records, ignore_index=True)
    frame["issue_time"] = issue_time
    frame["model"] = model
    frame["station"] = STATION
    frame["lead_hours"] = (
        frame["valid_time"] - frame["issue_time"]
    ).dt.total_seconds() / 3600.0

    ordered = [
        "station",
        "model",
        "member",
        "variable",
        "issue_time",
        "valid_time",
        "lead_hours",
        "value",
    ]
    return frame.loc[:, ordered].sort_values(
        ["variable", "member", "valid_time"]
    ).reset_index(drop=True)


def check_no_leakage(frame: pd.DataFrame) -> None:
    """Rule 1, enforced arithmetically: nothing may be valid before it was issued."""
    if frame.empty:
        return
    offending = frame.loc[frame["issue_time"] > frame["valid_time"]]
    if not offending.empty:
        first = offending.iloc[0]
        raise IngestError(
            f"{len(offending)} rows carry issue_time after valid_time, e.g. "
            f"{first['model']} member {first['member']} issued {first['issue_time']} "
            f"valid {first['valid_time']}"
        )
    if (frame["lead_hours"] < 0).any():
        raise IngestError("negative lead_hours present after parsing")


def _chunk_path(model: str, issue_time: pd.Timestamp) -> Path:
    stamp = issue_time.strftime("%Y%m%dT%H")
    return RAW_DIR / f"{model}_{stamp}.parquet"


def fetch_cycle(
    model: str,
    issue_time: pd.Timestamp,
    refresh: bool = False,
) -> pd.DataFrame:
    """One model cycle, from cache when present (rule 5)."""
    path = _chunk_path(model, issue_time)
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    start = issue_time.date()
    end = (issue_time + pd.Timedelta(hours=HORIZON_HOURS)).date()
    params = {
        "latitude": STATION_LAT,
        "longitude": STATION_LON,
        "hourly": ",".join(VARIABLES),
        "models": model,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        # Rule 6: ask for UTC and convert nowhere in this module.
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
    }
    chunk = _parse(_get(params), model, issue_time)
    chunk = chunk.loc[
        (chunk["lead_hours"] >= 0) & (chunk["lead_hours"] <= HORIZON_HOURS)
    ].reset_index(drop=True)
    check_no_leakage(chunk)
    chunk.to_parquet(path, index=False)
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    return chunk


def build(
    start: dt.date,
    end: dt.date,
    models: list[str] | None = None,
    run_hour: int = DEFAULT_RUN_HOUR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Every cycle from `start` to `end` inclusive, concatenated and cached."""
    models = models or DEFAULT_MODELS
    if start > end:
        raise IngestError(f"start {start} is after end {end}")

    chunks: list[pd.DataFrame] = []
    for model in models:
        day = start
        while day <= end:
            issue_time = pd.Timestamp(
                dt.datetime.combine(day, dt.time(hour=run_hour)), tz="UTC"
            )
            print(f"[fcst] {model} cycle {issue_time}", file=sys.stderr)
            chunks.append(fetch_cycle(model, issue_time, refresh=refresh))
            day += dt.timedelta(days=1)

    if not chunks:
        raise IngestError("no cycles requested")

    frame = (
        pd.concat(chunks, ignore_index=True)
        .drop_duplicates(
            subset=["model", "member", "variable", "issue_time", "valid_time"],
            keep="last",
        )
        .sort_values(["model", "issue_time", "variable", "member", "valid_time"])
        .reset_index(drop=True)
    )
    check_no_leakage(frame)

    FCST_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(FCST_PATH, index=False)
    print(
        f"[fcst] {len(frame)} rows, "
        f"{frame['member'].nunique()} members, "
        f"{frame['issue_time'].nunique()} cycles "
        f"-> {FCST_PATH.relative_to(REPO_ROOT)}",
        file=sys.stderr,
    )
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="first cycle date, YYYY-MM-DD")
    parser.add_argument(
        "--end",
        default=dt.date.today().isoformat(),
        help="last cycle date, YYYY-MM-DD (default today)",
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--run-hour", type=int, default=DEFAULT_RUN_HOUR)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    try:
        build(
            start=dt.date.fromisoformat(args.start),
            end=dt.date.fromisoformat(args.end),
            models=args.models,
            run_hour=args.run_hour,
            refresh=args.refresh,
        )
    except IngestError as exc:
        print(f"[fcst] FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
