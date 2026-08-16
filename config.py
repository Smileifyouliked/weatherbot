"""Station, source and cache configuration.

All times in this project are UTC internally (hard rule 6). The only place
STATION_TZ is used is the daily-max resolution boundary in wxio.build_daily().
"""

import argparse
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path


# --- Stations ----------------------------------------------------------------

@dataclass(frozen=True)
class Station:
    """One target station. Coordinates are from the IEM network metadata."""
    icao: str
    iem_id: str          # Iowa Environmental Mesonet id, and the NBM bulletin id
    latitude: float
    longitude: float
    tz: str
    iem_network: str
    note: str = ""
    # Whether the blend's spread scalar is applied by default. It is always
    # fitted on this station's own data; this only says whether applying it
    # improves calibration here, which is a measured property of the station.
    inflate: bool = True


STATIONS: dict[str, Station] = {
    "KNYC": Station("KNYC", "NYC", 40.7790, -73.9693, "America/New_York",
                    "NY_ASOS", "Central Park; the original target"),
    # KLGA: the scalar overshoots here. Uninflated PIT is 0.95; inflating moves
    # it to 1.06, further from 1.00, for +0.005 CRPS. KNYC goes 0.93 -> 1.02 and
    # keeps it. Same code, opposite verdict, so the choice belongs to the station.
    "KLGA": Station("KLGA", "LGA", 40.7794, -73.8803, "America/New_York",
                    "NY_ASOS", "LaGuardia; what the Polymarket NYC market settles on",
                    inflate=False),
}

# Which station the modules act on when nothing says otherwise. Every entry
# point takes --station, and WEATHERBOT_STATION overrides this default, so the
# station is a parameter rather than a constant baked into the code.
DEFAULT_STATION = os.environ.get("WEATHERBOT_STATION", "KNYC")

BASE_DATA_DIR = Path(__file__).parent / "data"

# Mutable view of the active station. use_station() rebinds these; modules read
# them at call time rather than capturing them at import.
STATION_ICAO = ""
STATION_IEM_ID = ""
LATITUDE = 0.0
LONGITUDE = 0.0
STATION_TZ = ""
IEM_NETWORK = ""
DATA_DIR = BASE_DATA_DIR
RAW_HOURLY = BASE_DATA_DIR / "raw_hourly.parquet"
DAILY = BASE_DATA_DIR / "daily.parquet"


def use_station(name: str) -> Station:
    """Point every module at `name`. Caches are per station, so switching is safe."""
    global STATION_ICAO, STATION_IEM_ID, LATITUDE, LONGITUDE, STATION_TZ
    global IEM_NETWORK, DATA_DIR, RAW_HOURLY, DAILY

    key = name.upper()
    if key not in STATIONS:
        raise SystemExit(f"unknown station {name!r}; known: {', '.join(STATIONS)}")
    s = STATIONS[key]
    STATION_ICAO, STATION_IEM_ID = s.icao, s.iem_id
    LATITUDE, LONGITUDE = s.latitude, s.longitude
    STATION_TZ, IEM_NETWORK = s.tz, s.iem_network
    DATA_DIR = BASE_DATA_DIR / s.icao
    RAW_HOURLY = DATA_DIR / "raw_hourly.parquet"
    DAILY = DATA_DIR / "daily.parquet"
    return s


def add_station_arg(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--station", default=DEFAULT_STATION,
                    choices=sorted(STATIONS),
                    help=f"target station (default {DEFAULT_STATION})")
    return ap

# --- Backfill ranges ---------------------------------------------------------
# Observations are cheap and carry no provenance problem, so they start earlier
# than the forecast backbone to give climatology/persistence baselines a runway.
OBS_START = date(2023, 1, 1)

# Deterministic backbone: ECMWF IFS 9 km via the Single Runs API. This is the
# deepest source with a server-validated issue_time. Probed boundary: the run at
# 2024-03-13T00:00 returns HTTP 400, 2024-03-14T00:00 returns HTTP 200.
BACKBONE_START = date(2024, 3, 14)
BACKBONE_MODEL = "ecmwf_ifs"
# Only 00Z and 12Z are archived for this model; 06Z/18Z return "not available".
# One run per day keeps exactly one issue_time per calendar day.
BACKBONE_RUN_HOURS = (0,)

# Ensemble: 51 members (50 perturbed + control). The Open-Meteo ensemble archive
# is ~4 days deep regardless of the date range the validator accepts, so this is
# pulled live and accumulates forward from today.
ENSEMBLE_MODEL = "ecmwf_ifs025"
ENSEMBLE_PAST_DAYS = 4
ENSEMBLE_FORECAST_DAYS = 7

VARIABLE = "temperature_2m"

# Predictors pulled per run, in one request each. temperature_850hPa is NOT
# here: ecmwf_ifs accepts the parameter but returns all nulls for it, and the
# models that do serve it (GFS, ICON) only have Single Runs archives from
# 2026-04-02, which would cut training from ~874 rows to ~130.
FORECAST_VARIABLES = [
    "temperature_2m",
    "cloud_cover",
    "wind_speed_10m",
    "dew_point_2m",
]

# --- Endpoints ---------------------------------------------------------------
SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
IEM_DAILY_URL = "https://mesonet.agron.iastate.edu/api/1/daily.json"

# --- Cache (hard rule 5: every pull lands on disk as parquet) ----------------
# A local day needs at least this many hourly values before a daily max is
# trusted. Short days are dropped and counted, never filled (hard rule 7).
MIN_HOURS_PER_DAY = 20

# --- HTTP --------------------------------------------------------------------
# Some individual archive runs hang server-side instead of returning an error
# (e.g. ecmwf_ifs 2024-03-17T00:00). A short timeout keeps one bad run from
# stalling a long backfill.
HTTP_TIMEOUT = 25
HTTP_RETRIES = 3
HTTP_BACKOFF = 2.0
REQUEST_SPACING = 0.15
# Archive reads are slow and mostly spent waiting, so the backbone backfill
# fetches concurrently. Kept modest to stay well inside Open-Meteo's limits.
MAX_WORKERS = 8

# Abort the backfill if more than this fraction of runs cannot be retrieved --
# isolated bad runs are tolerated and reported, a systemic outage is not.
MAX_UNREACHABLE_FRACTION = 0.05
# ...but only once enough runs were attempted for that fraction to mean
# anything. On an incremental pull of a few days, one timeout is already 25%.
MIN_RUNS_FOR_OUTAGE_CHECK = 20

# Bind the default so importing config alone yields a usable configuration.
use_station(DEFAULT_STATION)
