"""Station, source and cache configuration.

All times in this project are UTC internally (hard rule 6). The only place
STATION_TZ is used is the daily-max resolution boundary in wxio.build_daily().
"""

from datetime import date
from pathlib import Path

# --- Station -----------------------------------------------------------------
# KNYC = New York Central Park. "NYC" is the Iowa Environmental Mesonet ASOS id.
STATION_ICAO = "KNYC"
STATION_IEM_ID = "NYC"
LATITUDE = 40.7790
LONGITUDE = -73.9693
STATION_TZ = "America/New_York"

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

# --- Endpoints ---------------------------------------------------------------
SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
IEM_DAILY_URL = "https://mesonet.agron.iastate.edu/api/1/daily.json"
IEM_NETWORK = "NY_ASOS"

# --- Cache (hard rule 5: every pull lands on disk as parquet) ----------------
DATA_DIR = Path(__file__).parent / "data"
RAW_HOURLY = DATA_DIR / "raw_hourly.parquet"
DAILY = DATA_DIR / "daily.parquet"

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
