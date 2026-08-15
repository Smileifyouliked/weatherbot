#!/usr/bin/env bash
# One CLV polling pass: refresh inputs, snapshot market prices, resolve, persist.
#
# Designed to be run on a schedule several times a day. Everything it does is
# idempotent, so a missed or repeated run is harmless.
#
# The CLV log and cached prediction under data/ are runtime output, not
# repository content, so nothing here commits or pushes. On a persistent host
# the data directory is what needs backing up; see DEPLOY.md.
#
# For unattended operation prefer run_daily.py, which adds per-market error
# isolation, a timestamped log file and a meaningful exit code. This script is
# the thin manual equivalent.
#
# Usage: ./clv_poll.sh
set -uo pipefail
cd "$(dirname "$0")"

log() { printf '\n[%s] %s\n' "$(date -u +%H:%M:%S)" "$1"; }

# 1. Bring forecasts up to today. The Single Runs archive lags real time by up
#    to ~13 hours, so the run for today's target often is not retrievable until
#    mid-morning local. Failures here are expected and must not stop the poll.
log "ingest (best effort)"
python3 ingest_forecast.py --skip-ensemble --end "$(date -u +%F)" \
        --workers 3 --timeout 90 >/dev/null 2>&1 || echo "  ecmwf: nothing new"
python3 ingest_nbm.py --start "$(date -u -d '2 days ago' +%F)" \
        --end "$(date -u +%F)" >/dev/null 2>&1 || echo "  nbm: nothing new"

# 2. Recompute the predictive distribution only if today's is missing. This is
#    the slow step; the price snapshot below is cheap and is the part that
#    benefits from frequent polling.
log "predict"
python3 - <<'PY' || python3 src/clv.py predict
import sys, pandas as pd, pathlib
sys.path.insert(0, ".")
import config
p = config.DATA_DIR / "clv_pred.parquet"
if not p.exists():
    raise SystemExit(1)
have = set(pd.read_parquet(p)["target_date"].dt.date.astype(str))
today = pd.Timestamp.utcnow().date().isoformat()
raise SystemExit(0 if today in have else 1)
PY

log "log prices"
python3 src/clv.py log || echo "  log: nothing to snapshot"

log "resolve"
python3 src/clv.py resolve || echo "  resolve: nothing to resolve"

log "done"
