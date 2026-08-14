#!/usr/bin/env bash
# Rebuild every cache from scratch, in dependency order, and verify the result.
#
# Safe to re-run: each ingestion module skips what is already cached (hard rule
# 5), so an interrupted rebuild resumes rather than re-downloading. Any step
# that fails stops the script (hard rule 7) -- a partial cache is never left
# looking complete.
#
# Usage:
#   ./make_cache.sh              # rebuild / resume, then verify
#   ./make_cache.sh --fresh      # delete data/ first, then rebuild
#
set -euo pipefail

cd "$(dirname "$0")"

FRESH=0
[[ "${1:-}" == "--fresh" ]] && FRESH=1

started=$(date +%s)
step_start=0

step() {
  step_start=$(date +%s)
  printf '\n\033[1m==> %s\033[0m\n' "$1"
}

done_step() {
  printf '    (%s)\n' "$(fmt $(( $(date +%s) - step_start )))"
}

fmt() {
  local s=$1
  if (( s < 60 )); then printf '%ds' "$s"; else printf '%dm %02ds' $(( s / 60 )) $(( s % 60 )); fi
}

if (( FRESH )); then
  step "Removing existing caches"
  rm -rf data
  done_step
fi

# 1. Observations first: the forecast modules rebuild daily.parquet, which
#    carries the target variable, so the target should already be present.
step "Observations (ASOS hourly + reported daily max, 2023-01-01 onward)"
python3 ingest_asos.py
done_step

# 2. Deterministic backbone. Individual archive reads are slow and a few runs
#    hang server-side, so a first concurrent pass is followed by a slower,
#    more patient pass that retries exactly the runs that failed. The second
#    pass is allowed to fail without killing the script only in the sense that
#    stragglers are reported -- a systemic outage still aborts inside the module.
step "ECMWF backbone (single runs, 2024-03-14 onward) -- the long one"
python3 ingest_forecast.py --skip-ensemble
done_step

step "ECMWF backbone -- retry stragglers at lower concurrency"
python3 ingest_forecast.py --skip-ensemble --workers 3 --timeout 90
done_step

# 3. Ensemble is live-only (~4 day archive), so this captures today's window.
#    It accumulates across days; a single run gives one pull date.
step "ECMWF ensemble (live window only, ~4 days deep)"
python3 ingest_forecast.py --skip-backbone
done_step

# 4. NBM station bulletins from NOAA's S3 archive.
step "NBM day-ahead max + spread (station bulletins)"
python3 ingest_nbm.py
done_step

step "Verifying caches against the hard rules"
python3 verify_data.py
done_step

printf '\n\033[1mCache rebuild complete in %s\033[0m\n' "$(fmt $(( $(date +%s) - started )))"
du -sh data/ 2>/dev/null || true
