"""Single entry point for unattended operation.

One pass: refresh forecasts, build the predictive distribution, snapshot open
Polymarket prices against it, and backfill outcomes for days that have since
been observed.

Logging only. Nothing here places, sizes or recommends a trade, and nothing
reads an API key, because none of the sources used require one.

    python3 run_daily.py [--data-dir DIR] [--log-dir DIR] [--force]

Exit codes
    0  every stage either succeeded or was a documented no-op
    1  at least one stage failed
    2  the run was skipped because the last snapshot is too recent

Design notes for unattended use
-------------------------------
A snapshot newer than config.MIN_RUN_INTERVAL_MIN makes the run exit 2 without
touching anything, so a double-fire or two overlapping cron invocations cannot
double-log. It is a floor on the gap between snapshots, not a cap on cadence:
keep it below the cron interval or it throttles the schedule. --force overrides.

Partial failure is expected, not exceptional. Upstream archives lag the live
market: the Open-Meteo Single Runs API has not published a 00Z run an hour after
it was issued, and NBM bulletins land a few hours late, so "today is not ready
yet" is normal and is reported as a no-op rather than an error. A stage that
genuinely fails is recorded, the remaining stages still run, and the process
exits non-zero at the end.

One bad market cannot kill the run: bucket parsing and pricing are isolated per
market inside clv.market_rows, and a failure there is counted and skipped.

Configuration is environment-only. Nothing is read from a file that is not in
the repository, and no secret is required; see DEPLOY.md.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

# Environment overrides must be applied before config is imported by anything
# else, since config resolves its paths at import time.
if os.environ.get("WEATHERBOT_DATA_DIR"):
    os.environ.setdefault("_WEATHERBOT_DATA_DIR_APPLIED", "1")

import config  # noqa: E402

LOG = logging.getLogger("run_daily")


def setup_logging(log_dir: Path) -> Path:
    """Append to a per-day file and echo to stdout, both timestamped in UTC."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"run_daily-{datetime.now(timezone.utc):%Y-%m-%d}.log"

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    fmt.converter = lambda *a: datetime.now(timezone.utc).timetuple()

    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    LOG.addHandler(fh)
    LOG.addHandler(sh)
    return path


class Stage:
    """Run one stage, record the outcome, never propagate.

    `tolerated` names exceptions that mean "upstream is not ready", which is a
    normal state for this pipeline rather than a fault.
    """

    def __init__(self) -> None:
        self.failed: list[str] = []
        self.noop: list[str] = []

    def run(self, name: str, fn, tolerated: tuple[type, ...] = ()) -> object:
        LOG.info("=== %s", name)
        try:
            result = fn()
            LOG.info("    ok")
            return result
        except tolerated as exc:
            LOG.info("    no-op: %s", str(exc).splitlines()[0][:160])
            self.noop.append(name)
            return None
        except Exception as exc:  # noqa: BLE001 - deliberate: isolate stages
            LOG.error("    FAILED: %s: %s", type(exc).__name__,
                      str(exc).splitlines()[0][:200])
            LOG.debug("%s", traceback.format_exc())
            self.failed.append(name)
            return None


def minutes_since_last_snapshot() -> float | None:
    """Age of the newest price snapshot on disk, or None if there is none."""
    import pandas as pd
    from clv import LOG_PATH

    if not LOG_PATH.exists():
        return None
    log = pd.read_parquet(LOG_PATH, columns=["logged_at"])
    if log.empty:
        return None
    last = pd.Timestamp(log["logged_at"].max())
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    return (pd.Timestamp(datetime.now(timezone.utc)) - last).total_seconds() / 60.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", type=Path,
                    default=(Path(os.environ["WEATHERBOT_DATA_DIR"])
                             if os.environ.get("WEATHERBOT_DATA_DIR") else None),
                    help="root of the per-station cache layout (data/<ICAO>/)")
    ap.add_argument("--log-dir", type=Path,
                    default=Path(os.environ.get("WEATHERBOT_LOG_DIR", REPO / "logs")))
    ap.add_argument("--force", action="store_true",
                    help="run even if this hour already has a snapshot")
    args = ap.parse_args()

    # Override the root of the per-station cache layout, not a single station's
    # directory: config derives data/<ICAO>/ from this.
    if args.data_dir is not None:
        config.BASE_DATA_DIR = args.data_dir

    log_path = setup_logging(args.log_dir)
    started = datetime.now(timezone.utc)

    # Imported after config paths are set, since these resolve them at import.
    import clv  # noqa: E402
    import wxio  # noqa: E402

    # The market settles on KLGA, so that -- not the project's original KNYC --
    # is the station this job forecasts. Name it plainly: an earlier header said
    # "station=KNYC" above a KLGA forecast line, which read like a bug.
    market = config.use_station(clv.MARKET_STATION_ICAO)
    clv.PRED_PATH = config.DATA_DIR / "clv_pred.parquet"
    clv.LOG_PATH = config.DATA_DIR / "clv_log.parquet"
    LOG.info("run_daily start  market station=%s (%s)  data=%s  log=%s",
             market.icao, market.note, config.DATA_DIR, log_path)

    age = None if args.force else minutes_since_last_snapshot()
    if age is not None and age < config.MIN_RUN_INTERVAL_MIN:
        LOG.info("last snapshot was %.1f min ago, under the %.0f min minimum; "
                 "nothing to do", age, config.MIN_RUN_INTERVAL_MIN)
        LOG.info("run_daily skipped (use --force to override)")
        return 2

    st = Stage()
    today = date.today()
    tolerated = (wxio.SourceError,)

    # 1. Forecast inputs. Both lag the live market, so a miss here is normal and
    #    the next run backfills.
    def pull_ecmwf():
        import ingest_forecast
        sys.argv = ["ingest_forecast", "--station", market.icao, "--skip-ensemble",
                    "--end", today.isoformat(), "--workers", "3", "--timeout", "90"]
        ingest_forecast.main()

    def pull_nbm():
        import ingest_nbm
        sys.argv = ["ingest_nbm", "--station", market.icao,
                    "--start", (today - timedelta(days=2)).isoformat(),
                    "--end", today.isoformat()]
        ingest_nbm.main()

    st.run("pull ECMWF 00Z runs", pull_ecmwf, tolerated)
    st.run("pull NBM bulletins", pull_nbm, tolerated)

    # 2. Blend forecast -> KLGA predictive distribution, cached for the day.
    def predict():
        ns = argparse.Namespace(date=None)
        clv.cmd_predict(ns)

    st.run("build predictive distribution", predict, tolerated)

    # 3. Price snapshot. Per-market failures are isolated inside clv.
    def snapshot():
        clv.cmd_log(argparse.Namespace())

    st.run("snapshot Polymarket prices", snapshot, tolerated)

    # 4. Outcomes for anything observed since the last run.
    def backfill():
        clv.cmd_resolve(argparse.Namespace())

    st.run("backfill resolved outcomes", backfill, tolerated)

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    LOG.info("run_daily done in %.0fs  |  %d ok, %d no-op, %d failed",
             elapsed, 5 - len(st.failed) - len(st.noop), len(st.noop), len(st.failed))
    if st.noop:
        LOG.info("no-op stages (upstream not ready): %s", ", ".join(st.noop))
    if st.failed:
        LOG.error("failed stages: %s", ", ".join(st.failed))
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        # Nothing above should escape, but an unattended job must never exit 0
        # on an unexpected error.
        traceback.print_exc()
        sys.exit(1)
