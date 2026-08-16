"""Log Polymarket daily-temperature prices against our forecast, and score them.

Logging and scoring only. Nothing here places, sizes or recommends a trade.

    python3 src/clv.py predict     # cache today's predictive distribution (slow, once/day)
    python3 src/clv.py log         # snapshot market prices + our probabilities (fast, poll often)
    python3 src/clv.py resolve     # attach outcomes once the day is observed
    python3 src/clv.py score       # Brier, CLV, side-of-market, by distance from our mean

WHICH STATION
-------------
Polymarket's "NYC" temperature market resolves on **KLGA (LaGuardia)** via
Weather Underground, not KNYC (Central Park). KLGA minus KNYC is +0.55 F on
average with a standard deviation of 1.78 F, and +1.92 F in summer -- about as
large as the predictive uncertainty itself, so the two are not interchangeable.

This module therefore runs the **native KLGA** stack: its own ECMWF backbone
interpolated to LaGuardia, its own NBM bulletin, its own observations, blended
the same way as any other station. It does not touch the KNYC cache.

Native replaced an earlier KNYC-blend-plus-adaptation route, which reached 1.42
CRPS on the settled quantity against 1.33 for native over the same 364 days.

WHEN A SNAPSHOT IS WORTH TAKING
-------------------------------
Prices are only logged for a target day while that day's maximum is still
genuinely uncertain, i.e. before that season's cutoff in PEAK_LOCAL_HOUR_BY_SEASON
(DJF 15:00, SON 15:30, MAM and JJA 16:00) local time. Past that the
market has effectively seen the answer and quotes ~0 or ~1; recording those
prices measures nothing and contaminates the CLV comparison with
trivially-known outcomes. Rows captured before the guard existed are kept as a
record but flagged `excluded`, never deleted.

The binding constraint at the other end is publication lag. It is being measured
directly (tools/measure_run_lag.py); so far one 00Z run has been caught while
pending, appearing 7.46 h after issue, i.e. 07:27Z. The other two rows are upper
bounds only -- the run was already there when watching began. NBM's 00Z bulletin
is quicker, present by about 03:00Z. So the useful polling window for a target
local day runs from roughly 07:30Z to 20:00Z, the seasonal cutoff in UTC. The
cron window is deliberately not fixed on a single sample.

MARKET STRUCTURE
----------------
The event is not a single threshold. It is 11 mutually exclusive range buckets
("75F or below", "between 76-77F", ..., "94F or higher"), each its own binary
market. Our probability for a bucket is Phi(hi) - Phi(lo) under the predictive
Gaussian, with the bucket edges taken at the rounding boundary: a reported
integer of 76 or 77 lands in "76-77", so that bucket covers [75.5, 77.5).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import wxio  # noqa: E402
from baselines import _Phi, load as load_daily  # noqa: E402
from mos import (  # noqa: E402
    FEATURES_MEAN, LEAD, build_features, load_nbm, select_features, walk_forward,
)
from blend import (  # noqa: E402
    DEFAULT_MODE, NBM_FEATURES, NBM_SIGMA, blend_walk_forward, default_inflate,
)

GAMMA = "https://gamma-api.polymarket.com"
EVENT_SLUG = "highest-temperature-in-nyc-on-{month}-{day}-{year}"
MARKET_STATION = "LGA"           # IEM id, for observations
MARKET_STATION_ICAO = "KLGA"     # what Polymarket actually settles on
MARKET_NETWORK = "NY_ASOS"

# CLV state belongs to the market's station, not whatever was last selected.
config.use_station(MARKET_STATION_ICAO)
PRED_PATH = config.DATA_DIR / "clv_pred.parquet"
LOG_PATH = config.DATA_DIR / "clv_log.parquet"

ADAPT_FEATURES = ["blend_mu", "sin_doy", "cos_doy"]
ADAPT_SIGMA = ["log_blend_sigma", "sin_doy", "cos_doy"]
MIN_ADAPT_TRAIN = 180
HOLDOUT_FRAC = 0.25

# Distance of a bucket centre from our predictive mean, in units of our sigma.
DISTANCE_BINS = [0.0, 0.5, 1.0, 2.0, np.inf]
DISTANCE_LABELS = ["<0.5s (centre)", "0.5-1s", "1-2s", ">2s (tail)"]

# Minimum edge over the *quoted* price before the P&L simulation takes a
# position. Below the typical spread on these buckets there is nothing to take.
DEFAULT_EDGE = 0.05

# Resolved days required before a P&L figure is printed at all. The buckets
# within a day resolve together, so the effective sample is the day count, not
# the trade count -- and a P&L number reads as a result in a way a Brier score
# does not, so it is withheld rather than merely caveated. Stricter than the
# 30-day note on the Brier/CLV output, deliberately.
MIN_PNL_DAYS = 100

# Local hour by which the daily maximum has usually happened, per season. Past
# it a snapshot is scoring a near-known outcome, which flatters the market and
# tells us nothing about forecast skill; such rows are logged but excluded from
# scoring by default.
#
# A single 13:00 cutoff was too early. Scored the way the logger actually runs --
# a snapshot every 30 minutes from the moment the forecast lands, over 2,783
# complete KLGA days -- moving to these seasonal cutoffs yields 57,710 clean rows
# against 50,224, +14.9%, while leakage (rows timestamped after that day's
# realised max) goes 12.8% -> 20.1%. Leaked rows are flagged `excluded` and never
# scored, so the trade is more usable data against more dead storage.
# Run `python3 src/clv.py cutoffs` to reproduce it from this station's history.
PEAK_LOCAL_HOUR_BY_SEASON = {"DJF": 15.0, "MAM": 16.0, "JJA": 16.0, "SON": 15.5}
SEASON_OF_MONTH = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                   6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}

# Winter is bimodal -- about 10% of DJF days peak just after midnight on a
# frontal passage -- so the clock alone cannot say whether the day is settled.
# The same-day guard lives in config.DETERMINED_QUANTILE, read at call time so a
# tuned value takes effect without touching this module.

# The polling schedule `cutoffs` scores the cutoffs against. Snapshots are taken
# every POLL_INTERVAL_MIN minutes from the moment that day's 00Z ECMWF run can
# be retrieved until the seasonal cutoff, so what a cutoff actually costs is
# measured in rows, not in days.
#
# The open time is the measured publication lag of the 00Z run (see
# tools/measure_run_lag.py). It rests on one clean measurement so far -- the
# 2026-08-15 00Z run appeared 7.46 h after issue -- so it is an assumption the
# command prints and a flag the caller can move, not a schedule to cron against.
POLL_INTERVAL_MIN = 30
WINDOW_OPEN_HOURS_AFTER_00Z = 7.46


def peak_hour_for(target: date) -> float:
    return PEAK_LOCAL_HOUR_BY_SEASON[SEASON_OF_MONTH[target.month]]


def cutoff_summary() -> str:
    """The seasonal cutoffs as one short string, for messages."""
    return ", ".join(f"{s} {int(h)}:{round(h % 1 * 60):02d}"
                     for s, h in PEAK_LOCAL_HOUR_BY_SEASON.items())


def c_to_f(c):
    return np.asarray(c) * 9.0 / 5.0 + 32.0


def hours_before_peak(stamp: pd.Timestamp, target: date) -> float:
    """Hours from `stamp` until the target local day's likely maximum.

    Negative once that moment has passed, which is when a price snapshot stops
    carrying information about an uncertain outcome.
    """
    local = stamp.tz_convert(config.STATION_TZ)
    peak = (pd.Timestamp(target, tz=config.STATION_TZ)
            + pd.Timedelta(hours=peak_hour_for(target)))
    return float((peak - local).total_seconds() / 3600.0)


def realized_max_so_far(target: date) -> float | None:
    """Highest temperature reported at the station so far on `target`, in F.

    Returns None when nothing has been reported yet, or the fetch fails -- an
    absent reading must never be read as "cool so far".
    """
    config.use_station(MARKET_STATION_ICAO)
    try:
        r = requests.get(config.IEM_ASOS_URL, params={
            "station": config.STATION_IEM_ID, "data": "tmpf",
            "year1": target.year, "month1": target.month, "day1": target.day,
            "year2": target.year, "month2": target.month, "day2": target.day,
            "tz": config.STATION_TZ, "format": "onlycomma", "latlon": "no",
            "missing": "empty", "trace": "empty", "report_type": 3,
        }, timeout=config.HTTP_TIMEOUT)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    import io
    df = pd.read_csv(io.StringIO(r.text))
    if df.empty or "tmpf" not in df.columns:
        return None
    v = pd.to_numeric(df["tmpf"], errors="coerce").dropna()
    return float(v.max()) if len(v) else None


def already_determined(target: date, mu: float, sigma: float) -> tuple[bool, str]:
    """Whether today's max is already settled regardless of the clock.

    The seasonal cutoff assumes a single afternoon peak. Winter breaks that:
    roughly 10% of DJF days peak just after midnight on a frontal passage, and
    by 09:00 the answer is known while the clock says hours remain. If what has
    already been reported exceeds a high quantile of our own forecast, the
    remaining uncertainty is small enough to stop.

    The quantile is read from config on every call, not captured at import, so
    tuning it takes effect everywhere at once.
    """
    from statistics import NormalDist

    q = config.DETERMINED_QUANTILE
    threshold = mu + NormalDist().inv_cdf(q) * sigma
    seen = realized_max_so_far(target)
    if seen is None:
        return False, ""
    if seen >= threshold:
        return True, (f"realised max {seen:.0f}F already at/above our "
                      f"p{q * 100:.0f} of {threshold:.1f}F")
    return False, ""


def apply_exclusions(log: pd.DataFrame) -> pd.DataFrame:
    """Mark rows that must not be scored, without deleting them.

    Rows logged at or after the target day's likely maximum are kept as a record
    of what was quoted, but flagged so scoring never counts them. Applied on
    load so logs written before the guard existed are handled too.
    """
    if "excluded" not in log.columns:
        log["excluded"] = False
        log["exclude_reason"] = ""
    log["excluded"] = log["excluded"].fillna(False).astype(bool)
    log["exclude_reason"] = log["exclude_reason"].fillna("").astype(str)

    if "hours_before_peak" in log.columns:
        late = log["hours_before_peak"].fillna(-1.0) <= 0
        newly = late & ~log["excluded"]
        log.loc[newly, "excluded"] = True
        log.loc[newly, "exclude_reason"] = "logged after the season's peak hour"
    return log


def utcnow() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


# --- market ------------------------------------------------------------------

BUCKET_RE = [
    (re.compile(r"be\s+(-?\d+)\s*°?F\s+or\s+below", re.I),
     lambda m: (-np.inf, float(m.group(1)) + 0.5)),
    (re.compile(r"be\s+(-?\d+)\s*°?F\s+or\s+higher", re.I),
     lambda m: (float(m.group(1)) - 0.5, np.inf)),
    (re.compile(r"between\s+(-?\d+)\s*-\s*(-?\d+)\s*°?F", re.I),
     lambda m: (float(m.group(1)) - 0.5, float(m.group(2)) + 0.5)),
]


def parse_bucket(question: str) -> tuple[float, float] | None:
    """Bucket edges in degrees F, at the rounding boundary of reported integers."""
    for rx, fn in BUCKET_RE:
        m = rx.search(question)
        if m:
            return fn(m)
    return None


def fetch_event(target: date) -> dict | None:
    slug = EVENT_SLUG.format(month=target.strftime("%B").lower(),
                             day=target.day, year=target.year)
    r = requests.get(f"{GAMMA}/events", params={"slug": slug},
                     timeout=config.HTTP_TIMEOUT)
    if r.status_code != 200:
        raise wxio.SourceError(f"gamma /events {slug}: HTTP {r.status_code}")
    payload = r.json()
    if not payload:
        return None
    return payload[0] if isinstance(payload, list) else payload


def market_rows(event: dict, target: date) -> tuple[list[dict], list[str]]:
    """One row per bucket, with best bid/ask and a mid where both sides exist.

    Each market is parsed independently: one malformed or unexpected entry is
    skipped and reported, never allowed to lose the rest of the event. Returns
    (rows, failures).
    """
    out, failures = [], []
    for m in event.get("markets", []):
        q = m.get("question") or ""
        try:
            bounds = parse_bucket(q)
            if bounds is None:
                failures.append(f"unparseable bucket: {q[:70]!r}")
                continue
            bid = m.get("bestBid")
            ask = m.get("bestAsk")
            bid = float(bid) if bid is not None else np.nan
            ask = float(ask) if ask is not None else np.nan
            # Mid only where both sides are quoted; a one-sided book has no mid,
            # and substituting last-trade would silently mix two quantities.
            mid = (bid + ask) / 2.0 if np.isfinite(bid) and np.isfinite(ask) else np.nan
            last = m.get("lastTradePrice")
            out.append({
                "target_date": pd.Timestamp(target),
                "market_id": str(m.get("id")),
                "question": q,
                "lo_f": bounds[0], "hi_f": bounds[1],
                "bid": bid, "ask": ask, "mid": mid,
                "last": float(last) if last is not None else np.nan,
            })
        except (TypeError, ValueError, KeyError) as exc:
            failures.append(f"{q[:50]!r}: {type(exc).__name__}: {exc}")
    return out, failures


# --- prediction --------------------------------------------------------------

def klga_obs() -> pd.Series:
    """Reported KLGA daily maxima in F -- the quantity the market settles on."""
    config.use_station(MARKET_STATION_ICAO)
    r = requests.get(config.IEM_DAILY_URL,
                     params={"station": MARKET_STATION, "network": MARKET_NETWORK,
                             "sdate": config.OBS_START.isoformat(),
                             "edate": date.today().isoformat()},
                     timeout=config.HTTP_TIMEOUT)
    if r.status_code != 200:
        raise wxio.SourceError(f"IEM daily {MARKET_STATION}: HTTP {r.status_code}")
    df = pd.DataFrame(r.json()["data"])
    df["local_date"] = pd.to_datetime(df["date"])
    df = df[~df["tmpf_est"].fillna(False).astype(bool)].dropna(subset=["max_tmpf"])
    return df.set_index("local_date")["max_tmpf"].astype(float).sort_index()


def components(station: str) -> pd.DataFrame:
    """Per-day MOS and NBM-PP predictions for `station`, from its own cache.

    Left joins on the target so a day whose observation does not exist yet -- the
    live case -- survives and can still be predicted.
    """
    config.use_station(station)
    obs, _ = load_daily(LEAD)
    ec = build_features().join(obs.rename("obs"), how="left").dropna(subset=FEATURES_MEAN)
    nbm = nbm_frame_live(obs)
    feats = select_features(ec.dropna(subset=["obs"]), ec.index[int(len(ec) * 0.6)])

    mos = walk_forward(ec, ec.index, feats, "rolling")
    npp = walk_forward(nbm, nbm.index, NBM_FEATURES, "rolling",
                       sigma_features=NBM_SIGMA)
    comp = pd.DataFrame({"obs": mos["obs"], "mu_mos": mos["mu"],
                         "sigma_mos": mos["sigma"]}).join(
        pd.DataFrame({"mu_nbm": npp["mu"], "sigma_nbm": npp["sigma"]}), how="inner")

    iv_var = 1.0 / (1.0 / comp["sigma_mos"] ** 2 + 1.0 / comp["sigma_nbm"] ** 2)
    comp["log_ivsigma"] = np.log(np.sqrt(iv_var))
    doy = comp.index.dayofyear.values
    comp["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    comp["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    return comp


def nbm_frame_live(obs: pd.Series) -> pd.DataFrame:
    """nbm_frame, but keeping days whose observation is not in yet."""
    n = load_nbm()
    if n.empty:
        raise wxio.SourceError("no NBM rows cached; run ingest_nbm.py first")
    df = pd.DataFrame({"nbm_tmax": n["mu"], "xnd": n["sigma"]}, index=n.index)
    df = df.join(obs.rename("obs"), how="left").dropna(subset=["nbm_tmax", "xnd"])
    doy = df.index.dayofyear.values
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    df["log_xnd"] = np.log(df["xnd"].clip(lower=0.5))
    return df.sort_index()


def predict_market(target: date) -> tuple[float, float, int]:
    """Predictive mean and spread for the market's own station, natively.

    Until a native KLGA build existed this went through the KNYC blend plus a
    statistical adaptation. Native is better on the settled quantity -- 1.33 CRPS
    against 1.42 for the adaptation over the same 364 days -- because the
    forecast fields are interpolated to LaGuardia rather than Central Park.
    """
    comp = components(MARKET_STATION_ICAO)
    D = pd.Timestamp(target)
    if D not in comp.index:
        raise wxio.SourceError(
            f"no {MARKET_STATION_ICAO} forecast for {target}; the backbone run for "
            f"that day may not be cached yet (run ingest_forecast.py)")

    out = blend_walk_forward(comp, pd.DatetimeIndex([D]), DEFAULT_MODE,
                             inflate=default_inflate())
    if out.empty:
        raise wxio.SourceError(f"not enough training history before {target}")
    row = out.iloc[0]
    return float(row["mu"]), float(row["sigma"]), int(row["n_train"])


def cmd_predict(args) -> None:
    if args.date:
        targets = [args.date]
    else:
        # Yesterday is included because the archive lags: the 00Z run for a
        # local day is typically not retrievable until hours after it was
        # issued, so a scheduled run backfills as soon as the data lands.
        targets = [date.today() + timedelta(days=d) for d in (-1, 0, 1)]

    rows = []
    for t in targets:
        try:
            mu, sigma, n = predict_market(t)
        except wxio.SourceError as exc:
            print(f"  {t}: not yet available -- {str(exc)[:90]}")
            continue
        rows.append({"target_date": pd.Timestamp(t), "mu_f": mu, "sigma_f": sigma,
                     "n_train": n, "computed_at": utcnow()})
        print(f"  {t}: {MARKET_STATION_ICAO} mu {mu:.1f} F, sigma {sigma:.2f} F  "
              f"(n_train={n}, native)")

    if not rows:
        # A scheduled run that finds nothing new is normal, not a failure.
        print("  nothing predictable yet; upstream runs have not been archived")
        return

    new = pd.DataFrame(rows)
    if PRED_PATH.exists():
        old = pd.read_parquet(PRED_PATH)
        new = pd.concat([old, new], ignore_index=True)
        new = new.drop_duplicates(subset=["target_date"], keep="last")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    new.sort_values("target_date").to_parquet(PRED_PATH, index=False)
    print(f"  cached -> {PRED_PATH}")


# --- logging -----------------------------------------------------------------

def cmd_log(args) -> None:
    if not PRED_PATH.exists():
        raise wxio.SourceError("no cached prediction; run 'predict' first")
    pred = pd.read_parquet(PRED_PATH).set_index("target_date")

    stamp = utcnow()
    rows = []
    # Driven by what we can actually price: every cached prediction whose market
    # is still open. That absorbs the archive lag automatically.
    horizon = pd.Timestamp(date.today() - timedelta(days=2))
    for ts in pred.index[pred.index >= horizon]:
        t = ts.date()
        event = fetch_event(t)
        if event is None:
            print(f"  {t}: no event")
            continue
        if event.get("closed"):
            print(f"  {t}: event closed, skipped")
            continue

        # Only snapshot while the outcome is still genuinely uncertain. Past the
        # day's likely maximum the market has effectively seen the answer and
        # prices sit at ~0 or ~1; logging that records no information and
        # contaminates the CLV comparison with trivially-known outcomes.
        hours_left = hours_before_peak(stamp, t)
        cutoff = peak_hour_for(t)
        if hours_left <= 0:
            print(f"  {t}: past {cutoff:.1f} local "
                  f"({-hours_left:.1f}h ago) -- max likely determined, not logged")
            continue

        mu = float(pred.loc[ts, "mu_f"])
        sigma = float(pred.loc[ts, "sigma_f"])

        determined, why = already_determined(t, mu, sigma)
        if determined:
            print(f"  {t}: {why} -- day already settled, not logged")
            continue
        mkts, failures = market_rows(event, t)
        for why in failures:
            print(f"  {t}: skipped a market -- {why}")
        for r in mkts:
            p = float(_Phi((r["hi_f"] - mu) / sigma) - _Phi((r["lo_f"] - mu) / sigma))
            # Distance from our mean, in sigmas. The end buckets are open on one
            # side, so their midpoint is undefined; the finite edge is what
            # actually determines how far into the tail the bucket sits.
            if not np.isfinite(r["lo_f"]):
                ref = r["hi_f"]
            elif not np.isfinite(r["hi_f"]):
                ref = r["lo_f"]
            else:
                ref = (r["lo_f"] + r["hi_f"]) / 2.0
            local = stamp.tz_convert(config.STATION_TZ)
            day_start = pd.Timestamp(t, tz=config.STATION_TZ)
            r.update({
                "logged_at": stamp,
                "our_p": p,
                "mu_f": mu, "sigma_f": sigma,
                "z_centre": abs(ref - mu) / sigma,
                # Hours of the target local day still ahead when we looked.
                # Negative once the day is over.
                "hours_to_close": float(
                    (day_start + pd.Timedelta(days=1) - local).total_seconds() / 3600.0),
                "hours_before_peak": hours_left,
                "excluded": False,
                "exclude_reason": "",
            })
            rows.append(r)
        print(f"  {t}: {len(mkts)} buckets logged  (our mu {mu:.1f} F, "
              f"sigma {sigma:.2f}){f', {len(failures)} skipped' if failures else ''}")

    if not rows:
        print("  nothing to log")
        return

    new = pd.DataFrame(rows)
    total = new.groupby("target_date")["our_p"].sum()
    for t, s in total.items():
        if not 0.97 <= s <= 1.03:
            print(f"  WARN {t:%Y-%m-%d}: our bucket probabilities sum to {s:.3f}, "
                  f"not 1 -- buckets may not tile the line")

    if LOG_PATH.exists():
        new = pd.concat([pd.read_parquet(LOG_PATH), new], ignore_index=True)
    new = new.drop_duplicates(subset=["logged_at", "market_id"], keep="last")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    new.sort_values(["logged_at", "lo_f"]).to_parquet(LOG_PATH, index=False)
    print(f"  log now {len(new)} rows -> {LOG_PATH}")


# --- resolution --------------------------------------------------------------

def cmd_resolve(args) -> None:
    """Attach outcomes, but only for local days that have actually ended.

    IEM's daily endpoint serves a row for the current day carrying the maximum
    reported *so far*. Treating that as settled scores the forecast against a
    partial maximum -- at 07:00 local it is close to the overnight low -- and
    marks the wrong bucket correct. Because outcomes were only ever filled where
    null, such a row was never revisited: one premature resolve corrupted that
    day's record permanently.
    """
    if not LOG_PATH.exists():
        raise wxio.SourceError("no log to resolve")
    log = apply_exclusions(pd.read_parquet(LOG_PATH))
    obs = klga_obs()

    if "outcome" not in log.columns:
        log["outcome"] = np.nan
        log["observed_f"] = np.nan

    # A target day is settled once the station's local day has ended.
    today_local = (pd.Timestamp(datetime.now(timezone.utc))
                   .tz_convert(config.STATIONS[MARKET_STATION_ICAO].tz).date())
    settled = pd.to_datetime(log["target_date"]) < pd.Timestamp(today_local)

    # Clear anything attached before its day ended, so it resolves properly once
    # the day closes.
    premature = log["outcome"].notna() & ~settled
    if premature.any():
        log.loc[premature, "outcome"] = np.nan
        log.loc[premature, "observed_f"] = np.nan
        print(f"  cleared {int(premature.sum())} outcome(s) attached before the "
              f"local day had ended")

    # Recompute every settled row rather than only the null ones. An outcome is
    # a pure function of the reported daily max, so this is idempotent -- and it
    # is what repairs a day that was resolved early under the old rule and has
    # since closed, whose stored outcome would otherwise never be revisited.
    pending = int((log["outcome"].isna() & ~settled).sum())
    resolved = corrected = 0
    for i in log.index[settled]:
        t = pd.Timestamp(log.at[i, "target_date"])
        if t not in obs.index:
            continue
        y = float(obs.loc[t])
        was = log.at[i, "observed_f"]
        outcome = float(log.at[i, "lo_f"] <= y < log.at[i, "hi_f"])
        if pd.isna(was):
            resolved += 1
        elif float(was) != y:
            corrected += 1
        log.at[i, "observed_f"] = y
        log.at[i, "outcome"] = outcome

    log.to_parquet(LOG_PATH, index=False)
    done = int(log["outcome"].notna().sum())
    print(f"  resolved {resolved} rows this run; {done}/{len(log)} now have outcomes")
    if resolved:
        days = log.loc[log["outcome"].notna(), "target_date"].nunique()
        print(f"  covering {days} target days")
    if corrected:
        print(f"  corrected {corrected} row(s) whose stored observation disagreed "
              f"with the reported daily max")
    if pending:
        print(f"  {pending} row(s) awaiting the end of their local day")


# --- scoring -----------------------------------------------------------------

def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def skill_str(ours: float, theirs: float) -> str:
    """Brier skill against the market, or why it is not defined.

    A market Brier near zero means it priced the outcome essentially perfectly;
    the ratio then explodes and reports a meaningless number, so it is refused
    rather than printed.
    """
    if theirs < 1e-4:
        return "n/a"
    return f"{1 - ours / theirs:+.1%}"


def cmd_score(args) -> None:
    # Validated before anything is read, so a bad flag fails immediately rather
    # than partway through a report.
    if getattr(args, "pnl", False):
        validate_pnl_args(args.edge, args.stake)

    if not LOG_PATH.exists():
        raise wxio.SourceError("no log to score")
    log = apply_exclusions(pd.read_parquet(LOG_PATH))
    if "outcome" not in log.columns or log["outcome"].notna().sum() == 0:
        raise wxio.SourceError("no resolved rows yet; run 'resolve' after a market settles")

    resolved_all = log[log["outcome"].notna()]

    # Ahead of every mid-dependent abort below: the P&L fills at ask and bid and
    # needs no mid at all, so a log of entirely one-sided books -- the normal
    # state for deep out-of-the-money buckets -- must not suppress it.
    if getattr(args, "pnl", False):
        pnl_report(log, args.edge, args.stake, args.include_late)

    df = resolved_all[resolved_all["mid"].notna()].copy()
    dropped = len(resolved_all) - len(df)

    # Tested before the late filter, so that "no two-sided book" is reported as
    # itself. Folded in after, an all-one-sided log emptied df and was blamed on
    # the cutoff instead -- a diagnostic pointing at the wrong cause.
    if df.empty:
        raise wxio.SourceError(
            f"{len(resolved_all)} resolved rows but none has a two-sided book, so "
            f"no mid price exists to score against. Deep out-of-the-money buckets "
            f"are usually quoted ask-only.")

    late = 0
    if not args.include_late:
        late = int(df["excluded"].sum())
        df = df[~df["excluded"]]
        if df.empty:
            raise wxio.SourceError(
                f"every resolved row was logged after its season's cutoff "
                f"({cutoff_summary()}) local on "
                f"its target day, when the maximum has usually already happened and "
                f"the market prices a near-known outcome. Scoring those would "
                f"measure nothing. Use --include-late to override.")

    # Closing price per market = the last mid we logged before resolution.
    close = (df.sort_values("logged_at").groupby("market_id")["mid"].last()
             .rename("close"))
    df = df.join(close, on="market_id")

    # CLV: did the market move toward our view after we looked? Signed by the
    # side we were on. Positive means the line came to us.
    side = np.sign(df["our_p"] - df["mid"])
    df["clv"] = side * (df["close"] - df["mid"])
    df["right_side"] = (side == np.sign(df["outcome"] - df["mid"])).astype(float)
    df["dist"] = pd.cut(df["z_centre"], bins=DISTANCE_BINS, labels=DISTANCE_LABELS,
                        right=False)

    ours = brier(df["our_p"].to_numpy(float), df["outcome"].to_numpy(float))
    theirs = brier(df["mid"].to_numpy(float), df["outcome"].to_numpy(float))

    print(f"resolved rows : {len(df)}  across {df['target_date'].nunique()} target days")
    if dropped:
        print(f"dropped       : {dropped} resolved rows with a one-sided book "
              f"(no mid to score against)")
    if late:
        print(f"excluded      : {late} rows logged after the season's cutoff "
              f"({cutoff_summary()}) local, when the max has usually happened")
    print(f"log window    : {df['logged_at'].min():%Y-%m-%d %H:%M} -> "
          f"{df['logged_at'].max():%Y-%m-%d %H:%M} UTC")
    print(f"snapshots/day : {len(df) / max(df['target_date'].nunique(), 1) / 11:.1f}")

    print(f"\n{'':<18}{'ours':>9}{'market':>9}{'diff':>9}")
    print("-" * 45)
    print(f"{'Brier':<18}{ours:>9.4f}{theirs:>9.4f}{ours - theirs:>+9.4f}")
    print(f"{'skill vs market':<18}{'':>9}{'':>9}{skill_str(ours, theirs):>9}")
    print(f"\naverage CLV      {df['clv'].mean():+.4f}  "
          f"(market moved toward us by this much, in probability)")
    print(f"right side       {df['right_side'].mean():.1%} of rows")

    print(f"\nBy distance of bucket centre from our mean")
    head = (f"\n{'band':<16}{'n':>6}{'ourBrier':>10}{'mktBrier':>10}"
            f"{'skill':>9}{'CLV':>9}{'right':>8}")
    print(head)
    print("-" * (len(head) - 1))
    for label in DISTANCE_LABELS:
        g = df[df["dist"] == label]
        if g.empty:
            print(f"{label:<16}{0:>6}")
            continue
        b_us = brier(g["our_p"].to_numpy(float), g["outcome"].to_numpy(float))
        b_mk = brier(g["mid"].to_numpy(float), g["outcome"].to_numpy(float))
        print(f"{label:<16}{len(g):>6}{b_us:>10.4f}{b_mk:>10.4f}"
              f"{skill_str(b_us, b_mk):>9}{g['clv'].mean():>+9.4f}"
              f"{g['right_side'].mean():>8.0%}")

    days = df["target_date"].nunique()
    if days < 30:
        print(f"\n  NOTE: {days} resolved day(s). Brier differences and CLV are not")
        print(f"  interpretable at this sample size -- the buckets within a day are")
        print(f"  highly dependent, so the effective sample is nearer {days} than "
              f"{len(df)}.")


def hourly_history(start_year: int = 2019) -> pd.DataFrame:
    """Every reported temperature at the market station, in station-local time.

    Local rather than UTC by deliberate exception to hard rule 6: a daily
    maximum is defined on the local day, so this frame *is* the resolution
    boundary. Everything derived from it is put back on UTC timestamps.
    """
    import io
    config.use_station(MARKET_STATION_ICAO)
    today = date.today()
    r = requests.get(config.IEM_ASOS_URL, params={
        "station": config.STATION_IEM_ID, "data": "tmpf",
        "year1": start_year, "month1": 1, "day1": 1,
        "year2": today.year, "month2": today.month, "day2": today.day,
        "tz": config.STATION_TZ, "format": "onlycomma", "latlon": "no",
        "missing": "empty", "trace": "empty", "report_type": 3,
    }, timeout=300)
    if r.status_code != 200:
        raise wxio.SourceError(f"IEM hourly: HTTP {r.status_code}")

    d = pd.read_csv(io.StringIO(r.text))
    d["valid"] = pd.to_datetime(d["valid"])
    d["tmpf"] = pd.to_numeric(d["tmpf"], errors="coerce")
    d = d.dropna(subset=["tmpf"]).sort_values("valid")
    d["day"] = pd.to_datetime(d["valid"].dt.date)
    d["hour"] = d["valid"].dt.hour + d["valid"].dt.minute / 60.0
    return d


def peak_times(hourly: pd.DataFrame) -> pd.DataFrame:
    """One row per complete local day: when that day's maximum was reported."""
    counts = hourly.groupby("day")["tmpf"].size()
    peak = hourly.loc[hourly.groupby("day")["tmpf"].idxmax(),
                      ["day", "hour", "tmpf"]].copy()
    peak["n"] = counts.reindex(peak["day"]).values
    peak = peak[peak["n"] >= config.MIN_HOURS_PER_DAY].reset_index(drop=True)
    peak["season"] = peak["day"].dt.month.map(SEASON_OF_MONTH)
    return peak


def schedule_bounds(days: pd.Series, cut_hours: np.ndarray,
                    lag_h: float) -> tuple[pd.Series, pd.Series]:
    """UTC open and close of the polling window for each local day.

    Opens when that day's 00Z run becomes retrievable, closes at the seasonal
    cutoff in station-local time -- which is a different UTC hour in summer and
    winter, so the conversion cannot be folded into a constant.
    """
    local_midnight = days.dt.tz_localize(config.STATION_TZ)
    open_utc = days.dt.tz_localize("UTC") + pd.Timedelta(hours=lag_h)
    close_utc = (local_midnight + pd.to_timedelta(cut_hours, unit="h")).dt.tz_convert("UTC")
    return open_utc, close_utc


def simulate_schedule(peak: pd.DataFrame, cut_hours: np.ndarray,
                      lag_h: float, interval_min: int) -> pd.DataFrame:
    """Count the rows a polling schedule would produce, clean and post-peak.

    A row is post-peak if it is timestamped after that day's realised maximum.
    This is the metric that matters, because prices are snapshotted throughout
    the window rather than once at the cutoff: a day that peaks two hours before
    its cutoff still yields a full morning of clean rows and only leaks the last
    few.
    """
    step = pd.Timedelta(minutes=interval_min)
    open_utc, close_utc = schedule_bounds(peak["day"], cut_hours, lag_h)
    peak_utc = ((peak["day"].dt.tz_localize(config.STATION_TZ)
                 + pd.to_timedelta(peak["hour"], unit="h")).dt.tz_convert("UTC"))

    # Floor division on Timedeltas rather than on int64 nanoseconds: pandas
    # stores these columns in microseconds, so a hand-rolled ns conversion is
    # silently wrong by 1000x.
    total = np.maximum(((close_utc - open_utc) // step).to_numpy() + 1, 0)
    clean = np.clip(((peak_utc - open_utc) // step).to_numpy() + 1, 0, total)
    return pd.DataFrame({"season": peak["season"].to_numpy(),
                         "day": peak["day"].to_numpy(),
                         "total": total, "clean": clean, "post": total - clean})


def print_schedule(title: str, sim: pd.DataFrame) -> None:
    print(f"\n{title}")
    head = (f"{'season':<8}{'days':>7}{'clean':>9}{'post':>8}{'total':>9}"
            f"{'leak%':>8}{'clean/day':>11}")
    print(head)
    print("-" * len(head))
    for s_ in ("DJF", "MAM", "JJA", "SON", "ALL"):
        g = sim if s_ == "ALL" else sim[sim["season"] == s_]
        c, p, t = int(g["clean"].sum()), int(g["post"].sum()), int(g["total"].sum())
        print(f"{s_:<8}{len(g):>7}{c:>9,}{p:>8,}{t:>9,}"
              f"{(p / t if t else 0):>8.1%}{(c / max(len(g), 1)):>11.2f}")


def simulate_override(hourly: pd.DataFrame, peak: pd.DataFrame,
                      pred: pd.DataFrame, cut_hours: np.ndarray, lag_h: float,
                      interval_min: int, quantile: float) -> dict:
    """What the same-day override would have stopped, row by row.

    The override fires at the first snapshot whose realised max so far already
    exceeds the given quantile of our own predictive distribution; from then on
    the day is treated as settled and nothing further is logged. Reported as
    post-peak rows prevented (the point of it) against clean rows lost (the
    price of it).
    """
    from statistics import NormalDist

    z = NormalDist().inv_cdf(quantile)
    step = pd.Timedelta(minutes=interval_min)
    thr = (pred["mu"] + z * pred["sigma"]).to_dict()

    open_utc, close_utc = schedule_bounds(peak["day"], cut_hours, lag_h)
    peak_utc = ((peak["day"].dt.tz_localize(config.STATION_TZ)
                 + pd.to_timedelta(peak["hour"], unit="h")).dt.tz_convert("UTC"))
    by_day = {d: g for d, g in hourly.groupby("day")}

    rows = []
    for i, day in enumerate(peak["day"]):
        if day not in thr or day not in by_day:
            continue
        n = int(max((close_utc.iloc[i] - open_utc.iloc[i]) // step + 1, 0))
        if n == 0:
            continue
        stamps = (open_utc.iloc[i]
                  + pd.to_timedelta(np.arange(n) * interval_min, unit="m"))
        is_post = np.asarray(stamps > peak_utc.iloc[i])

        # Running max at each snapshot: the day's readings are already sorted,
        # so a cumulative max plus a searchsorted gives "what was known by then".
        # Compared tz-naive in UTC, because searchsorted on a tz-aware column
        # falls back to an object array of Timestamps.
        g = by_day[day]
        obs_utc = (g["valid"].dt.tz_localize(config.STATION_TZ, ambiguous="NaT",
                                             nonexistent="NaT").dt.tz_convert("UTC"))
        keep = obs_utc.notna()
        obs_utc, vals = obs_utc[keep], g.loc[keep, "tmpf"]
        if obs_utc.empty:
            continue
        seen = np.searchsorted(obs_utc.dt.tz_localize(None).to_numpy(),
                               stamps.tz_localize(None).to_numpy(), side="right")
        runmax = np.concatenate([[-np.inf], np.maximum.accumulate(vals.to_numpy())])[seen]

        fired = np.flatnonzero(runmax >= thr[day])
        k = int(fired[0]) if len(fired) else n     # index of the first stopped row
        rows.append({
            "season": peak["season"].iloc[i], "total": n,
            "post": int(is_post.sum()),
            "post_stopped": int(is_post[k:].sum()),
            "clean_stopped": int((~is_post[k:]).sum()),
        })
    return {"frame": pd.DataFrame(rows), "z": z}


def validate_pnl_args(edge: float, stake: float) -> None:
    """Reject nonsense before any data is read.

    A zero stake divides by zero halfway through the report; a negative edge
    makes every row qualify on both sides at once, which is not a strategy.
    """
    if stake <= 0:
        raise SystemExit(f"--stake must be positive, got {stake:g}")
    if edge < 0:
        raise SystemExit(
            f"--edge must be zero or positive, got {edge:g}. A negative edge "
            f"takes positions the quoted price already argues against.")


def pnl_report(log: pd.DataFrame, edge: float, stake: float,
               include_late: bool) -> None:
    """Notional P&L from the logged quotes, filled at the price actually offered.

    This is a simulation of betting, not a record of one. It is deliberately
    pessimistic in the one place backtests usually cheat: **fills are at the ask
    to go long and at the bid to go short, never at the mid.** You cannot trade
    at the mid. On these buckets the spread is often 3-6 cents against an edge of
    similar size, so a mid-filled P&L can show a profit that does not survive
    contact with the book.

    Two sides, both executable at quoted prices:
      long YES  -- pay `ask`,     collect 1 if the bucket contains the max
      long NO   -- pay `1 - bid`, collect 1 if it does not

    One entry per bucket per day, at the first snapshot where either side shows
    the edge. The two sides cannot both qualify *within one snapshot* -- that
    would need bid > ask -- but they can qualify at different times of day as
    our number and the quotes both move, so the entry is deduplicated on
    (day, bucket) alone. Booking both sides on one bucket is not a hedge, it is
    paying two spreads for a payoff of exactly 1.

    Holding to resolution means no exit assumption is needed.

    LIMITATION -- top of book only. The quotes carry a price and no size, so
    every fill here assumes the full stake trades at the best price. It does
    not. Any real order walks the book and fills worse, and these markets are
    thin. Read the result as an upper bound at one share, not as a P&L
    achievable at size.
    """
    validate_pnl_args(edge, stake)

    df = log[log["outcome"].notna()].copy()
    if not include_late:
        df = df[~df["excluded"]]
    if df.empty:
        print("\nNo resolved rows to simulate against.")
        return

    resolved_days = int(df["target_date"].nunique())
    if resolved_days < MIN_PNL_DAYS:
        print(f"\n\nNotional P&L: withheld. {resolved_days} resolved day(s) in the "
              f"log, {MIN_PNL_DAYS} required.")
        print(f"  The buckets within a day resolve together, so the effective "
              f"sample is the")
        print(f"  day count. Below {MIN_PNL_DAYS} days a P&L figure is noise that "
              f"reads like a result,")
        print("  which is worse than no figure at all. "
              f"{MIN_PNL_DAYS - resolved_days} to go.")
        return

    df = df.sort_values("logged_at")
    yes = df[df["ask"].notna() & (df["our_p"] - df["ask"] >= edge)].copy()
    yes["side"] = "YES"
    yes["price"] = yes["ask"]
    yes["won"] = yes["outcome"]

    no = df[df["bid"].notna() & (df["bid"] - df["our_p"] >= edge)].copy()
    no["side"] = "NO"
    no["price"] = 1.0 - no["bid"]
    no["won"] = 1.0 - no["outcome"]

    trades = pd.concat([yes, no], ignore_index=True)
    if trades.empty:
        print(f"\nNo bucket ever showed an edge of {edge:.0%} or more against the "
              f"quoted price.")
        print("  That is a result, not a failure: it says the market was never "
              "far enough")
        print("  from our number to be worth taking at the spread on offer.")
        return

    # One entry per bucket per day, from ONE snapshot. drop_duplicates keeps a
    # whole row; groupby().first() would take the first non-null of each column
    # independently and assemble a row that never existed -- a price from 08:00
    # beside a mid from 12:00, which invents a spread out of nothing.
    trades = (trades.sort_values("logged_at")
              .drop_duplicates(subset=["target_date", "market_id"], keep="first"))

    trades["pnl"] = stake * (trades["won"] - trades["price"])
    staked = float((trades["price"] * stake).sum())
    pnl = float(trades["pnl"].sum())

    print(f"\n\nNotional P&L  (edge >= {edge:.0%}, {stake:g} share per bucket, "
          f"held to resolution)")
    print("-" * 66)
    print(f"trades        : {len(trades)} across "
          f"{trades['target_date'].nunique()} day(s)   "
          f"[{int((trades['side'] == 'YES').sum())} YES, "
          f"{int((trades['side'] == 'NO').sum())} NO]")
    print(f"hit rate      : {trades['won'].mean():.1%}")
    print(f"staked        : {staked:.2f}")
    print(f"P&L           : {pnl:+.2f}   ({pnl / staked:+.1%} on stake)")

    for side in ("YES", "NO"):
        g = trades[trades["side"] == side]
        if g.empty:
            continue
        s = float((g["price"] * stake).sum())
        print(f"  {side:<4}: {len(g):>4} trades  hit {g['won'].mean():>5.1%}  "
              f"staked {s:>7.2f}  P&L {g['pnl'].sum():>+7.2f}  "
              f"({g['pnl'].sum() / s:+.1%})")

    # Spread cost, computed over ONE trade set: only the trades that have a mid.
    # Subtracting a mid-filled P&L over some trades from an ask-filled P&L over
    # all of them is not a spread, it is the difference between two populations.
    cmp = trades[trades["mid"].notna()]
    if len(cmp):
        cmp_mid_price = np.where(cmp["side"] == "YES", cmp["mid"], 1.0 - cmp["mid"])
        cmp_ask = float((stake * (cmp["won"] - cmp["price"])).sum())
        cmp_mid = float((stake * (cmp["won"] - cmp_mid_price)).sum())
        print(f"\n  On the {len(cmp)} of {len(trades)} trades with a two-sided "
              f"book:")
        print(f"    filled at ask/bid (real)  : {cmp_ask:+.2f}")
        print(f"    filled at mid (not real)  : {cmp_mid:+.2f}")
        print(f"    spread cost               : {cmp_mid - cmp_ask:.2f}")
    if len(cmp) < len(trades):
        print(f"  {len(trades) - len(cmp)} trade(s) had a one-sided book and are "
              f"excluded from that comparison")
        print("  only -- they are included in the P&L above, where the quoted "
              "side is all that is needed.")

    print("\n  Fills assume the whole stake trades at the best quote. The book's "
          "depth was")
    print("  never recorded, so this is an upper bound at one share, not a P&L "
          "achievable")
    print("  at size.")


def cmd_cutoffs(args) -> None:
    """Score the cutoffs the way the logger actually runs: per row, not per day.

    An earlier version reported the share of *days* whose maximum preceded the
    cutoff. That overstated the cost badly, because prices are snapshotted every
    30 minutes from the moment the forecast lands: a day peaking at 14:00 under
    a 16:00 cutoff is not a lost day, it is ~17 clean rows and 4 leaked ones. So
    leakage here is the share of logged rows timestamped after that day's
    realised maximum, and it is read against the absolute count of clean rows --
    a schedule that leaks more but yields far more clean rows is the better one.
    """
    hourly = hourly_history()
    peak = peak_times(hourly)
    print(f"station: {config.STATION_ICAO}   hourly history from IEM")
    print(f"{len(peak):,} complete days, {peak['day'].min():%Y-%m-%d} -> "
          f"{peak['day'].max():%Y-%m-%d}")
    print(f"schedule: every {args.interval} min, from 00Z+{args.lag:.2f}h "
          f"(measured ECMWF publication lag) to the cutoff, station-local")

    seasonal = peak["season"].map(PEAK_LOCAL_HOUR_BY_SEASON).to_numpy(float)
    flat = np.full(len(peak), args.flat)

    sim_flat = simulate_schedule(peak, flat, args.lag, args.interval)
    sim_seas = simulate_schedule(peak, seasonal, args.lag, args.interval)
    print_schedule(f"flat {args.flat:.0f}:00 cutoff -- row-weighted", sim_flat)
    print_schedule("season-aware cutoffs "
                   + ", ".join(f"{k} {v:.1f}" for k, v in
                               PEAK_LOCAL_HOUR_BY_SEASON.items())
                   + " -- row-weighted", sim_seas)

    cf, cs = int(sim_flat["clean"].sum()), int(sim_seas["clean"].sum())
    print(f"\n  season-aware buys {cs - cf:+,} clean rows ({(cs / cf - 1):+.1%}) "
          f"for leakage {sim_flat['post'].sum() / sim_flat['total'].sum():.1%} -> "
          f"{sim_seas['post'].sum() / sim_seas['total'].sum():.1%}.")
    print("  Post-peak rows are logged but flagged `excluded`, so they cost "
          "storage, not scoring.")

    if not args.override:
        print("\n  (pass --override to measure what the same-day p"
              f"{config.DETERMINED_QUANTILE * 100:.0f} guard actually stops; it needs")
        print("   the walk-forward blend and takes a few minutes)")
        return

    # --- what the same-day override adds on top of the clock ------------------
    comp = components(MARKET_STATION_ICAO)
    bl = blend_walk_forward(comp, comp.index, DEFAULT_MODE, inflate=default_inflate())
    pred = pd.DataFrame({"mu": bl["mu"], "sigma": bl["sigma"]})
    print(f"\nSame-day override, on the {len(pred)} days with a walk-forward "
          f"forecast ({pred.index.min():%Y-%m-%d} -> {pred.index.max():%Y-%m-%d})")
    print("  season-aware schedule; a fired override stops all later rows that day")

    head = (f"{'quantile':<10}{'rows':>8}{'post':>7}{'post stopped':>15}"
            f"{'clean lost':>13}")
    print(head)
    print("-" * len(head))
    shipped = config.DETERMINED_QUANTILE
    scan = {}
    for q in sorted({0.50, 0.60, 0.75, 0.90, shipped}):
        f = simulate_override(hourly, peak, pred, seasonal, args.lag,
                              args.interval, q)["frame"]
        tot, post = int(f["total"].sum()), int(f["post"].sum())
        ps, cl = int(f["post_stopped"].sum()), int(f["clean_stopped"].sum())
        scan[q] = (tot, post, ps, cl)
        mark = "  <- shipped" if q == shipped else ""
        print(f"p{q * 100:<9.0f}{tot:>8,}{post:>7,}"
              f"{ps:>10,} ({ps / max(post, 1):>3.0%}){cl:>8,} "
              f"({cl / max(tot - post, 1):>3.1%}){mark}")

    tot, post, ps, cl = scan[shipped]
    print(f"\n  At the shipped p{shipped * 100:.0f}: {post - ps:,} post-peak rows "
          f"({(post - ps) / tot:.1%} of all rows) still get through, and "
          f"{cl:,} clean")
    print(f"  rows ({cl / max(tot - post, 1):.1%}) are stopped early. The clock is "
          f"still the primary guard;")
    print("  this is a backstop with a measured price. Tune with "
          "WEATHERBOT_DETERMINED_QUANTILE.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("predict", cmd_predict), ("log", cmd_log),
                     ("resolve", cmd_resolve), ("score", cmd_score),
                     ("cutoffs", cmd_cutoffs)):
        sp = sub.add_parser(name)
        sp.set_defaults(fn=fn)
        if name == "score":
            sp.add_argument("--include-late", action="store_true",
                            help="also score snapshots taken after the daily max")
            sp.add_argument("--pnl", action="store_true",
                            help="also simulate notional P&L, filled at ask/bid")
            sp.add_argument("--edge", type=float, default=DEFAULT_EDGE,
                            help=f"minimum edge over the quoted price to take a "
                                 f"position (default {DEFAULT_EDGE:.2f})")
            sp.add_argument("--stake", type=float, default=1.0,
                            help="shares per qualifying bucket (default 1)")
        if name == "predict":
            sp.add_argument("--date", type=date.fromisoformat, default=None,
                            help="single target day instead of the default window")
        if name == "cutoffs":
            sp.add_argument("--override", action="store_true",
                            help="also simulate the same-day settlement guard "
                                 "across quantiles (slow)")
            sp.add_argument("--lag", type=float, default=WINDOW_OPEN_HOURS_AFTER_00Z,
                            help="hours after 00Z the run becomes retrievable "
                                 f"(default {WINDOW_OPEN_HOURS_AFTER_00Z}, measured)")
            sp.add_argument("--interval", type=int, default=POLL_INTERVAL_MIN,
                            help=f"minutes between snapshots (default {POLL_INTERVAL_MIN})")
            sp.add_argument("--flat", type=float, default=13.0,
                            help="comparison flat cutoff, local hour (default 13)")
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
