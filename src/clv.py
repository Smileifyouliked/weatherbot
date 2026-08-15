"""Log Polymarket daily-temperature prices against our forecast, and score them.

Logging and scoring only. Nothing here places, sizes or recommends a trade.

    python3 src/clv.py predict     # cache today's predictive distribution (slow, once/day)
    python3 src/clv.py log         # snapshot market prices + our probabilities (fast, poll often)
    python3 src/clv.py resolve     # attach outcomes once the day is observed
    python3 src/clv.py score       # Brier, CLV, side-of-market, by distance from our mean

STATION MISMATCH, and what this module does about it
----------------------------------------------------
Polymarket's "NYC" temperature market resolves on **KLGA (LaGuardia)** via
Weather Underground, not KNYC (Central Park), which is what the rest of this
project forecasts. Over 366 paired days KLGA minus KNYC is +0.55 F on average
with a standard deviation of 1.78 F, and +1.92 F in summer. The blend's own
sigma is about 2 F, so the station gap is as large as the entire predictive
uncertainty: feeding KNYC probabilities into a KLGA-settled market would swamp
any real edge and make the scoring meaningless.

So the cached prediction is a KLGA distribution, produced by regressing observed
KLGA daily maxima on the KNYC blend mean plus day-of-year harmonics, fitted
walk-forward exactly like every other model here (hard rules 1 and 2). This is a
thin adaptation layer, not a KLGA pipeline: the underlying forecast fields are
still interpolated to Central Park. A genuine KLGA build would re-pull ECMWF and
NBM at LaGuardia and should beat it.

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
    FEATURES_MEAN, LEAD, _standardize, build_features, fit_log_sigma,
    predict_sigma, ridge_fit, select_features, tune_alpha, walk_forward,
)
from blend import DEFAULT_MODE, NBM_FEATURES, NBM_SIGMA, nbm_frame  # noqa: E402

GAMMA = "https://gamma-api.polymarket.com"
EVENT_SLUG = "highest-temperature-in-nyc-on-{month}-{day}-{year}"
MARKET_STATION = "LGA"          # what Polymarket actually settles on
MARKET_NETWORK = "NY_ASOS"

PRED_PATH = config.DATA_DIR / "clv_pred.parquet"
LOG_PATH = config.DATA_DIR / "clv_log.parquet"

ADAPT_FEATURES = ["blend_mu", "sin_doy", "cos_doy"]
ADAPT_SIGMA = ["log_blend_sigma", "sin_doy", "cos_doy"]
MIN_ADAPT_TRAIN = 180
HOLDOUT_FRAC = 0.25

# Distance of a bucket centre from our predictive mean, in units of our sigma.
DISTANCE_BINS = [0.0, 0.5, 1.0, 2.0, np.inf]
DISTANCE_LABELS = ["<0.5s (centre)", "0.5-1s", "1-2s", ">2s (tail)"]

# Local hour by which the daily maximum has usually happened. A snapshot taken
# after this is scoring a near-known outcome: the market has seen the morning
# and prices it at ~0 or ~1, which flatters it and tells us nothing about
# forecast skill. Such rows are logged but excluded from scoring by default.
PEAK_LOCAL_HOUR = 13


def c_to_f(c):
    return np.asarray(c) * 9.0 / 5.0 + 32.0


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


def market_rows(event: dict, target: date) -> list[dict]:
    """One row per bucket, with best bid/ask and a mid where both sides exist."""
    out = []
    for m in event.get("markets", []):
        q = m.get("question") or ""
        bounds = parse_bucket(q)
        if bounds is None:
            continue
        bid = m.get("bestBid")
        ask = m.get("bestAsk")
        bid = float(bid) if bid is not None else np.nan
        ask = float(ask) if ask is not None else np.nan
        # Mid only where both sides are quoted; a one-sided book has no mid, and
        # substituting last-trade would silently mix two different quantities.
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
    return out


# --- prediction --------------------------------------------------------------

def klga_obs() -> pd.Series:
    """Reported KLGA daily maxima in F -- the quantity the market settles on."""
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


def blend_series() -> pd.DataFrame:
    """KNYC blend mean and spread per day, from the existing pipeline."""
    obs, fc_daily = load_daily(LEAD)
    ec = build_features().join(obs.rename("obs"), how="inner").dropna(
        subset=FEATURES_MEAN + ["obs"])
    nbm = nbm_frame(obs)
    feats = select_features(ec, ec.index[int(len(ec) * 0.6)])

    mos = walk_forward(ec, ec.index, feats, "rolling")
    npp = walk_forward(nbm, nbm.index, NBM_FEATURES, "rolling",
                       sigma_features=NBM_SIGMA)
    comp = pd.DataFrame({"mu_mos": mos["mu"], "sigma_mos": mos["sigma"]}).join(
        pd.DataFrame({"mu_nbm": npp["mu"], "sigma_nbm": npp["sigma"]}), how="inner")

    # Inverse-variance weighting: the project default (CLAUDE.md rule 5a).
    wa = 1.0 / comp["sigma_mos"] ** 2
    wb = 1.0 / comp["sigma_nbm"] ** 2
    w = wa / (wa + wb)
    comp["blend_mu"] = w * comp["mu_mos"] + (1.0 - w) * comp["mu_nbm"]
    comp["blend_sigma"] = np.sqrt(1.0 / (wa + wb))
    return comp[["blend_mu", "blend_sigma"]]


def predict_klga(target: date) -> tuple[float, float, int]:
    """Walk-forward KLGA predictive mean and spread for one target day.

    Fitted only on days whose observation was complete before the lead +1d
    forecast for `target` was issued (cutoff target-2), like the rest of the
    project. Returns (mu_F, sigma_F, n_train).
    """
    blend = blend_series()
    obs = klga_obs()
    df = blend.join(obs.rename("obs"), how="left")
    doy = df.index.dayofyear.values
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    df["log_blend_sigma"] = np.log(df["blend_sigma"].clip(lower=0.1))

    D = pd.Timestamp(target)
    if D not in df.index:
        raise wxio.SourceError(
            f"no blend forecast for {target}; the backbone run for that day may "
            f"not be cached yet (run ingest_forecast.py)")

    cutoff = D - pd.Timedelta(days=2)
    train = df.loc[:cutoff].dropna(subset=ADAPT_FEATURES + ["obs"])
    if len(train) < MIN_ADAPT_TRAIN:
        raise wxio.SourceError(
            f"only {len(train)} training days before {target}; need "
            f"{MIN_ADAPT_TRAIN}")

    X = train[ADAPT_FEATURES].to_numpy(float)
    y = train["obs"].to_numpy(float)
    Xte = df.loc[[D], ADAPT_FEATURES].to_numpy(float)
    alpha = tune_alpha(X, y)
    Xs, Xte_s = _standardize(X, Xte)
    beta, b0 = ridge_fit(Xs, y, alpha)
    mu = float((Xte_s @ beta + b0)[0])

    resid = y - (Xs @ beta + b0)
    Z = train[ADAPT_SIGMA].to_numpy(float)
    Zte = df.loc[[D], ADAPT_SIGMA].to_numpy(float)
    Zs, Zte_s = _standardize(Z, Zte)
    gamma = fit_log_sigma(Zs, resid)
    sigma = float(predict_sigma(gamma, Zte_s)[0])

    # Same held-out spread scalar as the blend: fitted on the whole window it is
    # identically 1, being redundant with the log-sigma intercept.
    k = int(len(train) * (1.0 - HOLDOUT_FRAC))
    if k >= 60 and len(train) - k >= 30:
        Za, Zb = _standardize(Z[:k], Z[k:])
        s_held = predict_sigma(fit_log_sigma(Za, resid[:k]), Zb)
        sigma *= float(np.sqrt(np.mean((resid[k:] / s_held) ** 2)))

    return mu, sigma, len(train)


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
            mu, sigma, n = predict_klga(t)
        except wxio.SourceError as exc:
            print(f"  {t}: not yet available -- {str(exc)[:90]}")
            continue
        rows.append({"target_date": pd.Timestamp(t), "mu_f": mu, "sigma_f": sigma,
                     "n_train": n, "computed_at": utcnow()})
        print(f"  {t}: KLGA mu {mu:.1f} F, sigma {sigma:.2f} F  (n_train={n})")

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

        mu = float(pred.loc[ts, "mu_f"])
        sigma = float(pred.loc[ts, "sigma_f"])
        mkts = market_rows(event, t)
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
                "hours_before_peak": float(
                    (day_start + pd.Timedelta(hours=PEAK_LOCAL_HOUR) - local)
                    .total_seconds() / 3600.0),
            })
            rows.append(r)
        print(f"  {t}: {len(mkts)} buckets logged  (our mu {mu:.1f} F, sigma {sigma:.2f})")

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
    if not LOG_PATH.exists():
        raise wxio.SourceError("no log to resolve")
    log = pd.read_parquet(LOG_PATH)
    obs = klga_obs()

    if "outcome" not in log.columns:
        log["outcome"] = np.nan
        log["observed_f"] = np.nan

    need = log["outcome"].isna()
    resolved = 0
    for i in log.index[need]:
        t = pd.Timestamp(log.at[i, "target_date"])
        if t not in obs.index:
            continue
        y = float(obs.loc[t])
        log.at[i, "observed_f"] = y
        log.at[i, "outcome"] = float(log.at[i, "lo_f"] <= y < log.at[i, "hi_f"])
        resolved += 1

    log.to_parquet(LOG_PATH, index=False)
    done = int(log["outcome"].notna().sum())
    print(f"  resolved {resolved} rows this run; {done}/{len(log)} now have outcomes")
    if resolved:
        days = log.loc[log["outcome"].notna(), "target_date"].nunique()
        print(f"  covering {days} target days")


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
    if not LOG_PATH.exists():
        raise wxio.SourceError("no log to score")
    log = pd.read_parquet(LOG_PATH)
    if "outcome" not in log.columns or log["outcome"].notna().sum() == 0:
        raise wxio.SourceError("no resolved rows yet; run 'resolve' after a market settles")

    resolved_all = log[log["outcome"].notna()]
    df = resolved_all[resolved_all["mid"].notna()].copy()
    dropped = len(resolved_all) - len(df)

    late = 0
    if not args.include_late and "hours_before_peak" in df.columns:
        late = int((df["hours_before_peak"] <= 0).sum())
        df = df[df["hours_before_peak"] > 0]
        if df.empty:
            raise wxio.SourceError(
                f"every resolved row was logged after {PEAK_LOCAL_HOUR}:00 local on "
                f"its target day, when the maximum has usually already happened and "
                f"the market prices a near-known outcome. Scoring those would "
                f"measure nothing. Use --include-late to override.")
    if df.empty:
        raise wxio.SourceError(
            f"{len(resolved_all)} resolved rows but none has a two-sided book, so "
            f"no mid price exists to score against. Deep out-of-the-money buckets "
            f"are usually quoted ask-only.")

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
        print(f"excluded      : {late} rows logged after {PEAK_LOCAL_HOUR}:00 local, "
              f"when the max has usually already happened")
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("predict", cmd_predict), ("log", cmd_log),
                     ("resolve", cmd_resolve), ("score", cmd_score)):
        sp = sub.add_parser(name)
        sp.set_defaults(fn=fn)
        if name == "score":
            sp.add_argument("--include-late", action="store_true",
                            help="also score snapshots taken after the daily max")
        if name == "predict":
            sp.add_argument("--date", type=date.fromisoformat, default=None,
                            help="single target day instead of the default window")
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
