"""Three probabilistic baselines for daily max temperature at the target station.

No machine learning. Each baseline emits a Gaussian predictive distribution
(mu, sigma) per target day, and all three are scored on exactly the same days so
the comparison means something (hard rule 4).

  raw          ECMWF deterministic forecast max as mu; sigma from the standard
               deviation of recent forecast residuals.
  climatology  mean and std of the reported daily max for that calendar day,
               within a +/-7 day window, over history available at issue time.
  persistence  the most recent complete observed max as mu; sigma from the
               standard deviation of day-over-day changes.

Everything is walk-forward (hard rule 2): every statistic for target day D is
computed only from observations that were complete before that forecast was
issued. Nothing is fitted, so there is no train/test split to get wrong -- but
the trailing windows would leak just as easily, so they are cut explicitly.

Timing, which drives the cutoff below:
    lead +1d for local day D is the 00Z run on D, which is 20:00 local on D-1.
    At that moment local day D-1 is still in progress, so the last *complete*
    observed day is D-2. Using D-1 would mean using a daily max aggregated over
    hours that had not happened yet -- a hard rule 1 violation.

Usage:
    python3 src/baselines.py [--months 12] [--lead 1]
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# Brier thresholds, degrees F.
THRESHOLDS_F = list(range(40, 100, 5))

RESIDUAL_WINDOW_DAYS = 90
CLIM_HALF_WINDOW_DAYS = 7

# Minimum sample counts before a baseline is considered defined for a day.
MIN_RESIDUALS = 30
MIN_CLIM_SAMPLES = 20
MIN_DIFFS = 30

SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def c_to_f(c):
    return c * 9.0 / 5.0 + 32.0


# --- scoring -----------------------------------------------------------------

def _phi(z):
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _Phi(z):
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def gaussian_crps(y, mu, sigma):
    """Closed-form CRPS for a Gaussian forecast (Gneiting & Raftery 2007)."""
    y, mu, sigma = np.asarray(y, float), np.asarray(mu, float), np.asarray(sigma, float)
    if np.any(sigma <= 0):
        raise ValueError("sigma must be positive for every day")
    z = (y - mu) / sigma
    return sigma * (z * (2.0 * _Phi(z) - 1.0) + 2.0 * _phi(z) - 1.0 / math.sqrt(math.pi))


def exceedance_prob(mu, sigma, threshold):
    """P(Y > threshold) under the Gaussian forecast."""
    return 1.0 - _Phi((threshold - np.asarray(mu, float)) / np.asarray(sigma, float))


# --- data --------------------------------------------------------------------

def load(lead: int):
    if not config.DAILY.exists():
        raise SystemExit(f"{config.DAILY} not found -- run the ingestion modules first")
    daily = pd.read_parquet(config.DAILY)

    obs = daily[daily["source"] == "asos_daily_official"][["local_date", "tmax_c"]]
    obs = obs.assign(local_date=pd.to_datetime(obs["local_date"]))
    obs = obs.set_index("local_date")["tmax_c"].sort_index().map(c_to_f)

    fc = daily[
        (daily["source"] == "single_runs")
        & (daily["issue_time_confirmed"].fillna(False))
        & (daily["lead_days"] == lead)
    ][["local_date", "tmax_c", "issue_time"]]
    fc = fc.assign(local_date=pd.to_datetime(fc["local_date"]))
    fc = fc.set_index("local_date")["tmax_c"].sort_index().map(c_to_f)

    if obs.empty or fc.empty:
        raise SystemExit(f"no observations or no lead +{lead}d forecasts in {config.DAILY}")
    return obs, fc


# --- baselines ---------------------------------------------------------------

def build(obs: pd.Series, fc: pd.Series, targets: pd.DatetimeIndex):
    """Return one row per target day with mu/sigma for each baseline.

    For target day D the cutoff is D-2: the last local day that was complete
    when the lead +1d forecast was issued. Every statistic below is computed
    from obs.loc[:cutoff] only.
    """
    resid_all = (obs - fc).dropna()          # obs minus forecast, by target day
    diffs_all = obs.diff().dropna()          # day-over-day change
    obs_doy = obs.index.dayofyear.values

    rows = []
    for D in targets:
        cutoff = D - timedelta(days=2)

        hist = obs.loc[:cutoff]
        resid = resid_all.loc[:cutoff]
        resid = resid.loc[resid.index > cutoff - timedelta(days=RESIDUAL_WINDOW_DAYS)]
        diffs = diffs_all.loc[:cutoff]

        # 1. raw deterministic: model mean, climatological residual spread
        raw_mu = fc.get(D, np.nan)
        raw_sigma = resid.std(ddof=1) if len(resid) >= MIN_RESIDUALS else np.nan

        # 2. climatology: same calendar day +/-7d across available history
        dist = np.abs(obs_doy - D.dayofyear)
        near = np.minimum(dist, 366 - dist) <= CLIM_HALF_WINDOW_DAYS
        clim_hist = obs[near & (obs.index <= cutoff)]
        if len(clim_hist) >= MIN_CLIM_SAMPLES:
            clim_mu, clim_sigma = clim_hist.mean(), clim_hist.std(ddof=1)
        else:
            clim_mu = clim_sigma = np.nan

        # 3. persistence: last complete observed max, spread of daily changes
        pers_mu = hist.iloc[-1] if len(hist) else np.nan
        pers_sigma = diffs.std(ddof=1) if len(diffs) >= MIN_DIFFS else np.nan

        rows.append(
            {
                "local_date": D,
                "obs": obs.get(D, np.nan),
                "raw_mu": raw_mu, "raw_sigma": raw_sigma,
                "clim_mu": clim_mu, "clim_sigma": clim_sigma,
                "pers_mu": pers_mu, "pers_sigma": pers_sigma,
                "n_resid": len(resid), "n_clim": len(clim_hist), "n_diff": len(diffs),
            }
        )
    return pd.DataFrame(rows).set_index("local_date")


# --- reporting ---------------------------------------------------------------

def score(df: pd.DataFrame, name: str) -> dict:
    y = df["obs"].values
    mu, sigma = df[f"{name}_mu"].values, df[f"{name}_sigma"].values
    out = {
        "n": len(df),
        "CRPS": gaussian_crps(y, mu, sigma).mean(),
        "MAE": np.abs(y - mu).mean(),
    }
    for t in THRESHOLDS_F:
        p = exceedance_prob(mu, sigma, t)
        out[f"B{t}"] = np.mean((p - (y > t).astype(float)) ** 2)
    return out


def check_no_leakage(obs: pd.Series, fc: pd.Series, targets: pd.DatetimeIndex,
                     sample: int = 40) -> None:
    """Recompute sample days against a truncated history and require a match.

    If any statistic depended on an observation dated after the cutoff, hiding
    those observations would change the numbers. Identical output is direct
    evidence that nothing downstream of the cutoff was used (hard rules 1, 2).
    """
    rng = np.random.default_rng(0)
    picks = rng.choice(len(targets), size=min(sample, len(targets)), replace=False)
    cols = ["raw_mu", "raw_sigma", "clim_mu", "clim_sigma", "pers_mu", "pers_sigma"]

    full = build(obs, fc, targets)
    bad = []
    for i in sorted(picks):
        D = targets[i]
        cutoff = D - timedelta(days=2)
        one = build(obs.loc[:cutoff], fc, pd.DatetimeIndex([D]))
        a, b = full.loc[D, cols], one.loc[D, cols]
        if not np.allclose(a.values.astype(float), b.values.astype(float),
                           rtol=0, atol=1e-12, equal_nan=True):
            bad.append((D, a.to_dict(), b.to_dict()))

    if bad:
        for D, a, b in bad[:3]:
            print(f"  LEAK {D:%Y-%m-%d}\n    full={a}\n    trunc={b}")
        raise SystemExit(f"leakage check failed on {len(bad)}/{len(picks)} sampled days")
    print(f"leakage check: {len(picks)} sampled days identical when history is "
          f"truncated at the cutoff")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--lead", type=int, default=1)
    ap.add_argument("--check", action="store_true",
                    help="verify no future data reaches any baseline, then exit")
    args = ap.parse_args()

    obs, fc = load(args.lead)

    end = min(obs.index.max(), fc.index.max())
    start = end - timedelta(days=round(args.months * 30.44))
    targets = obs.index[(obs.index >= start) & (obs.index <= end)]

    if args.check:
        check_no_leakage(obs, fc, targets)
        return

    df = build(obs, fc, targets)

    # Every baseline is scored on the same days, so a day missing any one of them
    # is dropped from all three rather than quietly scoring different sets.
    needed = ["obs", "raw_mu", "raw_sigma", "clim_mu", "clim_sigma",
              "pers_mu", "pers_sigma"]
    complete = df.dropna(subset=needed)
    dropped = len(df) - len(complete)

    print(f"station    : {config.STATION_ICAO}")
    print(f"lead       : +{args.lead}d   (00Z run, issued 20:00 local the day before)")
    print(f"window     : {targets.min():%Y-%m-%d} -> {targets.max():%Y-%m-%d} "
          f"({args.months} months)")
    print(f"scored     : {len(complete)} days common to all baselines"
          + (f", {dropped} dropped for missing forecast or history" if dropped else ""))
    print(f"residuals  : trailing {RESIDUAL_WINDOW_DAYS}d, "
          f"median n={int(complete['n_resid'].median())}")
    print(f"climatology: +/-{CLIM_HALF_WINDOW_DAYS}d calendar window, "
          f"median n={int(complete['n_clim'].median())}")
    print("all statistics use only days complete before issue time (cutoff D-2)")

    names = [("raw", "raw deterministic"), ("clim", "climatology"),
             ("pers", "persistence")]
    scores = {label: score(complete, key) for key, label in names}

    # --- main table ---
    print(f"\nAll units degrees F. Brier at exceedance thresholds (P(Tmax > t)).")
    head = f"{'baseline':<20}{'CRPS':>7}{'MAE':>7}" + "".join(f"{'B'+str(t):>7}" for t in THRESHOLDS_F)
    print("\n" + head)
    print("-" * len(head))
    for _, label in names:
        s = scores[label]
        row = f"{label:<20}{s['CRPS']:>7.2f}{s['MAE']:>7.2f}"
        row += "".join(f"{s['B'+str(t)]:>7.3f}" for t in THRESHOLDS_F)
        print(row)

    # --- CRPS by season ---
    season = complete.index.month.map(SEASONS)
    print(f"\nCRPS by season (degrees F, lower is better)")
    order = ["DJF", "MAM", "JJA", "SON"]
    head2 = f"{'baseline':<20}" + "".join(f"{s:>9}" for s in order) + f"{'all':>9}"
    print("\n" + head2)
    print("-" * len(head2))
    for key, label in names:
        cells = []
        for s in order:
            sub = complete[season == s]
            cells.append(f"{gaussian_crps(sub['obs'], sub[f'{key}_mu'], sub[f'{key}_sigma']).mean():>9.2f}"
                         if len(sub) else f"{'-':>9}")
        cells.append(f"{scores[label]['CRPS']:>9.2f}")
        print(f"{label:<20}" + "".join(cells))
    counts = "".join(f"{int((season == s).sum()):>9}" for s in order)
    print(f"{'n days':<20}" + counts + f"{len(complete):>9}")


if __name__ == "__main__":
    main()
