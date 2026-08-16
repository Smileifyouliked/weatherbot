"""Nonhomogeneous Gaussian regression MOS for daily max temperature, lead +1d.

The predictive distribution is Gaussian with both parameters regressed on
forecast predictors:

    mu        = ridge regression on forecast fields and day-of-year harmonics
    log sigma = linear in day-of-year harmonics, cloud cover and wind,
                fitted by Gaussian maximum likelihood

Modelling sigma is the whole point: there is no ensemble spread for this
station (the Open-Meteo ensemble archive is ~4 days deep), so uncertainty has
to be learned from the predictors instead of read off a spread.

Sample size drives every design choice here. There are ~874 usable training
rows, so the model is deliberately small: a ridge penalty tuned only inside the
training window, backward feature selection run once on pre-evaluation data,
and no interactions.

Walk-forward (hard rule 2): for target day D the model is refitted on rows whose
target day is D-2 or earlier. D-1 is excluded because the lead +1d forecast for D
is the 00Z run on D, issued 20:00 local on D-1, when local day D-1 is still in
progress and its max is not yet known.

Usage:
    python3 src/mos.py [--months 12] [--check]
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import wxio  # noqa: E402
from baselines import (  # noqa: E402
    SEASONS, THRESHOLDS_F, build as build_baselines, c_to_f,
    exceedance_prob, gaussian_crps, load as load_daily, score as score_row,
)

LEAD = 1
FEATURES_MEAN = ["fc_tmax", "cloud", "wind", "dewpt", "depression", "sin_doy", "cos_doy"]
FEATURES_SIGMA = ["sin_doy", "cos_doy", "cloud", "wind"]
ALPHA_GRID = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
SIGMA_RIDGE = 1.0
MIN_TRAIN_ROWS = 120
ROLLING_DAYS = 365
PIT_BINS = 20
REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"


# --- features ----------------------------------------------------------------

def load_nbm() -> pd.DataFrame:
    """NBM day-ahead max and its published standard deviation, in degrees F.

    Same issue time and lead as the ECMWF backbone (00Z cycle on the target
    local day), so it drops straight into the comparison.
    """
    daily = pd.read_parquet(config.DAILY)
    n = daily[daily["source"] == "nbm"]
    if n.empty:
        return pd.DataFrame(columns=["mu", "sigma"]).rename_axis("local_date")
    out = pd.DataFrame({
        "mu": c_to_f(n["tmax_c"].to_numpy(float)),
        # A standard deviation is a difference, so it scales by 9/5 alone.
        "sigma": n["tmax_sigma_c"].to_numpy(float) * 9.0 / 5.0,
    }, index=pd.to_datetime(n["local_date"]).values)
    out.index.name = "local_date"
    return out.sort_index()


def build_features() -> pd.DataFrame:
    """Daily predictors per target day, from the lead +1d run only.

    Aggregates are taken over the station-local day, matching how the target
    daily max is defined. Rule 1 holds by construction: every hour used comes
    from a single run whose issue_time precedes all of them.
    """
    raw = wxio.read_raw()
    fc = raw[(raw["source"] == "single_runs") & raw["issue_time_confirmed"].fillna(False)]
    if fc.empty:
        raise SystemExit("no confirmed-provenance forecast rows in the raw cache")

    fc = fc.copy()
    local = fc["valid_time"].dt.tz_convert(config.STATION_TZ)
    fc["local_date"] = pd.to_datetime(local.dt.date)
    issue_local = fc["issue_time"].dt.tz_convert(config.STATION_TZ)
    fc["lead_days"] = (fc["local_date"] - pd.to_datetime(issue_local.dt.date)).dt.days
    fc = fc[fc["lead_days"] == LEAD]

    wide = fc.pivot_table(index=["issue_time", "local_date", "valid_time"],
                          columns="variable", values="value", aggfunc="first").reset_index()

    needed = set(config.FORECAST_VARIABLES)
    if not needed <= set(wide.columns):
        missing = sorted(needed - set(wide.columns))
        raise SystemExit(
            f"raw cache is missing predictor(s) {missing} -- rerun ingest_forecast.py")

    # Hourly dewpoint depression, then averaged: pairing daily-mean T with
    # daily-mean Td would mix quantities from different hours.
    wide["depression_c"] = wide["temperature_2m"] - wide["dew_point_2m"]

    g = wide.groupby(["issue_time", "local_date"], observed=True)
    feats = g.agg(
        fc_tmax_c=("temperature_2m", "max"),
        dewpt_c=("dew_point_2m", "mean"),
        depression_c=("depression_c", "mean"),
        cloud=("cloud_cover", "mean"),
        wind=("wind_speed_10m", "mean"),
        n_hours=("valid_time", "count"),
    ).reset_index()

    feats = feats[feats["n_hours"] >= config.MIN_HOURS_PER_DAY].copy()

    feats["fc_tmax"] = c_to_f(feats["fc_tmax_c"])
    feats["dewpt"] = c_to_f(feats["dewpt_c"])
    feats["depression"] = feats["depression_c"] * 9.0 / 5.0   # a difference, not a level
    doy = feats["local_date"].dt.dayofyear.values
    feats["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    feats["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

    return feats.set_index("local_date").sort_index()


# --- fitting -----------------------------------------------------------------

def _standardize(train: np.ndarray, other: np.ndarray):
    mu, sd = train.mean(axis=0), train.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (train - mu) / sd, (other - mu) / sd


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float):
    """Closed-form ridge on standardized X; intercept is the training mean."""
    n, p = X.shape
    A = X.T @ X + alpha * np.eye(p)
    b = X.T @ (y - y.mean())
    return np.linalg.solve(A, b), y.mean()


def tune_alpha(X: np.ndarray, y: np.ndarray) -> float:
    """Pick alpha on a time-ordered inner split of the training window only.

    The validation slice is the tail of the training data, never anything at or
    after the target day, so tuning cannot see the evaluation period.
    """
    cut = int(len(X) * 0.75)
    if cut < 40 or len(X) - cut < 20:
        return 10.0
    Xtr, Xva = _standardize(X[:cut], X[cut:])
    best, best_err = ALPHA_GRID[0], np.inf
    for a in ALPHA_GRID:
        beta, b0 = ridge_fit(Xtr, y[:cut], a)
        err = np.sqrt(np.mean((y[cut:] - (Xva @ beta + b0)) ** 2))
        if err < best_err:
            best, best_err = a, err
    return best


def fit_log_sigma(Z: np.ndarray, resid: np.ndarray, ridge: float = SIGMA_RIDGE):
    """Gaussian MLE for log sigma = intercept + Z @ gamma, by Newton steps.

    Minimises sum(log sigma_i + r_i^2 / (2 sigma_i^2)) with a ridge penalty on
    the slopes. The Hessian is positive definite, so a handful of steps converge.
    """
    n, p = Z.shape
    D = np.hstack([np.ones((n, 1)), Z])
    gamma = np.zeros(p + 1)
    gamma[0] = np.log(max(resid.std(ddof=1), 1e-3))
    pen = ridge * np.eye(p + 1)
    pen[0, 0] = 0.0  # never penalise the intercept

    r2 = resid ** 2
    for _ in range(50):
        s2 = np.exp(2.0 * (D @ gamma))
        w = r2 / s2
        grad = D.T @ (1.0 - w) + pen @ gamma
        H = 2.0 * (D * w[:, None]).T @ D + pen
        step = np.linalg.solve(H + 1e-8 * np.eye(p + 1), grad)
        gamma -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    return gamma


def predict_sigma(gamma: np.ndarray, Z: np.ndarray) -> np.ndarray:
    D = np.hstack([np.ones((len(Z), 1)), Z])
    return np.exp(np.clip(D @ gamma, np.log(0.3), np.log(40.0)))


# --- walk-forward ------------------------------------------------------------

def walk_forward(data: pd.DataFrame, targets: pd.DatetimeIndex,
                 features: list[str], window: str,
                 sigma_features: list[str] | None = None,
                 keep_coefficients: bool = False) -> pd.DataFrame:
    """Refit for every target day. `window` is 'expanding' or 'rolling'.

    `sigma_features` overrides FEATURES_SIGMA, so an ablation can restrict the
    spread model to the same predictors as its mean model. With
    `keep_coefficients`, each row also carries the fitted mean coefficients,
    both standardised (comparable across predictors) and in natural units
    (degrees F per unit of the raw predictor).
    """
    sigma_features = sigma_features or FEATURES_SIGMA
    rows = []
    for D in targets:
        cutoff = D - timedelta(days=2)
        train = data.loc[:cutoff]
        if window == "rolling":
            train = train.loc[train.index > cutoff - timedelta(days=ROLLING_DAYS)]
        if len(train) < MIN_TRAIN_ROWS or D not in data.index:
            continue

        Xtr = train[features].to_numpy(float)
        ytr = train["obs"].to_numpy(float)
        Xte = data.loc[[D], features].to_numpy(float)

        alpha = tune_alpha(Xtr, ytr)
        Xtr_s, Xte_s = _standardize(Xtr, Xte)
        beta, b0 = ridge_fit(Xtr_s, ytr, alpha)
        mu = float((Xte_s @ beta + b0)[0])

        resid = ytr - (Xtr_s @ beta + b0)
        Ztr = train[sigma_features].to_numpy(float)
        Zte = data.loc[[D], sigma_features].to_numpy(float)
        Ztr_s, Zte_s = _standardize(Ztr, Zte)
        gamma = fit_log_sigma(Ztr_s, resid)
        sigma = float(predict_sigma(gamma, Zte_s)[0])

        row = {"local_date": D, "mu": mu, "sigma": sigma,
               "obs": float(data.loc[D, "obs"]), "n_train": len(train),
               "alpha": alpha}
        if keep_coefficients:
            sd = Xtr.std(axis=0, ddof=0)
            sd = np.where(sd < 1e-12, 1.0, sd)
            for name, b, s in zip(features, beta, sd):
                row[f"std__{name}"] = b        # per standard deviation
                row[f"nat__{name}"] = b / s    # per raw unit
        rows.append(row)
    return pd.DataFrame(rows).set_index("local_date")


def select_features(data: pd.DataFrame, eval_start: pd.Timestamp) -> list[str]:
    """Backward elimination on pre-evaluation data only.

    Uses a time-ordered split entirely before the evaluation window, so the
    evaluation period plays no part in choosing the feature set.
    """
    pre = data.loc[:eval_start - timedelta(days=3)]
    if len(pre) < MIN_TRAIN_ROWS:
        return list(FEATURES_MEAN)

    cut = int(len(pre) * 0.75)
    ytr, yva = pre["obs"].to_numpy(float)[:cut], pre["obs"].to_numpy(float)[cut:]

    def val_rmse(feats: list[str]) -> float:
        X = pre[feats].to_numpy(float)
        Xtr, Xva = _standardize(X[:cut], X[cut:])
        beta, b0 = ridge_fit(Xtr, ytr, tune_alpha(X[:cut], ytr))
        return float(np.sqrt(np.mean((yva - (Xva @ beta + b0)) ** 2)))

    keep = list(FEATURES_MEAN)
    best = val_rmse(keep)
    dropped = []
    while len(keep) > 2:
        scores = [(val_rmse([f for f in keep if f != c]), c) for c in keep]
        score_wo, cand = min(scores)
        if score_wo <= best - 1e-4:      # removing it genuinely helps
            keep = [f for f in keep if f != cand]
            dropped.append(cand)
            best = score_wo
        else:
            break

    print(f"feature selection (pre-eval data only, n={len(pre)}):")
    print(f"  kept    : {', '.join(keep)}")
    print(f"  dropped : {', '.join(dropped) if dropped else '(none)'}")
    print(f"  holdout RMSE {best:.3f} F")
    return keep


# --- PIT ---------------------------------------------------------------------

def pit_values(df: pd.DataFrame) -> np.ndarray:
    from baselines import _Phi
    return _Phi((df["obs"].to_numpy(float) - df["mu"].to_numpy(float))
                / df["sigma"].to_numpy(float))


def diagnose_pit(p: np.ndarray) -> tuple[str, float]:
    """Classify calibration from the spread of PIT values.

    Var(PIT) is 1/12 for a calibrated forecast. Larger means probability mass
    piled at 0 and 1 (U-shaped, overconfident); smaller means piled in the
    middle (hump-shaped, underconfident).
    """
    var = float(np.var(p))
    ref = 1.0 / 12.0
    ratio = var / ref
    if ratio > 1.10:
        return "U-shaped (overconfident -- sigma too small)", ratio
    if ratio < 0.90:
        return "hump-shaped (underconfident -- sigma too large)", ratio
    return "flat (calibrated)", ratio


def plot_pit(panels: list[tuple[str, np.ndarray]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6), sharey=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (name, p) in zip(axes, panels):
        ax.hist(p, bins=PIT_BINS, range=(0, 1), color="#4878a8", edgecolor="white")
        ax.axhline(len(p) / PIT_BINS, color="#c04040", ls="--", lw=1.2, label="uniform")
        shape, ratio = diagnose_pit(p)
        ax.set_title(f"{name}\n{shape.split(' (')[0]}  (var/ref {ratio:.2f})", fontsize=9)
        ax.set_xlabel("PIT")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("count")
    fig.suptitle(f"PIT histograms, {config.STATION_ICAO}, lead +{LEAD}d", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --- leakage check -----------------------------------------------------------

def check_no_leakage(data: pd.DataFrame, targets: pd.DatetimeIndex,
                     features: list[str], sample: int = 12) -> None:
    """Refit sampled days against a truncated history and require a match.

    Covers the whole MOS pipeline -- feature standardisation, alpha tuning,
    ridge fit and the sigma MLE -- not just the baseline statistics.
    """
    rng = np.random.default_rng(0)
    picks = sorted(rng.choice(len(targets), size=min(sample, len(targets)), replace=False))
    full = walk_forward(data, targets, features, "expanding")

    bad = []
    for i in picks:
        D = targets[i]
        if D not in full.index:
            continue
        cutoff = D - timedelta(days=2)
        # Keep row D itself (its predictors are known at issue time) but hide
        # everything between the cutoff and D.
        trunc = data.loc[(data.index <= cutoff) | (data.index == D)]
        one = walk_forward(trunc, pd.DatetimeIndex([D]), features, "expanding")
        if D not in one.index:
            continue
        a = full.loc[D, ["mu", "sigma"]].to_numpy(float)
        b = one.loc[D, ["mu", "sigma"]].to_numpy(float)
        if not np.allclose(a, b, rtol=0, atol=1e-10):
            bad.append((D, a, b))

    if bad:
        for D, a, b in bad[:3]:
            print(f"  LEAK {D:%Y-%m-%d} full mu/sigma={a} truncated={b}")
        raise SystemExit(f"MOS leakage check failed on {len(bad)}/{len(picks)} days")
    print(f"leakage check: {len(picks)} sampled days identical when the history "
          f"between cutoff and target is hidden")


# --- main --------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    config.add_station_arg(ap)
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    config.use_station(args.station)

    obs, fc_daily = load_daily(LEAD)
    feats = build_features()
    nbm = load_nbm()

    data = feats.join(obs.rename("obs"), how="inner").dropna(
        subset=FEATURES_MEAN + ["obs"])

    end = min(obs.index.max(), fc_daily.index.max())
    start = end - timedelta(days=round(args.months * 30.44))
    targets = obs.index[(obs.index >= start) & (obs.index <= end)]

    if args.check:
        check_no_leakage(data, targets[-60:], FEATURES_MEAN)
        return

    print(f"station : {config.STATION_ICAO}   lead +{LEAD}d")
    print(f"rows    : {len(data)} usable (features + target), "
          f"{data.index.min():%Y-%m-%d} -> {data.index.max():%Y-%m-%d}")
    print()

    features = select_features(data, targets.min())
    print()

    mos_exp = walk_forward(data, targets, features, "expanding")
    mos_rol = walk_forward(data, targets, features, "rolling")
    base = build_baselines(obs, fc_daily, targets)

    # Score every row on the same days, so the table compares like with like.
    common = base.dropna(subset=["obs", "raw_mu", "raw_sigma", "clim_mu",
                                 "clim_sigma", "pers_mu", "pers_sigma"]).index
    common = common.intersection(mos_exp.index).intersection(mos_rol.index)
    nbm_ok = nbm.dropna(subset=["mu", "sigma"]).index
    dropped_for_nbm = len(common.difference(nbm_ok))
    common = common.intersection(nbm_ok)
    base_c, exp_c, rol_c = base.loc[common], mos_exp.loc[common], mos_rol.loc[common]
    nbm_c = nbm.loc[common].assign(obs=base_c["obs"].to_numpy(float))

    print(f"evaluation: {common.min():%Y-%m-%d} -> {common.max():%Y-%m-%d}, "
          f"{len(common)} days common to every row"
          + (f" ({dropped_for_nbm} dropped for missing NBM)" if dropped_for_nbm else ""))
    print(f"training  : expanding {exp_c['n_train'].min()}-{exp_c['n_train'].max()} rows, "
          f"rolling {rol_c['n_train'].min()}-{rol_c['n_train'].max()} rows")
    print(f"ridge alpha: expanding median {np.median(exp_c['alpha']):g}, "
          f"rolling median {np.median(rol_c['alpha']):g}")

    def mos_scores(df):
        y, mu, sg = df["obs"].values, df["mu"].values, df["sigma"].values
        out = {"CRPS": gaussian_crps(y, mu, sg).mean(), "MAE": np.abs(y - mu).mean()}
        for t in THRESHOLDS_F:
            out[f"B{t}"] = np.mean((exceedance_prob(mu, sg, t) - (y > t).astype(float)) ** 2)
        return out

    rows = [
        ("MOS expanding", mos_scores(exp_c)),
        ("MOS rolling", mos_scores(rol_c)),
        ("NBM", mos_scores(nbm_c)),
        ("raw deterministic", score_row(base_c, "raw")),
        ("climatology", score_row(base_c, "clim")),
        ("persistence", score_row(base_c, "pers")),
    ]

    print("\nAll units degrees F. Brier at exceedance thresholds (P(Tmax > t)).")
    head = f"{'model':<20}{'CRPS':>7}{'MAE':>7}" + "".join(f"{'B'+str(t):>7}" for t in THRESHOLDS_F)
    print("\n" + head)
    print("-" * len(head))
    for label, s in rows:
        print(f"{label:<20}{s['CRPS']:>7.2f}{s['MAE']:>7.2f}"
              + "".join(f"{s['B'+str(t)]:>7.3f}" for t in THRESHOLDS_F))

    # --- seasonal CRPS ---
    season = common.month.map(SEASONS)
    order = ["DJF", "MAM", "JJA", "SON"]
    print("\nCRPS by season (degrees F, lower is better)")
    head2 = f"{'model':<20}" + "".join(f"{s:>9}" for s in order) + f"{'all':>9}"
    print("\n" + head2)
    print("-" * len(head2))

    series = {
        "MOS expanding": (exp_c["obs"], exp_c["mu"], exp_c["sigma"]),
        "MOS rolling": (rol_c["obs"], rol_c["mu"], rol_c["sigma"]),
        "NBM": (nbm_c["obs"], nbm_c["mu"], nbm_c["sigma"]),
        "raw deterministic": (base_c["obs"], base_c["raw_mu"], base_c["raw_sigma"]),
        "climatology": (base_c["obs"], base_c["clim_mu"], base_c["clim_sigma"]),
        "persistence": (base_c["obs"], base_c["pers_mu"], base_c["pers_sigma"]),
    }
    for label, (y, mu, sg) in series.items():
        cells = []
        for s in order:
            m = season == s
            cells.append(f"{gaussian_crps(y[m], mu[m], sg[m]).mean():>9.2f}"
                         if m.sum() else f"{'-':>9}")
        cells.append(f"{gaussian_crps(y, mu, sg).mean():>9.2f}")
        print(f"{label:<20}" + "".join(cells))
    print(f"{'n days':<20}" + "".join(f"{int((season == s).sum()):>9}" for s in order)
          + f"{len(common):>9}")

    # --- PIT ---
    panels = [("MOS expanding", pit_values(exp_c)), ("MOS rolling", pit_values(rol_c)),
              ("NBM", pit_values(nbm_c))]
    raw_pit = pd.DataFrame({"obs": base_c["obs"], "mu": base_c["raw_mu"],
                            "sigma": base_c["raw_sigma"]})
    panels.append(("raw deterministic", pit_values(raw_pit)))

    out = REPORT_DIR / f"pit_lead{LEAD}.png"
    plot_pit(panels, out)
    print("\nPIT calibration")
    for name, p in panels:
        shape, ratio = diagnose_pit(p)
        print(f"  {name:<20} var/uniform {ratio:.2f}  ->  {shape}")
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
