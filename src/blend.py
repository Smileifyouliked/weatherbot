"""Post-processed NBM, and a blend of it with the ECMWF MOS.

Two things:

  NBM-PP    NBM put through the same treatment as ECMWF: its mean is regressed
            on its own forecast plus day-of-year harmonics (which is what
            removes the winter cold bias), and its published spread XND is
            recalibrated against realised errors rather than trusted as-is.
            XND enters the spread model as log(XND), so the fit can rescale and
            reshape it instead of only shifting it.

  blend     ECMWF-MOS and NBM-PP combined. Inverse-variance weighting is the
            default because it fits nothing and, at this sample size, fitted
            weights do not pay for themselves: a single fitted weight and four
            season-specific weights both score worse than it.

Every weight is fitted walk-forward on data strictly before the target day's
issue time (hard rules 1 and 2). The component predictions used to fit a weight
were themselves produced walk-forward, so a weight fitted at day D sees only
out-of-sample component output from days at or before D-2.

A note on the blend's spread. Inverse-variance combination assumes independent
estimates; these two forecasts see the same atmosphere and their errors are
strongly correlated, so combining variances that way understates the true
spread badly. The weights therefore set the *mean* only, and the spread is
refitted by maximum likelihood on the blended residuals.

That refit still leaves the blend slightly underconfident out of sample (PIT
variance ratio 0.93), so one further scalar sharpens it. See
blend_walk_forward for why that scalar has to be fitted on a held-out tail of
the training window rather than on all of it.

Usage:
    python3 src/blend.py [--months 12]
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
from baselines import (  # noqa: E402
    THRESHOLDS_F, _Phi, build as build_baselines, exceedance_prob,
    gaussian_crps, load as load_daily, score as score_row,
)
from mos import (  # noqa: E402
    FEATURES_MEAN, LEAD, MIN_TRAIN_ROWS, ROLLING_DAYS, _standardize,
    build_features, fit_log_sigma, load_nbm, predict_sigma, select_features,
    walk_forward,
)

SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
ORDER = ["DJF", "MAM", "JJA", "SON"]

NBM_FEATURES = ["nbm_tmax", "sin_doy", "cos_doy"]
NBM_SIGMA = ["log_xnd", "sin_doy", "cos_doy"]
BLEND_SIGMA = ["log_ivsigma", "sin_doy", "cos_doy"]

MIN_BLEND_TRAIN = 120        # days before a blend weight is fitted at all
MIN_SEASON_TRAIN = 45        # days of a season before it gets its own weight
NOISE_FLOOR_CRPS = 0.05      # below this, a seasonal gain is not a gain

# Inverse-variance weighting is the default: it fits nothing, and at this sample
# size fitted weights do not pay for themselves (see CLAUDE.md rule 5a).
DEFAULT_MODE = "invvar"

# Whether the default blend carries the spread scalar is a property of the
# station, not of the code: it is measured, and the two stations disagree.
# KNYC uninflated sits at PIT 0.93 and inflating moves it to 1.02, worth +0.003
# CRPS. KLGA uninflated sits at 0.95 and inflating overshoots to 1.06, further
# from 1.00. See config.Station.inflate and CLAUDE.md rule 5b.
def default_inflate() -> bool:
    return config.STATIONS[config.STATION_ICAO].inflate

# Held-out tail of the training window used to fit the spread scalar.
HOLDOUT_FRAC = 0.25
MIN_HOLDOUT_FIT = 60
MIN_HOLDOUT_EVAL = 30
# A calibration fix is not worth paying for beyond this in CRPS.
INFLATION_COST_LIMIT = 0.02


# --- components --------------------------------------------------------------

def nbm_frame(obs: pd.Series) -> pd.DataFrame:
    """NBM forecast, its published spread, and harmonics, joined to the target."""
    nbm = load_nbm()
    if nbm.empty:
        raise SystemExit("no NBM rows cached; run ingest_nbm.py first")

    df = pd.DataFrame({"nbm_tmax": nbm["mu"], "xnd": nbm["sigma"]}, index=nbm.index)
    df = df.join(obs.rename("obs"), how="inner").dropna()
    doy = df.index.dayofyear.values
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    # XND is published in whole degrees F and is occasionally floored, so the
    # log is taken of a clipped value rather than of zero.
    df["log_xnd"] = np.log(df["xnd"].clip(lower=0.5))
    return df.sort_index()


# --- blending ----------------------------------------------------------------

def fit_weight(y: np.ndarray, mu_a: np.ndarray, mu_b: np.ndarray) -> float:
    """Least-squares weight on mu_a in w*mu_a + (1-w)*mu_b, clipped to [0, 1].

    Clipping keeps the blend a genuine interpolation of the two forecasts. An
    unclipped weight can extrapolate beyond both when they are highly
    correlated, which fits the training window and generalises badly.
    """
    d = mu_a - mu_b
    denom = float(d @ d)
    if denom < 1e-9:
        return 0.5
    return float(np.clip((d @ (y - mu_b)) / denom, 0.0, 1.0))


def blend_walk_forward(comp: pd.DataFrame, targets: pd.DatetimeIndex,
                       mode: str, inflate: bool = False) -> pd.DataFrame:
    """Blend two component forecasts, refitting weights for every target day.

    mode: 'invvar'   inverse-variance weights, nothing fitted
          'fixed'    one weight fitted on the training window
          'seasonal' one weight per season, fitted on that season's training days

    `inflate` applies one extra scalar to the predictive spread: c > 1 widens,
    c < 1 sharpens. It is fitted on a held-out tail of the training window, not
    on the whole of it.

    That split is not incidental. The Gaussian log-sigma MLE carries an
    intercept, so its first-order condition forces mean((r/s)^2) == 1 over the
    rows it was fitted on. A scalar estimated from those same rows is therefore
    identically 1.000 and cannot move calibration at all -- it is exactly
    redundant with the intercept. Fitting it on residuals the spread model did
    not see is what makes it a real parameter, and it is still fitted only on
    data at or before the cutoff.
    """
    rows = []
    for D in targets:
        cutoff = D - timedelta(days=2)
        train = comp.loc[:cutoff]
        if D not in comp.index or len(train) < MIN_BLEND_TRAIN:
            continue
        here = comp.loc[D]

        # Inverse-variance weight on the MOS component, computed per day from
        # the two predictive spreads. Used directly in 'invvar' mode and as the
        # fallback elsewhere.
        iv_w = (1.0 / here["sigma_mos"] ** 2) / (
            1.0 / here["sigma_mos"] ** 2 + 1.0 / here["sigma_nbm"] ** 2)

        if mode == "invvar":
            w = float(iv_w)
        elif mode == "fixed":
            w = fit_weight(train["obs"].to_numpy(float),
                           train["mu_mos"].to_numpy(float),
                           train["mu_nbm"].to_numpy(float))
        elif mode == "seasonal":
            season = SEASONS[D.month]
            sub = train[train.index.month.map(SEASONS) == season]
            if len(sub) >= MIN_SEASON_TRAIN:
                w = fit_weight(sub["obs"].to_numpy(float),
                               sub["mu_mos"].to_numpy(float),
                               sub["mu_nbm"].to_numpy(float))
            else:
                # Not enough of this season seen yet: fall back to the pooled
                # weight rather than fitting four parameters on a handful of days.
                w = fit_weight(train["obs"].to_numpy(float),
                               train["mu_mos"].to_numpy(float),
                               train["mu_nbm"].to_numpy(float))
        else:
            raise ValueError(mode)

        mu = w * float(here["mu_mos"]) + (1.0 - w) * float(here["mu_nbm"])

        # Spread is refitted on blended residuals rather than combined
        # analytically: the components are strongly correlated, so an
        # inverse-variance variance would be far too tight.
        tr_w = w
        resid = (train["obs"].to_numpy(float)
                 - (tr_w * train["mu_mos"].to_numpy(float)
                    + (1.0 - tr_w) * train["mu_nbm"].to_numpy(float)))
        Ztr = train[BLEND_SIGMA].to_numpy(float)
        Zte = comp.loc[[D], BLEND_SIGMA].to_numpy(float)
        Ztr_s, Zte_s = _standardize(Ztr, Zte)
        gamma = fit_log_sigma(Ztr_s, resid)
        sigma = float(predict_sigma(gamma, Zte_s)[0])

        c = 1.0
        if inflate:
            k = int(len(train) * (1.0 - HOLDOUT_FRAC))
            if k >= MIN_HOLDOUT_FIT and len(train) - k >= MIN_HOLDOUT_EVAL:
                Za, Zb = _standardize(Ztr[:k], Ztr[k:])
                g_early = fit_log_sigma(Za, resid[:k])
                s_held = predict_sigma(g_early, Zb)
                c = float(np.sqrt(np.mean((resid[k:] / s_held) ** 2)))
                sigma *= c

        rows.append({"local_date": D, "mu": mu, "sigma": sigma,
                     "obs": float(here["obs"]), "w": w, "iv_w": float(iv_w),
                     "c": c, "n_train": len(train)})
    return pd.DataFrame(rows).set_index("local_date")


# --- scoring -----------------------------------------------------------------

def score(df: pd.DataFrame) -> dict:
    y, mu, sg = (df["obs"].to_numpy(float), df["mu"].to_numpy(float),
                 df["sigma"].to_numpy(float))
    out = {"CRPS": float(gaussian_crps(y, mu, sg).mean()),
           "MAE": float(np.abs(y - mu).mean()),
           "PIT": float(np.var(_Phi((y - mu) / sg)) * 12.0)}
    for t in THRESHOLDS_F:
        out[f"B{t}"] = float(np.mean((exceedance_prob(mu, sg, t)
                                      - (y > t).astype(float)) ** 2))
    return out


def baseline_row(base: pd.DataFrame, key: str) -> dict:
    s = score_row(base, key)
    y = base["obs"].to_numpy(float)
    mu, sg = base[f"{key}_mu"].to_numpy(float), base[f"{key}_sigma"].to_numpy(float)
    s["PIT"] = float(np.var(_Phi((y - mu) / sg)) * 12.0)
    return s


def as_frame(base: pd.DataFrame, key: str) -> pd.DataFrame:
    return pd.DataFrame({"obs": base["obs"], "mu": base[f"{key}_mu"],
                         "sigma": base[f"{key}_sigma"]})


def main() -> None:
    ap = argparse.ArgumentParser()
    config.add_station_arg(ap)
    ap.add_argument("--months", type=int, default=12)
    args = ap.parse_args()
    config.use_station(args.station)

    obs, fc_daily = load_daily(LEAD)
    ecmwf = build_features().join(obs.rename("obs"), how="inner").dropna(
        subset=FEATURES_MEAN + ["obs"])
    nbm = nbm_frame(obs)

    end = min(obs.index.max(), fc_daily.index.max())
    start = end - timedelta(days=round(args.months * 30.44))
    targets = obs.index[(obs.index >= start) & (obs.index <= end)]

    # Components are run over their whole usable span, not just the evaluation
    # window, so the blend has out-of-sample component output to fit weights on
    # before the window opens.
    feature_set = select_features(ecmwf, targets.min())
    print()
    all_ec = ecmwf.index[ecmwf.index >= ecmwf.index.min()]
    all_nbm = nbm.index

    mos = walk_forward(ecmwf, all_ec, feature_set, "rolling")
    nbm_pp = walk_forward(nbm, all_nbm, NBM_FEATURES, "rolling",
                          sigma_features=NBM_SIGMA)

    comp = pd.DataFrame({
        "obs": mos["obs"],
        "mu_mos": mos["mu"], "sigma_mos": mos["sigma"],
    }).join(pd.DataFrame({"mu_nbm": nbm_pp["mu"], "sigma_nbm": nbm_pp["sigma"]}),
            how="inner")
    iv_var = 1.0 / (1.0 / comp["sigma_mos"] ** 2 + 1.0 / comp["sigma_nbm"] ** 2)
    comp["log_ivsigma"] = np.log(np.sqrt(iv_var))
    doy = comp.index.dayofyear.values
    comp["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    comp["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

    # Both variants are always computed, whichever the station ships, so the
    # diagnostic below compares a real pair rather than a row against itself.
    bl_ivc = blend_walk_forward(comp, targets, DEFAULT_MODE, inflate=True)
    bl_iv = blend_walk_forward(comp, targets, DEFAULT_MODE, inflate=False)
    bl_default = bl_ivc if default_inflate() else bl_iv
    bl_fx = blend_walk_forward(comp, targets, "fixed")
    bl_se = blend_walk_forward(comp, targets, "seasonal")
    base = build_baselines(obs, fc_daily, targets)

    common = base.dropna(subset=["obs", "raw_mu", "raw_sigma", "clim_mu",
                                 "clim_sigma", "pers_mu", "pers_sigma"]).index
    for frame in (mos, nbm_pp, bl_iv, bl_ivc, bl_default, bl_fx, bl_se):
        common = common.intersection(frame.index)
    nbm_raw = nbm.loc[nbm.index.intersection(common)]
    common = common.intersection(nbm_raw.index)

    base_c = base.loc[common]
    rows = [
        ("MOS rolling", mos.loc[common]),
        ("NBM raw", pd.DataFrame({"obs": nbm.loc[common, "obs"],
                                  "mu": nbm.loc[common, "nbm_tmax"],
                                  "sigma": nbm.loc[common, "xnd"].clip(lower=0.5)})),
        ("NBM post-processed", nbm_pp.loc[common]),
        (f"blend (default, inflation {'on' if default_inflate() else 'off'})",
         bl_default.loc[common]),
        ("  variant: + inflation", bl_ivc.loc[common]),
        ("  variant: no inflation", bl_iv.loc[common]),
        ("blend fixed", bl_fx.loc[common]),
        ("blend seasonal", bl_se.loc[common]),
        ("raw deterministic", as_frame(base_c, "raw")),
        ("climatology", as_frame(base_c, "clim")),
        ("persistence", as_frame(base_c, "pers")),
    ]
    scored = [(label, score(frame)) for label, frame in rows]

    print(f"station          : {config.STATION_ICAO}   lead +{LEAD}d")
    print(f"EVALUATION WINDOW: {common.min():%Y-%m-%d} -> {common.max():%Y-%m-%d}"
          f"   n = {len(common)} days")
    print(f"component spans  : MOS {mos.index.min():%Y-%m-%d}+, "
          f"NBM-PP {nbm_pp.index.min():%Y-%m-%d}+")
    print(f"blend training   : {int(bl_fx['n_train'].min())}-"
          f"{int(bl_fx['n_train'].max())} days before each target")

    # --- main table ---
    print("\nAll units degrees F. Brier at exceedance thresholds. "
          "PIT = var(PIT)/uniform; 1.00 is calibrated.")
    head = (f"{'model':<24}{'CRPS':>7}{'MAE':>7}{'PIT':>7}  "
            + "".join(f"{'B'+str(t):>7}" for t in THRESHOLDS_F))
    print("\n" + head)
    print("-" * len(head))
    for label, s in scored:
        print(f"{label:<24}{s['CRPS']:>7.2f}{s['MAE']:>7.2f}{s['PIT']:>7.2f}  "
              + "".join(f"{s['B'+str(t)]:>7.3f}" for t in THRESHOLDS_F))

    # --- seasonal ---
    season = common.month.map(SEASONS)
    print("\nCRPS by season (degrees F)")
    head2 = f"{'model':<24}" + "".join(f"{s:>9}" for s in ORDER) + f"{'all':>9}"
    print("\n" + head2)
    print("-" * len(head2))
    for label, frame in rows:
        cells = []
        for s in ORDER:
            m = np.asarray(season == s)
            cells.append(f"{score(frame[m])['CRPS']:>9.2f}" if m.sum() else f"{'-':>9}")
        print(f"{label:<24}" + "".join(cells) + f"{score(frame)['CRPS']:>9.2f}")
    print(f"{'n days':<24}" + "".join(f"{int((season == s).sum()):>9}" for s in ORDER)
          + f"{len(common):>9}")

    # --- weights ---
    print("\nFitted blend weights (share on ECMWF-MOS; 1.00 = MOS only)")
    print(f"\n{'mode':<24}" + "".join(f"{s:>9}" for s in ORDER) + f"{'all':>9}")
    print("-" * (24 + 9 * 5))
    for label, frame in (("inverse-variance", bl_iv), ("fixed", bl_fx),
                         ("seasonal", bl_se)):
        f = frame.loc[common]
        cells = [f"{f['w'][np.asarray(season == s)].mean():>9.2f}"
                 if (season == s).sum() else f"{'-':>9}" for s in ORDER]
        print(f"{label:<24}" + "".join(cells) + f"{f['w'].mean():>9.2f}")

    # --- overfitting verdict ---
    fx, se = score(bl_fx.loc[common])["CRPS"], score(bl_se.loc[common])["CRPS"]
    gain = fx - se
    print(f"\nSeasonal vs fixed weighting")
    print(f"  fixed    CRPS {fx:.3f}")
    print(f"  seasonal CRPS {se:.3f}")
    print(f"  gain     {gain:+.3f} F  (noise floor {NOISE_FLOOR_CRPS:.2f})")
    if gain < NOISE_FLOOR_CRPS:
        print(f"  VERDICT: noise. Seasonal weighting adds 4 parameters fitted on "
              f"~{len(common) // 4} days each")
        print(f"           and does not clear the floor. Prefer the fixed weight.")
    else:
        print(f"  VERDICT: seasonal weighting clears the floor.")

    # --- spread inflation ---
    before, after = score(bl_iv.loc[common]), score(bl_ivc.loc[common])
    cost = after["CRPS"] - before["CRPS"]
    state = "ON" if default_inflate() else "OFF"
    print(f"\nSpread inflation (one scalar, {state} by default for "
          f"{config.STATION_ICAO})")
    print(f"  before   CRPS {before['CRPS']:.3f}   PIT {before['PIT']:.3f}")
    print(f"  after    CRPS {after['CRPS']:.3f}   PIT {after['PIT']:.3f}")
    print(f"  cost     {cost:+.3f} F CRPS  (limit {INFLATION_COST_LIMIT:.2f})")
    print(f"  mean c   {bl_ivc.loc[common, 'c'].mean():.3f}   "
          f"(<1 sharpens, >1 widens)")
    print(f"  shipped  {'with' if default_inflate() else 'without'} the scalar "
          f"for {config.STATION_ICAO} (config.Station.inflate)")
    if abs(after["PIT"] - 1.0) < abs(before["PIT"] - 1.0) and cost <= INFLATION_COST_LIMIT:
        print(f"  VERDICT: on. Calibration improves and the CRPS cost is inside "
              f"the limit.")
    elif cost > INFLATION_COST_LIMIT:
        print(f"  VERDICT: drop. The CRPS cost exceeds the limit.")
    else:
        print(f"  VERDICT: drop. Calibration does not improve.")


if __name__ == "__main__":
    main()
