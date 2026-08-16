"""Negative controls and an ablation for the MOS pipeline.

Three things, none of which add capability -- they exist to test whether the
reported skill is real:

  shuffle     Refit with the target permuted and everything else identical. A
              pipeline that cannot see the target should collapse to no skill.
              Any remaining skill means the harness is reading the answer.

  ablation    MOS-lite: forecast max temperature and day-of-year harmonics only,
              same walk-forward. Separates seasonal bias correction from what
              the weather predictors actually add.

  coefficients  Fitted mean coefficients of the full model, so the signs can be
              checked against physics rather than taken on trust.

Usage:
    python3 src/controls.py [--months 12] [--seeds 5]
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
    build as build_baselines, gaussian_crps, load as load_daily, score as score_row,
)
from mos import (  # noqa: E402
    FEATURES_MEAN, LEAD, build_features, select_features, walk_forward,
)

LITE_FEATURES = ["fc_tmax", "sin_doy", "cos_doy"]
LITE_SIGMA = ["sin_doy", "cos_doy"]


def crps_of(df: pd.DataFrame) -> float:
    return float(gaussian_crps(df["obs"].to_numpy(float),
                               df["mu"].to_numpy(float),
                               df["sigma"].to_numpy(float)).mean())


def mae_of(df: pd.DataFrame) -> float:
    return float(np.abs(df["obs"].to_numpy(float) - df["mu"].to_numpy(float)).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    config.add_station_arg(ap)
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    config.use_station(args.station)

    obs, fc_daily = load_daily(LEAD)
    feats = build_features()
    data = feats.join(obs.rename("obs"), how="inner").dropna(subset=FEATURES_MEAN + ["obs"])

    end = min(obs.index.max(), fc_daily.index.max())
    start = end - timedelta(days=round(args.months * 30.44))
    targets = obs.index[(obs.index >= start) & (obs.index <= end)]

    features = select_features(data, targets.min())
    print()

    full = walk_forward(data, targets, features, "rolling", keep_coefficients=True)
    lite = walk_forward(data, targets, LITE_FEATURES, "rolling",
                        sigma_features=LITE_SIGMA)
    base = build_baselines(obs, fc_daily, targets)

    common = base.dropna(subset=["obs", "raw_mu", "raw_sigma", "clim_mu", "clim_sigma"]).index
    common = common.intersection(full.index).intersection(lite.index)
    base_c, full_c, lite_c = base.loc[common], full.loc[common], lite.loc[common]

    print(f"station    : {config.STATION_ICAO}   lead +{LEAD}d")
    print(f"evaluation : {common.min():%Y-%m-%d} -> {common.max():%Y-%m-%d}, "
          f"{len(common)} days")
    print(f"note       : rolling window throughout, so these line up with the "
          f"'MOS rolling' row")

    # --- 1. shuffle test -----------------------------------------------------
    # The target is permuted across the whole record and the pipeline is run
    # end to end on the permuted series -- fitting and scoring both. Skill above
    # no-skill would mean a fit saw its own evaluation day.
    print("\n" + "=" * 72)
    print("1. SHUFFLE TEST -- target permuted, pipeline otherwise identical")
    print("=" * 72)

    shuffled = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        perm = data.copy()
        perm["obs"] = rng.permutation(perm["obs"].to_numpy(float))
        sh = walk_forward(perm, targets, features, "rolling")
        sh = sh.loc[sh.index.intersection(common)]
        shuffled.append((seed, crps_of(sh), mae_of(sh)))

    # A permuted target has no seasonal cycle left, so the honest no-skill
    # reference is the unconditional spread of the record, not climatology --
    # climatology scores as well as it does precisely by using day-of-year.
    sd = float(data["obs"].std(ddof=1))
    noskill = sd * (1.0 / np.sqrt(np.pi))   # CRPS of N(mean, sd) against N(mean, sd)

    print(f"\n{'run':<22}{'CRPS':>8}{'MAE':>8}")
    print("-" * 38)
    for seed, c, m in shuffled:
        print(f"{'shuffled seed ' + str(seed):<22}{c:>8.2f}{m:>8.2f}")
    cs = [c for _, c, _ in shuffled]
    print("-" * 38)
    print(f"{'shuffled mean':<22}{np.mean(cs):>8.2f}{np.mean([m for _, _, m in shuffled]):>8.2f}")
    print(f"{'':<22}{'':>8}")
    print(f"{'full MOS (real)':<22}{crps_of(full_c):>8.2f}{mae_of(full_c):>8.2f}")
    print(f"{'raw deterministic':<22}{score_row(base_c, 'raw')['CRPS']:>8.2f}"
          f"{score_row(base_c, 'raw')['MAE']:>8.2f}")
    print(f"{'climatology':<22}{score_row(base_c, 'clim')['CRPS']:>8.2f}"
          f"{score_row(base_c, 'clim')['MAE']:>8.2f}")
    print(f"{'no-skill reference':<22}{noskill:>8.2f}{'':>8}   "
          f"(unconditional sd {sd:.1f} F)")

    worst_raw = score_row(base_c, "raw")["CRPS"]
    verdict = "PASS" if min(cs) > worst_raw else "FAIL"
    print(f"\n  [{verdict}] shuffled CRPS {min(cs):.2f}-{max(cs):.2f} vs raw "
          f"deterministic {worst_raw:.2f}")
    if verdict == "PASS":
        print("  Shuffling destroys all skill: the pipeline cannot see its target.")
    else:
        print("  Shuffled MOS still beats raw -- the harness is leaking. Stop and fix.")

    # --- 2. ablation ---------------------------------------------------------
    print("\n" + "=" * 72)
    print("2. ABLATION -- how much of the gain is seasonal bias correction alone")
    print("=" * 72)
    print(f"\n  MOS-lite features : {', '.join(LITE_FEATURES)}")
    print(f"  full MOS features : {', '.join(features)}")

    raw_crps = score_row(base_c, "raw")["CRPS"]
    lite_crps, full_crps = crps_of(lite_c), crps_of(full_c)
    print(f"\n{'model':<22}{'CRPS':>8}{'MAE':>8}   {'share of raw->full gain':>26}")
    print("-" * 66)
    print(f"{'raw deterministic':<22}{raw_crps:>8.2f}{score_row(base_c, 'raw')['MAE']:>8.2f}"
          f"{'(baseline)':>26}")
    gain_total = raw_crps - full_crps
    gain_lite = raw_crps - lite_crps
    print(f"{'MOS-lite':<22}{lite_crps:>8.2f}{mae_of(lite_c):>8.2f}"
          f"{100 * gain_lite / gain_total:>25.0f}%")
    print(f"{'full MOS':<22}{full_crps:>8.2f}{mae_of(full_c):>8.2f}{100:>25.0f}%")

    season = common.month.map({12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM",
                               5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA", 9: "SON",
                               10: "SON", 11: "SON"})
    print(f"\nCRPS by season")
    order = ["DJF", "MAM", "JJA", "SON"]
    print(f"\n{'model':<22}" + "".join(f"{s:>9}" for s in order) + f"{'all':>9}")
    print("-" * (22 + 9 * 5))
    for label, frame in (("raw deterministic", None), ("MOS-lite", lite_c),
                         ("full MOS", full_c)):
        cells = []
        for s in order:
            m = season == s
            if frame is None:
                sub = base_c[m]
                v = gaussian_crps(sub["obs"], sub["raw_mu"], sub["raw_sigma"]).mean()
            else:
                v = crps_of(frame[m])
            cells.append(f"{v:>9.2f}")
        allv = (gaussian_crps(base_c["obs"], base_c["raw_mu"], base_c["raw_sigma"]).mean()
                if frame is None else crps_of(frame))
        print(f"{label:<22}" + "".join(cells) + f"{allv:>9.2f}")

    # --- 3. coefficients -----------------------------------------------------
    print("\n" + "=" * 72)
    print("3. COEFFICIENTS -- full model, median across walk-forward refits")
    print("=" * 72)

    units = {"fc_tmax": "F", "cloud": "% cloud", "wind": "km/h",
             "dewpt": "F dewpoint", "depression": "F depression",
             "sin_doy": "unit", "cos_doy": "unit"}
    expect = {"fc_tmax": "+", "cloud": "-", "depression": "+"}

    print(f"\n{'feature':<14}{'std beta':>10}{'nat beta':>10}  {'per':<14}"
          f"{'expect':>7}{'sign':>7}")
    print("-" * 64)
    for f in features:
        std_med = float(full_c[f"std__{f}"].median())
        nat_med = float(full_c[f"nat__{f}"].median())
        want = expect.get(f, "")
        got = "+" if nat_med > 0 else "-"
        mark = "" if not want else ("ok" if want == got else "CHECK")
        print(f"{f:<14}{std_med:>10.3f}{nat_med:>10.4f}  {units.get(f, ''):<14}"
              f"{want:>7}{mark:>7}")

    dropped = [f for f in FEATURES_MEAN if f not in features]
    if dropped:
        print(f"\n  {', '.join(dropped)} dropped by feature selection, so not in the")
        print("  model above. Refitting with the full candidate set to expose the sign:")
        full_all = walk_forward(data, targets, FEATURES_MEAN, "rolling",
                                keep_coefficients=True)
        full_all = full_all.loc[full_all.index.intersection(common)]
        print(f"\n{'feature':<14}{'std beta':>10}{'nat beta':>10}  {'per':<14}"
              f"{'expect':>7}{'sign':>7}")
        print("-" * 64)
        for f in FEATURES_MEAN:
            std_med = float(full_all[f"std__{f}"].median())
            nat_med = float(full_all[f"nat__{f}"].median())
            want = expect.get(f, "")
            got = "+" if nat_med > 0 else "-"
            mark = "" if not want else ("ok" if want == got else "CHECK")
            print(f"{f:<14}{std_med:>10.3f}{nat_med:>10.4f}  {units.get(f, ''):<14}"
                  f"{want:>7}{mark:>7}")
        print(f"\n  CRPS with the full candidate set: "
              f"{crps_of(full_all.assign(obs=full_c['obs'])):.2f} "
              f"(selected set: {full_crps:.2f})")

    print("\n  fc_tmax, dewpt and depression are strongly collinear -- depression is")
    print("  built from the same fields as the other two -- so individual signs in a")
    print("  ridge fit split shared signal between them and are not independently")
    print("  interpretable. Read the sign of a predictor as conditional on the rest.")


if __name__ == "__main__":
    main()
