"""Check the two parquet caches against the project's hard rules.

FAIL conditions stop the pipeline. WARN conditions are known, accepted states
that must stay visible -- notably the ensemble rows, whose issue_time cannot be
established from the endpoint.

Usage:
    python3 verify_data.py
"""

from __future__ import annotations

import sys

import pandas as pd

import config
import wxio

FAILURES: list[str] = []
WARNINGS: list[str] = []


def hdr(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{label}: {detail}" if detail else label)


def warn(label: str, detail: str = "") -> None:
    print(f"  [WARN] {label}" + (f" -- {detail}" if detail else ""))
    WARNINGS.append(f"{label}: {detail}" if detail else label)


def main() -> int:
    if not config.RAW_HOURLY.exists():
        print(f"FAIL: {config.RAW_HOURLY} does not exist. Run the ingestion modules.")
        return 1
    if not config.DAILY.exists():
        print(f"FAIL: {config.DAILY} does not exist. Run the ingestion modules.")
        return 1

    raw = wxio.read_raw()
    daily = pd.read_parquet(config.DAILY)

    print(f"station : {config.STATION_ICAO}  ({config.LATITUDE}, {config.LONGITUDE})")
    print(f"raw     : {config.RAW_HOURLY}  {len(raw):,} rows  "
          f"{config.RAW_HOURLY.stat().st_size / 1e6:.2f} MB")
    print(f"daily   : {config.DAILY}  {len(daily):,} rows  "
          f"{config.DAILY.stat().st_size / 1e6:.2f} MB")

    # --- inventory -----------------------------------------------------------
    hdr("Inventory by source")
    inv = raw.groupby(["source", "model"], dropna=False, observed=True).agg(
        rows=("value_c", "size"),
        members=("member", lambda s: s.nunique(dropna=True)),
        first_valid=("valid_time", "min"),
        last_valid=("valid_time", "max"),
        confirmed=("issue_time_confirmed", "sum"),
    )
    for (src, model), r in inv.iterrows():
        print(f"  {src:<12} {str(model) or '-':<14} {r.rows:>8,} rows  "
              f"members={r.members:<3} {r.first_valid:%Y-%m-%d} -> {r.last_valid:%Y-%m-%d}  "
              f"issue_time_confirmed={int(r.confirmed):,}")

    # --- hard rule 1 ---------------------------------------------------------
    hdr("Hard rule 1 -- no data timestamped after issue time")
    have = raw["issue_time"].notna()
    late = raw[have & (raw["issue_time"] > raw["valid_time"])]
    check(len(late) == 0, "issue_time <= valid_time for every row carrying an issue_time",
          f"{len(late):,} violations" if len(late) else f"{int(have.sum()):,} rows checked")

    fc = raw[raw["source"] != "asos_obs"]
    missing = fc[fc["issue_time"].isna()]
    if len(missing):
        by = missing.groupby("source", observed=True).size().to_dict()
        warn("forecast rows with NO issue_time -- unusable as rule-1 features",
             f"{len(missing):,} rows {by}")
    else:
        check(True, "every forecast row carries an issue_time")

    obs_prov = raw[(raw["source"] == "asos_obs") & raw["issue_time_confirmed"].fillna(False)]
    check(len(obs_prov) == 0, "observations are not marked as having forecast provenance")

    # --- hard rule 6 ---------------------------------------------------------
    hdr("Hard rule 6 -- UTC internally")
    for col in ("issue_time", "valid_time", "fetched_at"):
        tz = getattr(raw[col].dtype, "tz", None)
        check(str(tz) == "UTC", f"{col} stored tz-aware UTC", f"tz={tz}")

    # --- hard rule 7 ---------------------------------------------------------
    hdr("Hard rule 7 -- no silent fills")
    check(int(raw["value_c"].isna().sum()) == 0, "no null values in raw cache",
          f"{int(raw['value_c'].isna().sum()):,} nulls")
    lo, hi = raw["value_c"].min(), raw["value_c"].max()
    check(-40 < lo and hi < 55, "temperatures physically plausible",
          f"range {lo:.1f}C to {hi:.1f}C")
    dupes = int(raw.duplicated(subset=wxio.IDENTITY).sum())
    check(dupes == 0, "no duplicate rows on the identity key", f"{dupes:,} duplicates")

    # --- provenance ----------------------------------------------------------
    hdr("issue_time provenance")
    conf = raw[raw["issue_time_confirmed"].fillna(False)]
    if len(conf):
        runs = conf["issue_time"].nunique()
        print(f"  confirmed : {len(conf):,} rows across {runs:,} validated runs "
              f"({conf['issue_time'].min():%Y-%m-%d} -> {conf['issue_time'].max():%Y-%m-%d})")
    unconf = fc[~fc["issue_time_confirmed"].fillna(False)]
    if len(unconf):
        print(f"  UNCONFIRMED: {len(unconf):,} forecast rows "
              f"({sorted(unconf['source'].unique())})")
        print("             the ensemble endpoint returns no run identifier, and its")
        print("             past days are spliced across runs. These rows are stored")
        print("             but must not be used as features under hard rule 1.")

    # --- coverage ------------------------------------------------------------
    hdr("Backbone run coverage")
    bb = raw[raw["source"] == "single_runs"]
    if bb.empty:
        check(False, "backbone runs present", "no single_runs rows")
    else:
        runs = pd.Series(sorted(bb["issue_time"].dropna().unique()))
        expected = pd.date_range(runs.min(), runs.max(), freq="D", tz="UTC")
        missing_runs = expected.difference(pd.DatetimeIndex(runs))
        pct = 100 * (1 - len(missing_runs) / max(len(expected), 1))
        print(f"  {len(runs):,} runs, {runs.min():%Y-%m-%d} -> {runs.max():%Y-%m-%d}, "
              f"{len(expected):,} days expected, {pct:.1f}% present")
        if len(missing_runs):
            warn(f"{len(missing_runs)} missing run days",
                 ", ".join(d.strftime("%Y-%m-%d") for d in missing_runs[:8])
                 + (" ..." if len(missing_runs) > 8 else ""))
        leads = bb.groupby("issue_time").size()
        print(f"  hours per run: min {leads.min()}, median {int(leads.median())}, max {leads.max()}")

    # --- observations --------------------------------------------------------
    hdr("Target variable coverage (reported daily max)")
    obs_d = daily[daily["source"] == "asos_daily_official"]
    if obs_d.empty:
        check(False, "observed daily maxima present", "none")
    else:
        d = pd.to_datetime(obs_d["local_date"])
        span = pd.date_range(d.min(), d.max(), freq="D")
        gaps = span.difference(pd.DatetimeIndex(d))
        print(f"  {len(obs_d):,} days, {d.min():%Y-%m-%d} -> {d.max():%Y-%m-%d}, "
              f"{100 * (1 - len(gaps) / len(span)):.1f}% of calendar days")
        if len(gaps):
            warn(f"{len(gaps)} days with no usable observed max",
                 ", ".join(x.strftime("%Y-%m-%d") for x in gaps[:8])
                 + (" ..." if len(gaps) > 8 else ""))

        # QC: the hourly-derived max should track the reported max closely.
        qc = daily[daily["source"] == "asos_obs"][["local_date", "tmax_c"]]
        cmp = obs_d[["local_date", "tmax_c"]].merge(
            qc, on="local_date", how="inner", suffixes=("_rep", "_hourly"))
        if len(cmp):
            diff = cmp["tmax_c_hourly"] - cmp["tmax_c_rep"]
            print(f"  QC vs max-of-hourly over {len(cmp):,} days: "
                  f"mean {diff.mean():+.2f} C, MAE {diff.abs().mean():.2f} C, "
                  f"{int((diff.abs() > 1.0).sum())} days >1.0 C apart")
            check(diff.abs().mean() < 1.0, "hourly-derived max tracks reported max",
                  f"MAE {diff.abs().mean():.2f} C")

    # --- trainable overlap ---------------------------------------------------
    hdr("Trainable set (confirmed provenance only)")
    fc_d = daily[(daily["source"] == "single_runs") & daily["issue_time_confirmed"].fillna(False)]
    if fc_d.empty or obs_d.empty:
        check(False, "forecast/observation overlap exists", "one side empty")
    else:
        merged = fc_d.merge(obs_d[["local_date", "tmax_c"]], on="local_date",
                            how="inner", suffixes=("_fc", "_obs"))
        future = merged[merged["lead_days"] < 0]
        check(len(future) == 0, "no target day precedes its own run",
              f"{len(future)} rows with negative lead")
        print(f"  {len(merged):,} matched (run, target-day) pairs")
        by_lead = merged.groupby("lead_days").size()
        for lead, n in by_lead.head(8).items():
            print(f"    lead +{lead}d : {n:,} pairs")
        d1 = merged[merged["lead_days"] == 1]
        if len(d1):
            err = (d1["tmax_c_fc"] - d1["tmax_c_obs"])
            print(f"  lead +1d raw model bias: {err.mean():+.2f} C, MAE {err.abs().mean():.2f} C "
                  f"(n={len(d1):,})  [sanity only -- not a scoring metric]")

    # --- summary -------------------------------------------------------------
    hdr("Summary")
    print(f"  {len(FAILURES)} failure(s), {len(WARNINGS)} warning(s)")
    for f in FAILURES:
        print(f"    FAIL: {f}")
    for w in WARNINGS:
        print(f"    WARN: {w}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
