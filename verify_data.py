"""Report on what is actually in the ingestion caches.

Prints, for each cache: row count, date coverage, null counts per column, and the
five largest gaps in each series.

This is a report, not a repair. It never writes, never fills, and exits non-zero
if a cache is missing or a series cannot be read, so a broken ingest fails here
loudly instead of quietly becoming a smaller dataset (rule 7).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
OBS_PATH = REPO_ROOT / "data" / "obs.parquet"
FCST_PATH = REPO_ROOT / "data" / "fcst.parquet"

TOP_GAPS = 5


def _rule(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def _nulls(frame: pd.DataFrame) -> None:
    print("null counts per column:")
    counts = frame.isna().sum()
    width = max(len(str(c)) for c in frame.columns)
    for column in frame.columns:
        count = int(counts[column])
        share = count / len(frame) * 100 if len(frame) else 0.0
        print(f"  {str(column):<{width}}  {count:>8}  ({share:5.2f}%)")


def _gaps(series: pd.Series, label: str, unit: str) -> None:
    """The `TOP_GAPS` largest jumps between consecutive values of a sorted series."""
    ordered = pd.Series(sorted(pd.Series(series).dropna().unique()))
    print(f"{TOP_GAPS} largest gaps in {label}:")
    if len(ordered) < 2:
        print(f"  not enough distinct values ({len(ordered)}) to measure a gap")
        return

    deltas = ordered.diff().iloc[1:]
    if unit == "days":
        sizes = deltas.dt.days if hasattr(deltas, "dt") else deltas
    else:
        sizes = deltas.dt.total_seconds() / 3600.0

    ranked = sizes.sort_values(ascending=False).head(TOP_GAPS)
    for position, size in ranked.items():
        after = ordered.iloc[position - 1]
        before = ordered.iloc[position]
        # For a daily series the count of absent days is unambiguous. For an
        # hourly one the expected cadence is a property of the model, not of this
        # frame, so the raw delta is reported without guessing what is "missing".
        note = f"  ({size - 1:.0f} absent)" if unit == "days" else ""
        print(f"  {size:>8.0f} {unit}  after {after}  next {before}{note}")


def _coverage(dates: pd.Series, label: str) -> None:
    ordered = pd.Series(sorted(pd.Series(dates).dropna().unique()))
    if ordered.empty:
        print(f"{label}: empty")
        return
    first, last = ordered.iloc[0], ordered.iloc[-1]
    span = (pd.Timestamp(last) - pd.Timestamp(first)).days + 1
    present = len(ordered)
    print(f"{label}: {first} to {last}")
    print(
        f"  {present} distinct of {span} calendar days "
        f"({present / span * 100:.2f}% present, {span - present} absent)"
    )


def verify_obs() -> bool:
    _rule(f"OBS  {OBS_PATH.relative_to(REPO_ROOT)}")
    if not OBS_PATH.exists():
        print("MISSING: no observation cache. Run `python src/obs.py` first.")
        return False

    frame = pd.read_parquet(OBS_PATH)
    print(f"rows: {len(frame)}")
    print(f"columns: {list(frame.columns)}")
    if frame.empty:
        print("EMPTY: cache exists but holds no rows.")
        return False

    _coverage(frame["local_date"], "date coverage (station-local)")
    if "sparse" in frame:
        sparse = int(frame["sparse"].sum())
        print(f"  {sparse} days flagged sparse (below the daily observation floor)")
    print()
    _nulls(frame)
    print()
    _gaps(frame["local_date"], "daily max series (local_date)", "days")
    return True


def verify_fcst() -> bool:
    _rule(f"FCST  {FCST_PATH.relative_to(REPO_ROOT)}")
    if not FCST_PATH.exists():
        print("MISSING: no forecast cache. Run `python src/forecast.py --start ...` first.")
        return False

    frame = pd.read_parquet(FCST_PATH)
    print(f"rows: {len(frame)}")
    print(f"columns: {list(frame.columns)}")
    if frame.empty:
        print("EMPTY: cache exists but holds no rows.")
        return False

    print(f"models: {sorted(frame['model'].unique())}")
    print(f"variables: {sorted(frame['variable'].unique())}")
    print(f"members: {frame['member'].nunique()} ({frame['member'].min()}..{frame['member'].max()})")
    print(f"cycles: {frame['issue_time'].nunique()}")
    print(f"lead hours: {frame['lead_hours'].min():.0f} to {frame['lead_hours'].max():.0f}")

    leaks = int((frame["issue_time"] > frame["valid_time"]).sum())
    print(f"rule 1 check: {leaks} rows with issue_time after valid_time")

    _coverage(frame["issue_time"].dt.date, "issue_time coverage")
    _coverage(frame["valid_time"].dt.date, "valid_time coverage")
    print()
    _nulls(frame)
    print()
    _gaps(frame["issue_time"], "forecast cycles (issue_time)", "hours")

    models = sorted(frame["model"].unique())
    if len(models) > 1:
        for model in models:
            subset = frame.loc[frame["model"] == model]
            print()
            _gaps(subset["issue_time"], f"cycles for model {model}", "hours")
    return True


def main() -> int:
    print("weatherbot data verification")
    print(f"repo: {REPO_ROOT}")

    results = {"obs": verify_obs(), "fcst": verify_fcst()}

    _rule("SUMMARY")
    for name, ok in results.items():
        print(f"  {name:<5} {'ok' if ok else 'FAILED'}")

    if not all(results.values()):
        print()
        print("Verification failed. Nothing was filled in or worked around.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
