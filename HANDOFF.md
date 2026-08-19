# Handoff

Operational state as of **2026-08-19**. Paste this into a new conversation, or
point a fresh session at this repository — `CLAUDE.md` carries the hard rules and
the merged PRs carry the reasoning behind every decision. What is here is the
part the code cannot tell you: what is running, what is broken, and what is
still unanswered.

## What this project is

A post-processing model for daily maximum temperature at one ASOS station,
emitting a **calibrated probability distribution** rather than a point estimate.
It logs Polymarket's NYC temperature market against that distribution and scores
both. **There is no trading logic anywhere** — no position, no sizing, no
execution — and that is deliberate, not an unfinished feature.

## Current state

Running unattended on an Ubuntu 24.04 VPS at `/opt/weatherbot/app` as user
`weatherbot`, cron `*/30 5-21 * * *` UTC. Deployment steps are in `DEPLOY.md`.

| | |
|---|---|
| Target station | **KLGA** (LaGuardia) — what the market settles on, not KNYC |
| Model | Blend of ECMWF-MOS and post-processed NBM, inverse-variance weighted |
| Skill | **1.33 CRPS** vs NBM's 1.50, walk-forward over 636 days |
| Logging | ~20 snapshots/day × 11 buckets ≈ 220 rows/day |
| Usable days so far | **1** (2026-08-16) |

Thresholds before anything is readable: **30 usable days** for CRPS/Brier/CLV,
**100** before the P&L simulation prints at all.

## Days lost, and why

| Day | Outcome | Cause |
|---|---|---|
| 2026-08-16 | usable — forecast 81.9 F, actual 82.0 F | — |
| 2026-08-17 | **dead**, 297 unpriced rows | NaN forecast bug (fixed) |
| 2026-08-18 | **dead**, zero rows | Open-Meteo published an empty run |

2026-08-17 is unrecoverable: the rows carry real market prices and no
probability. 2026-08-18 produced nothing at all, which is the correct behaviour
now that the guards exist.

## Decisions that should not be relitigated

Each was measured. `CLAUDE.md` carries the full rules; these are the ones most
likely to be second-guessed.

- **Inverse-variance blending, not fitted weights.** A fitted weight scored 1.26
  CRPS against 1.25; four seasonal weights scored 1.26, worse than the fixed
  weight rather than merely indistinguishable.
- **The spread scalar is kept even though it costs CRPS.** PIT 0.93 → 1.02 for
  +0.003 CRPS. The project exists to emit a calibrated distribution; a slightly
  worse forecast that is honest about its uncertainty is the better product.
- **It must be fitted on a held-out tail.** Fitted on the whole training window
  it is identically 1.000, because the Gaussian log-sigma MLE intercept forces
  `mean((r/s)^2) = 1` over the rows it saw.
- **NBM is the bar.** A gain over climatology, persistence and raw ECMWF means
  little without it.
- **KLGA is not KNYC.** +0.55 F mean difference, 1.78 F sd, +1.92 F in summer —
  comparable to the predictive uncertainty itself.
- **Fills are at ask/bid, never mid.** A mid-filled P&L is the standard way a
  backtest lies; the spread on these buckets rivals the edge.

## Known limitations

- **Top-of-book only.** Quotes carry a price and no size, and depth was never
  recorded. The P&L is an upper bound at one share, not achievable at size.
- **Publication lag rests on two samples** — 6.30 h and 7.46 h for the 00Z ECMWF
  run. The 05:00Z cron open has margin below both, but the sample is thin.
- **Empty archive runs.** Open-Meteo sometimes returns HTTP 200 with every value
  null. Treated as a gap; the day is lost.
- **Ensemble data is quarantined.** The endpoint exposes no run identifier, so
  `issue_time` cannot be established and the rows must not be used as features.

## Open questions

1. **How often are archive runs empty?** One in three observed days, far too
   small a sample. If it holds through September, the archive's reliability is
   the limiting factor and a second forecast source is worth discussing — more
   debugging will not help.
2. **Is the 05:00Z window open right?** Revisit once more lag samples land.
   `tools/measure_run_lag.py` collects them; nothing depends on it.
3. **Is there any edge in the market?** Entirely unmeasured. The forecast is
   good; whether Polymarket is mispriced relative to it is a separate question
   needing ~30 days for CLV and ~100 for P&L.

## Watch for

`~/wbstatus.sh` on the VPS prints the dashboard. Three things matter:

- **`rows` ≠ `priced`** — forecasts failing; that day is dead.
- **`snapshots/day` well below 20** — ticks failing or the forecast landing late.
- **rate-limit count above 2** — the archive-gap cache is not holding.

## Bugs found the hard way

Recorded so they are not reintroduced. Several were silent — the failure mode
this project is most exposed to is not a crash but a plausible-looking number.

- Resolving a day before its local day ended, scoring against a partial maximum.
- `walk_forward` training on rows with no observation — one NaN in `ytr` makes
  every ridge coefficient NaN, and the forecast NaN, with no error raised.
- `run_daily` never ingesting observations, so the training target froze while
  scoring kept working via a different endpoint.
- `_dedup_daily` comparing `local_date` as an object, so a `Timestamp` from
  parquet and a `date` from a fresh fetch duplicated instead of replacing.
- Re-requesting six permanently-absent archive runs on every cron pass, ~600
  wasted requests a day, until the API returned 429 and real runs failed too.
- NBM three-digit truncation parsing `102` as `02`, inflating JJA CRPS 1.53 →
  3.62.
- Max-of-hourly used as the target instead of the reported daily max — they
  differ by more than 1 C on 3.6% of days.
