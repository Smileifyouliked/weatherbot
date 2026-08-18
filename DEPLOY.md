# Deploying on Ubuntu 24.04

Unattended operation of `run_daily.py`: refresh forecasts, build the predictive
distribution, snapshot open Polymarket prices against it, backfill outcomes.

Logging only. Nothing in this repository places, sizes or recommends a trade.

**No credentials are required.** Every source used — Open-Meteo, NOAA's public
S3 bucket, Iowa Environmental Mesonet, the Polymarket Gamma API — is public and
unauthenticated. There is no API key to install, and nothing reads one. If that
ever changes, the value belongs in the systemd unit's `Environment=` or the
cron environment, never in a file in the repository.

---

## 1. Prerequisites

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git ca-certificates tzdata
```

Ubuntu 24.04 ships Python 3.12. The pins in `requirements.txt` were verified on
Python 3.11.15; they install and import cleanly on 3.12 as well. Check what you
have:

```bash
python3 --version
```

Ubuntu 24.04 marks the system Python as externally managed (PEP 668), so
`pip install` outside a virtualenv will refuse to run. The venv below is not
optional.

## 2. Install

```bash
sudo useradd --system --create-home --home-dir /opt/weatherbot --shell /usr/sbin/nologin weatherbot
sudo -u weatherbot -H bash -lc '
  cd /opt/weatherbot &&
  git clone https://github.com/Smileifyouliked/weatherbot.git app &&
  cd app &&
  python3 -m venv .venv &&
  .venv/bin/pip install --upgrade pip &&
  .venv/bin/pip install -r requirements.txt
'
```

Verify:

```bash
sudo -u weatherbot -H /opt/weatherbot/app/.venv/bin/python -c \
  "import pandas, pyarrow, numpy, requests; print('deps ok')"
```

## 3. First run: build the caches

The repository ships a committed snapshot of `data/raw_hourly.parquet` and
`data/daily.parquet`, so the models run immediately. Bring them up to the
present before scheduling anything:

```bash
sudo -u weatherbot -H bash -lc '
  cd /opt/weatherbot/app &&
  PATH=".venv/bin:$PATH" ./make_cache.sh
'
```

Expect **3–5 minutes** from the shipped snapshot. From an empty `data/`
(`./make_cache.sh --fresh`) expect **45–55 minutes** — the ECMWF Single Runs
archive serves roughly 0.4–0.5 runs per second and some runs hang server-side.

The script exits non-zero if any stage fails, and finishes by running
`verify_data.py`, which must report `0 failure(s)`. Two warnings are expected
and are not faults: ensemble rows carry no `issue_time`, and six backbone runs
are permanently absent from the archive.

## 4. Schedule

```bash
sudo -u weatherbot crontab -e
```

```cron
SHELL=/bin/bash
# Public sources only; no secrets. Adjust WEATHERBOT_* only if you moved things.
# Hours are UTC, deliberately -- see "the polling window" below.
*/30 5-21 * * * cd /opt/weatherbot/app && flock -n /tmp/weatherbot.lock .venv/bin/python run_daily.py >> /opt/weatherbot/app/logs/cron.log 2>&1
```

34 invocations a day, of which roughly 21 produce a usable snapshot. `flock -n`
is what stops a slow run overlapping the next tick; without it a first-time
cache build would be started several times at once.

**Do not raise `WEATHERBOT_MIN_RUN_INTERVAL_MIN` above the cron interval.**
`run_daily.py` exits 2 if the last snapshot is newer than that minimum, which
exists to absorb a double-fire or an overlap. Set it at or above 30 and it stops
being a safety floor and becomes a cap on cadence — silently, since the skipped
runs look identical to legitimate ones in the log. The default of 25 minutes
sits just under the 30-minute tick.

### The polling window

The market settles on the maximum over a **local** calendar day at KLGA, so the
window is bounded at both ends, and both ends were measured rather than guessed.

**It opens when the 00Z ECMWF run becomes retrievable.** This is the binding
constraint and it is the reason the window does not start earlier. Measured
directly by `tools/measure_run_lag.py`, which polls a pending run every five
minutes and records first sight:

| run | first seen | lag | |
|---|---|---|---|
| 2026-08-15 00Z | 07:27Z | 7.46 h | measured (2 polls pending) |
| 2026-08-16 00Z | 06:17Z | 6.30 h | measured (51 polls pending) |

So **06:15Z–07:30Z on the evidence available**, mean 6.88 h. Opening the cron at
**05:00Z** leaves roughly 1.3 h of margin below the earlier of the two, which is
worth having on a two-sample base. Runs before the forecast lands are cheap
no-ops: `predict` reports "not yet available" and nothing is logged.

NBM's 00Z bulletin is present by about 03:00Z, well ahead of ECMWF, so it never
binds.

**It closes at the seasonal cutoff** in `PEAK_LOCAL_HOUR_BY_SEASON`
(`src/clv.py`) — DJF 15:00, SON 15:30, MAM and JJA 16:00, station-local. Past
that the market is pricing a known answer and `clv.py log` refuses to record it.
Because the cutoffs are local and DST moves underneath them, the close lands on
a different UTC hour through the year:

| season | cutoff (local) | close (UTC) |
|---|---|---|
| DJF | 15:00 EST | 20:00Z |
| MAM | 16:00 EST (to ~Mar 8) | **21:00Z** ← binding |
| MAM | 16:00 EDT | 20:00Z |
| JJA | 16:00 EDT | 20:00Z |
| SON | 15:30 EDT (Sep–Oct) | 19:30Z |
| SON | 15:30 EST (late Nov) | 20:30Z |

**Early March binds at 21:00Z**, when 16:00 local is still EST. The cron runs to
`21`, so its last tick is 21:30Z and every season's close is covered.

**The cron is in UTC on purpose.** The seasonal cutoffs already handle DST
inside the code; a local-time cron would shift the whole window twice a year on
top of that, for no gain.

Ticks outside the useful window are not harmful, just wasted — each one either
finds no forecast yet or is refused by the cutoff guard.

### Why every 30 minutes, and not hourly

Cadence sets how much of the window is actually captured. Simulated over 2,783
complete KLGA days at the measured 6.88 h lag (`python3 src/clv.py cutoffs`):

| cadence | clean rows | post-peak | leak % | clean/day |
|---|---|---|---|---|
| every 30 min | 60,227 | 14,811 | 19.7% | 21.6 |
| hourly | 30,380 | 8,212 | 21.3% | 10.9 |

The leak *rate* barely moves — it is a ratio of times within a fixed window —
but the volume halves. Post-peak rows are flagged `excluded` and never scored,
so they cost storage rather than accuracy.

### systemd timer instead of cron

```ini
# /etc/systemd/system/weatherbot.service
[Unit]
Description=weatherbot daily pass
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=weatherbot
WorkingDirectory=/opt/weatherbot/app
ExecStart=/opt/weatherbot/app/.venv/bin/python run_daily.py
# Any credential, if one is ever needed, goes here -- not in the repo.
# Environment=EXAMPLE_TOKEN=...
```

```ini
# /etc/systemd/system/weatherbot.timer
[Unit]
Description=Run weatherbot every 30 minutes across the polling window

[Timer]
# UTC, matching the cron above: 05:00-21:30, every 30 minutes.
OnCalendar=*-*-* 05..21:00,30:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now weatherbot.timer
systemctl list-timers weatherbot.timer
```

systemd records the exit code, so `systemctl status weatherbot.service` shows
failures directly. Exit 2 (last snapshot too recent) is reported as a failure by
systemd's default `SuccessExitStatus`; add `SuccessExitStatus=0 2` to the
`[Service]` section to treat it as success.

## 5. Exit codes

| Code | Meaning | Action |
|---|---|---|
| 0 | All stages succeeded or were documented no-ops | none |
| 1 | At least one stage failed | check the log |
| 2 | Skipped: this UTC hour already has a snapshot | none, expected |
| 130 | Interrupted | none |

A stage reported as `no-op` means an upstream source has not published yet. That
is normal and self-correcting; the next run backfills.

## 6. Logs

`run_daily.py` appends to `logs/run_daily-YYYY-MM-DD.log`, timestamped in UTC,
and echoes the same lines to stdout for cron or journald.

```bash
sudo install -d -o weatherbot -g weatherbot /opt/weatherbot/app/logs
tail -f /opt/weatherbot/app/logs/run_daily-$(date -u +%F).log
```

Rotate them:

```bash
sudo tee /etc/logrotate.d/weatherbot >/dev/null <<'EOF'
/opt/weatherbot/app/logs/*.log {
    weekly
    rotate 8
    compress
    delaycompress
    missingok
    notifempty
    su weatherbot weatherbot
}
EOF
```

## 7. Back up `data/`

**This is the part that is easy to get wrong.** `data/clv_log.parquet` and
`data/clv_pred.parquet` are runtime output and are deliberately **not** in git.
The CLV log is the only record of what the market was quoting at the moment we
looked; it cannot be reconstructed after the fact from any public source.
Everything else under `data/` can be rebuilt with `./make_cache.sh`.

```bash
sudo tee /etc/cron.daily/weatherbot-backup >/dev/null <<'EOF'
#!/bin/sh
set -e
d=/opt/weatherbot/backups
mkdir -p "$d"
tar -czf "$d/clv-$(date -u +%F).tar.gz" -C /opt/weatherbot/app/data \
    clv_log.parquet clv_pred.parquet 2>/dev/null || exit 0
find "$d" -name 'clv-*.tar.gz' -mtime +90 -delete
EOF
sudo chmod +x /etc/cron.daily/weatherbot-backup
```

## 8. Environment variables

All optional. Configuration is environment-only; there is no config file to
edit outside the repository.

| Variable | Default | Purpose |
|---|---|---|
| `WEATHERBOT_DATA_DIR` | `<repo>/data` | Where the parquet caches and CLV log live |
| `WEATHERBOT_LOG_DIR` | `<repo>/logs` | Where the timestamped run log is written |
| `WEATHERBOT_STATION` | `KNYC` | Default station for modules run without `--station` (the CLV job always uses KLGA regardless) |
| `WEATHERBOT_DETERMINED_QUANTILE` | `0.75` | Same-day settlement guard: stop logging once the realised max exceeds this quantile of our forecast. Must be in (0.5, 1.0); see `src/clv.py cutoffs --override` for the trade |
| `WEATHERBOT_MIN_RUN_INTERVAL_MIN` | `25` | Minimum minutes between snapshots; a run finds one newer than this and exits 2. Must stay **below** the cron interval or it caps cadence |
| `HTTPS_PROXY` | unset | Standard `requests` proxy support, if egress is proxied |

## 9. Checking it works

```bash
cd /opt/weatherbot/app

# One pass, ignoring the idempotence guard.
sudo -u weatherbot -H .venv/bin/python run_daily.py --force; echo "exit=$?"

# What has been logged, and whether anything has resolved.
sudo -u weatherbot -H .venv/bin/python src/clv.py resolve
sudo -u weatherbot -H .venv/bin/python src/clv.py score
```

`score` will refuse to report until there is something meaningful to report.
With one resolved day the eleven buckets within it are near-perfectly dependent,
so the effective sample size is about one, not eleven. Expect several weeks
before the Brier comparison and CLV mean anything.

### The Open-Meteo daily request budget

The Single Runs archive has a daily request limit, and a `*/30` cron makes 34
passes a day against it. The thing that used to blow the budget was runs the
archive simply does not have: six holes in the ECMWF record were re-attempted on
every pass, three attempts each, forever — about 600 requests a day spent asking
for data that does not exist. On the live host that produced real `429 Daily API
request limit exceeded` responses, which is worse than wasteful: once the budget
is gone, the run that *does* matter cannot be fetched either.

Runs the server reports as absent are now recorded in `data/<ICAO>/archive_gaps.parquet`
and skipped once they are more than `GAP_SETTLED_DAYS` (7) old. Newer absences
are always retried, because today's run is absent by definition for the first
several hours and blacklisting it would mean never fetching it.

Only *gaps* are recorded, never transport failures. A timeout or a 429 may well
be hiding a run that is really there, and writing those off would lose real data.

To force a recheck of everything:

```bash
sudo -u weatherbot -H bash -c 'cd /opt/weatherbot/app && \
  .venv/bin/python ingest_forecast.py --station KLGA --skip-ensemble --retry-gaps'
```

Watch for budget exhaustion with:

```bash
sudo grep -icE "429|request limit" /opt/weatherbot/app/logs/cron.log
```

Anything above zero means the quota is being hit and the cause needs finding —
it should now be rare.

## 10. Updating

```bash
sudo -u weatherbot -H bash -lc '
  cd /opt/weatherbot/app &&
  git pull --ff-only &&
  .venv/bin/pip install -r requirements.txt &&
  PATH=".venv/bin:$PATH" ./make_cache.sh
'
```

`data/` is gitignored apart from the two committed snapshots, so a pull will not
clobber the CLV log. If a pull ever reports a conflict on
`data/raw_hourly.parquet` or `data/daily.parquet`, take the repository's version
(`git checkout --theirs`) and re-run `./make_cache.sh`; both are rebuildable.
