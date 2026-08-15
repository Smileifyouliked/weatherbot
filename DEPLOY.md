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

`run_daily.py` is idempotent within the UTC hour, so scheduling it more often
than hourly is harmless — extra invocations exit 2 without touching anything.
Every 30 minutes gives roughly a dozen useful price snapshots a day while the
market is open.

```bash
sudo -u weatherbot crontab -e
```

```cron
SHELL=/bin/bash
# Public sources only; no secrets. Adjust WEATHERBOT_* only if you moved things.
*/30 * * * * cd /opt/weatherbot/app && flock -n /tmp/weatherbot.lock .venv/bin/python run_daily.py >> /opt/weatherbot/app/logs/cron.log 2>&1
```

`flock -n` is what stops a slow run overlapping the next tick; without it a
first-time cache build would be started several times at once.

Why every 30 minutes rather than once a day: the upstream archives lag the live
market. The Open-Meteo Single Runs API has not published a 00Z run an hour after
it was issued, and NBM bulletins land a few hours late, so the forecast for
today typically becomes available mid-morning US Eastern. Frequent polling picks
it up as soon as it lands and captures price movement through the day.

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
Description=Run weatherbot every 30 minutes

[Timer]
OnCalendar=*:0/30
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
failures directly. Exit 2 (nothing to do this hour) is reported as a failure by
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
