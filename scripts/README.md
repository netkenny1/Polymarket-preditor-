# scripts/

Helper scripts for running `polymarket_bot` on a schedule.

## Scripts

- **`run_paper.sh`** - Activates `.venv` (if present), sets `PAPER_TRADING=true`, and runs the long-lived paper-trading loop. Output is tee'd to `logs/paper-YYYYMMDD.log`.
- **`run_backtest.sh`** - Runs a 7-day backtest with `--show-trades` against real Binance BTC klines. Output is tee'd to `logs/backtest-YYYYMMDD-HHMM.log`.

Both scripts `cd` to the repo root, so they work from any working directory. Env vars (`PRIVATE_KEY`, `POLYMARKET_API_KEY`, `POLYMARKET_API_SECRET`, `POLYMARKET_API_PASSPHRASE`, `POLYMARKET_FUNDER`, ...) are loaded from `.env` via python-dotenv.

Make them executable:

```bash
chmod +x scripts/*.sh
```

## systemd user unit (paper loop)

`~/.config/systemd/user/polymarket-paper.service`:

```ini
[Unit]
Description=Polymarket bot (paper mode)
After=network-online.target

[Service]
Type=simple
ExecStart=/home/user/Polymarket-preditor-/scripts/run_paper.sh
Restart=on-failure
RestartSec=30

[Install]
WantedBy=default.target
```

Start and enable on login:

```bash
systemctl --user daemon-reload
systemctl --user start polymarket-paper.service
systemctl --user enable polymarket-paper.service
journalctl --user -u polymarket-paper.service -f
```

## systemd timer (daily backtest at 02:00)

`~/.config/systemd/user/polymarket-backtest.service`:

```ini
[Unit]
Description=Polymarket bot backtest

[Service]
Type=oneshot
ExecStart=/home/user/Polymarket-preditor-/scripts/run_backtest.sh
```

`~/.config/systemd/user/polymarket-backtest.timer`:

```ini
[Unit]
Description=Run polymarket backtest daily at 02:00

[Timer]
OnCalendar=*-*-* 02:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

Enable with:

```bash
systemctl --user enable --now polymarket-backtest.timer
```

## crontab

Paper mode is a long-running process, so systemd is preferred. Cron is fine for the backtest:

```cron
# Daily backtest at 02:00
0 2 * * * /home/user/Polymarket-preditor-/scripts/run_backtest.sh

# NOT recommended for paper mode (long-running) - use systemd instead:
# */5 * * * * /home/user/Polymarket-preditor-/scripts/run_paper.sh
```

> These unit files and cron lines are templates - adjust the `ExecStart` / path to match your actual repo location.
