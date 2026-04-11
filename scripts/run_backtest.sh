#!/usr/bin/env bash
#
# run_backtest.sh - Run a 7-day backtest against real Binance BTC klines.
#
# Activates the local virtualenv if present, ensures a logs/ directory
# exists, and tees output to a timestamped logfile. Intended for ad-hoc
# use or scheduled execution via systemd timer / cron.

set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

mkdir -p logs

python -m polymarket_bot backtest --days 7 --show-trades 2>&1 | tee -a "logs/backtest-$(date +%Y%m%d-%H%M).log"
