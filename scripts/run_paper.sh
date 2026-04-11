#!/usr/bin/env bash
#
# run_paper.sh - Launch the polymarket_bot in paper-trading mode.
#
# Activates the local virtualenv if present, ensures a logs/ directory
# exists, and tees output to a date-stamped logfile. Intended to be
# invoked directly, from systemd, or from cron.

set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

mkdir -p logs

export PAPER_TRADING=true

python -m polymarket_bot paper 2>&1 | tee -a "logs/paper-$(date +%Y%m%d).log"
