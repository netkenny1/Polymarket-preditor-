# Polymarket Predictor Bot

A Python prediction-market research and trading bot for [Polymarket](https://polymarket.com/). It ingests live market data, runs signal and narrative analysis, applies strategy logic with risk controls, and supports historical backtesting, Monte Carlo simulation, and stress testing.

> This project is for research and paper-trading. It does not provide financial advice.

## Features

- **Live data clients** — streaming Polymarket order-book and trade feeds.
- **Signal engine** — configurable signal generators (mean-reversion, momentum, cross-market, narrative/news).
- **Strategy layer** — pluggable strategies that consume signals and emit orders.
- **Risk module** — position limits, drawdown guards, and fee-aware sizing.
- **Backtesting & simulation** — historical replay, Monte Carlo rollouts, month-long simulations, and stress tests via dedicated entry-points (`run_historical_backtest.py`, `run_monte_carlo.py`, `run_month_sim.py`, `run_realistic_backtest.py`, `run_stress_tests.py`).
- **Execution adapter** — pluggable execution interface for paper or live endpoints.
- **Learning module** — parameter tuning / offline learning hooks.

## Project structure

```
polymarket_bot/
├── backtesting/     # replay + stats harness
├── clients/         # Polymarket / data-source clients
├── config.py        # strategy + risk config
├── data/            # cached market data
├── discovery/       # market discovery / filtering
├── execution/       # order submission adapters
├── learning/        # offline learning / tuning
├── narrative/       # news & narrative signal inputs
├── risk/            # position + drawdown controls
├── signals/         # signal generators
├── strategies/      # strategy orchestration
└── utils/           # shared helpers
tests/               # pytest suite
```

## Quick start

```bash
git clone https://github.com/netkenny1/Polymarket-preditor-.git
cd Polymarket-preditor-
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Historical backtest
python run_historical_backtest.py

# Monte Carlo rollout
python run_monte_carlo.py

# Stress tests
python run_stress_tests.py
```

## Stack

**Python 3.11+**, asyncio, websockets, pandas, numpy, pytest.

## Status

Active research project. The strategy and signal surface is intentionally pluggable — new markets and alpha ideas are added as standalone modules under `signals/` and `strategies/` without touching the execution core.

## Disclaimer

Nothing in this repository constitutes investment advice. Markets carry risk; run everything in paper mode unless you know what you are doing.
