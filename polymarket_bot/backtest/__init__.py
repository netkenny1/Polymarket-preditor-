"""Backtesting against real Binance BTC historical data."""

from polymarket_bot.backtest.runner import (
    BacktestConfig,
    BacktestResult,
    BacktestTrade,
    run_backtest,
)

__all__ = ["BacktestConfig", "BacktestResult", "BacktestTrade", "run_backtest"]
