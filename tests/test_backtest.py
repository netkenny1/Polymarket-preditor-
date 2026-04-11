"""Unit tests for polymarket_bot.backtest.runner.

Exercises the deterministic backtest harness against synthetic 1-minute
BTCUSDT klines so the test doesn't touch the network.
"""

from __future__ import annotations

import math

import pytest

from polymarket_bot.backtest.runner import BacktestConfig, run_backtest
from polymarket_bot.clients.binance_feed import Kline


# ── Helpers ─────────────────────────────────────────────────────────

def _synthetic_klines(minutes: int = 180, start_ms: int = 1_700_000_000_000) -> list[Kline]:
    """Generate a deterministic sinusoidal 1-minute BTC price series.

    Returns `minutes` Kline objects with open_time_ms at exact UTC minute
    boundaries. Prices oscillate gently around 60_000 USD.
    """
    # Snap the start to a whole-minute boundary aligned to 5-min grid.
    start_ms = (start_ms // 300_000) * 300_000
    klines: list[Kline] = []
    base = 60_000.0
    for i in range(minutes):
        phase = 2.0 * math.pi * (i / 30.0)  # 30-minute period
        mid = base + 50.0 * math.sin(phase)
        high = mid + 5.0
        low = mid - 5.0
        open_px = mid - 1.0
        close = mid + 1.0
        klines.append(
            Kline(
                open_time_ms=start_ms + i * 60_000,
                open=open_px,
                high=high,
                low=low,
                close=close,
                volume=10.0,
            )
        )
    return klines


# ── Tests ───────────────────────────────────────────────────────────

def test_backtest_deterministic_on_synthetic_klines():
    klines = _synthetic_klines(minutes=180)
    cfg = BacktestConfig(days=1, verbose=False)
    result = run_backtest(cfg, klines=klines)
    # 180 minutes / 5-min windows => at most ~36 windows (exact count depends
    # on alignment + vol warm-up). Just assert we got a non-trivial run and
    # that final_capital is a finite float.
    assert result.total_windows > 0
    assert isinstance(result.final_capital, float)
    assert math.isfinite(result.final_capital)
    # Equity curve should start at initial capital.
    assert result.equity_curve[0] == pytest.approx(cfg.initial_capital)


def test_backtest_raises_on_too_few_klines():
    klines = _synthetic_klines(minutes=5)
    with pytest.raises(RuntimeError):
        run_backtest(BacktestConfig(days=1, verbose=False), klines=klines)


def test_backtest_no_trades_when_min_edge_unreachable():
    klines = _synthetic_klines(minutes=180)
    cfg = BacktestConfig(days=1, verbose=False, min_edge=0.99)
    result = run_backtest(cfg, klines=klines)
    assert len(result.trades) == 0
    assert result.final_capital == pytest.approx(cfg.initial_capital)
