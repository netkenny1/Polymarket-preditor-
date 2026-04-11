"""Unit tests for polymarket_bot.strategies.btc_5min.

Covers BTC5MinStrategy.compute_fair and the end-to-end generate_signals
edge-detection logic against synthetic Markets + OrderBooks.
"""

from __future__ import annotations

import math

import pytest

from polymarket_bot.config import StrategyConfig
from polymarket_bot.data.models import (
    Market,
    OrderBook,
    OrderBookLevel,
    Side,
    Token,
)
from polymarket_bot.strategies.btc_5min import BTC5MinContext, BTC5MinStrategy


# ── Helpers ─────────────────────────────────────────────────────────

def _make_strategy(**overrides) -> BTC5MinStrategy:
    cfg = StrategyConfig(**overrides) if overrides else StrategyConfig()
    return BTC5MinStrategy(cfg)


def _make_market(
    *,
    strike: float = 60_000.0,
    up_ask: float = 0.50,
    down_ask: float = 0.50,
    start_ts: int = 1_000_000,
    end_ts: int = 1_000_300,
) -> tuple[Market, dict[str, OrderBook]]:
    up_tok = Token(token_id="up_tok", outcome="Up", price=up_ask)
    down_tok = Token(token_id="down_tok", outcome="Down", price=down_ask)
    market = Market(
        condition_id="cond_test",
        slug="btc-updown-5m-1000000",
        question="Will BTC be up or down in the next 5 minutes?",
        tokens=[up_tok, down_tok],
        start_ts=start_ts,
        end_ts=end_ts,
        strike_price=strike,
        active=True,
        liquidity=500.0,
    )
    books = {
        up_tok.token_id: OrderBook(
            token_id=up_tok.token_id,
            bids=[OrderBookLevel(price=max(0.01, up_ask - 0.01), size=500.0)],
            asks=[OrderBookLevel(price=up_ask, size=500.0)],
        ),
        down_tok.token_id: OrderBook(
            token_id=down_tok.token_id,
            bids=[OrderBookLevel(price=max(0.01, down_ask - 0.01), size=500.0)],
            asks=[OrderBookLevel(price=down_ask, size=500.0)],
        ),
    }
    return market, books


def _make_context(
    *,
    spot: float = 60_000.0,
    now_ts: int = 1_000_150,
    vol_annual: float = 0.6,
) -> BTC5MinContext:
    # Build a 1-min close series that yields approximately `vol_annual`
    # when passed through ewma_vol. sigma_per_min = vol_annual / sqrt(525600).
    sigma_per_min = vol_annual / math.sqrt(525_600)
    closes: list[float] = [spot]
    for i in range(120):
        bump = sigma_per_min if i % 2 == 0 else -sigma_per_min
        closes.append(closes[-1] * math.exp(bump))
    return BTC5MinContext(
        btc_spot=spot,
        btc_closes_1m=closes,
        now_ts=now_ts,
        bankroll_usd=100.0,
    )


# ── compute_fair tests ─────────────────────────────────────────────

def test_compute_fair_up_at_money():
    strat = _make_strategy()
    fair = strat.compute_fair(spot=60_000, strike=60_000, seconds_left=150, sigma_annual=0.6)
    assert fair.fair_up == pytest.approx(0.5, abs=0.01)


def test_compute_fair_up_when_spot_above_strike_is_greater_than_half():
    strat = _make_strategy()
    fair = strat.compute_fair(spot=60_100, strike=60_000, seconds_left=150, sigma_annual=0.6)
    assert fair.fair_up > 0.5


def test_compute_fair_up_when_spot_below_strike_is_less_than_half():
    strat = _make_strategy()
    fair = strat.compute_fair(spot=59_900, strike=60_000, seconds_left=150, sigma_annual=0.6)
    assert fair.fair_up < 0.5


# ── generate_signals tests ─────────────────────────────────────────

def test_generate_signals_no_spot_returns_empty():
    strat = _make_strategy()
    market, books = _make_market()
    ctx = _make_context(spot=60_000)
    ctx.btc_spot = 0.0
    assert strat.generate_signals([market], books, ctx) == []


def test_generate_signals_too_early_in_window_returns_empty():
    strat = _make_strategy()
    # end_ts - now_ts > max_seconds_to_expiry (default 270).
    market, books = _make_market(start_ts=1_000_000, end_ts=1_000_300)
    ctx = _make_context(now_ts=1_000_000)  # 300s left > 270
    assert strat.generate_signals([market], books, ctx) == []


def test_generate_signals_too_late_in_window_returns_empty():
    strat = _make_strategy()
    # seconds_left < min_seconds_to_expiry (default 20).
    market, books = _make_market(start_ts=1_000_000, end_ts=1_000_300)
    ctx = _make_context(now_ts=1_000_290)  # 10s left < 20
    assert strat.generate_signals([market], books, ctx) == []


def test_generate_signals_fires_on_big_edge():
    strat = _make_strategy()
    # Spot well above strike -> fair_up near 1; up ask at 0.30 leaves huge edge.
    market, books = _make_market(strike=60_000, up_ask=0.30, down_ask=0.70)
    ctx = _make_context(spot=60_500, now_ts=1_000_150)
    signals = strat.generate_signals([market], books, ctx)
    assert len(signals) == 1
    sig = signals[0]
    assert sig.side == Side.BUY
    assert sig.outcome.lower() == "up"
    assert sig.edge > 0


def test_generate_signals_fires_on_down_side():
    strat = _make_strategy()
    # Spot well below strike -> fair_down near 1; down ask at 0.30 leaves edge.
    market, books = _make_market(strike=60_000, up_ask=0.70, down_ask=0.30)
    ctx = _make_context(spot=59_500, now_ts=1_000_150)
    signals = strat.generate_signals([market], books, ctx)
    assert len(signals) == 1
    sig = signals[0]
    assert sig.side == Side.BUY
    assert sig.outcome.lower() == "down"
    assert sig.edge > 0
