"""Unit tests for polymarket_bot.pricing.black_scholes.

Covers digital call/put probabilities, the seconds->years helper, and the
realized / EWMA volatility estimators used by the BTC 5-min strategy.
"""

from __future__ import annotations

import math

import pytest

from polymarket_bot.pricing.black_scholes import (
    MINUTES_PER_YEAR,
    SECONDS_PER_YEAR,
    digital_call_prob,
    digital_put_prob,
    ewma_vol,
    realized_vol,
    years_from_seconds,
)


def test_digital_call_prob_deep_itm_returns_near_one():
    p = digital_call_prob(S=110_000, K=100_000, T=1 / MINUTES_PER_YEAR, sigma=0.6)
    assert p == pytest.approx(1.0, abs=1e-6)


def test_digital_call_prob_deep_otm_returns_near_zero():
    p = digital_call_prob(S=90_000, K=100_000, T=1 / MINUTES_PER_YEAR, sigma=0.6)
    assert p == pytest.approx(0.0, abs=1e-6)


def test_digital_call_prob_at_money_is_half():
    p = digital_call_prob(S=100_000, K=100_000, T=5 / MINUTES_PER_YEAR, sigma=0.6)
    assert p == pytest.approx(0.5, abs=0.01)


def test_digital_call_prob_degenerate_time():
    assert digital_call_prob(S=101, K=100, T=0, sigma=0.6) == 1.0
    assert digital_call_prob(S=99, K=100, T=0, sigma=0.6) == 0.0
    assert digital_call_prob(S=100, K=100, T=0, sigma=0.6) == 0.5


def test_digital_call_and_put_sum_to_one():
    S, K, T, sigma = 101_234.5, 100_000.0, 180 / SECONDS_PER_YEAR, 0.72
    call = digital_call_prob(S, K, T, sigma)
    put = digital_put_prob(S, K, T, sigma)
    assert call + put == pytest.approx(1.0, abs=1e-12)


def test_years_from_seconds_five_min():
    assert years_from_seconds(300) == pytest.approx(300 / SECONDS_PER_YEAR, rel=1e-12)


def test_realized_vol_positive_for_random_walk():
    # Deterministic drifting log-normal walk: r_t alternates +/- .001 with drift.
    prices = [100.0]
    for i in range(100):
        bump = 0.001 if i % 2 == 0 else -0.0008
        prices.append(prices[-1] * math.exp(bump))
    sigma = realized_vol(prices)
    assert sigma > 0


def test_realized_vol_zero_for_constant():
    prices = [100.0] * 50
    assert realized_vol(prices) == pytest.approx(0.0, abs=1e-12)


def test_ewma_vol_reacts_to_new_shock():
    # 60 flat bars, then a final bar that jumps 2%.
    flat = [100.0] * 60
    shocked = flat + [102.0]
    pre = ewma_vol(flat)
    post = ewma_vol(shocked)
    assert post > pre


def test_ewma_vol_handles_empty_and_single():
    assert ewma_vol([]) == 0.0
    assert ewma_vol([100.0]) == 0.0
