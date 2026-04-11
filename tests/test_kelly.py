"""Unit tests for polymarket_bot.pricing.kelly.

Covers the fractional Kelly formula, Bayesian shrinkage toward the market
price, and the Polymarket dynamic taker-fee approximation.
"""

from __future__ import annotations

import pytest

from polymarket_bot.pricing.kelly import (
    kelly_fraction,
    polymarket_taker_fee,
    shrink_toward_market,
)


def test_kelly_zero_when_no_edge():
    assert kelly_fraction(q=0.5, p=0.5) == 0.0


def test_kelly_formula_matches_classic():
    f = kelly_fraction(q=0.6, p=0.5, fraction=1.0, cap=1.0)
    assert f == pytest.approx((0.6 - 0.5) / (1 - 0.5), rel=1e-12)


def test_kelly_quarter_kelly_default():
    # Default fraction is 0.25 => 0.25 * 0.2 = 0.05.
    f = kelly_fraction(q=0.6, p=0.5)
    assert f == pytest.approx(0.05, rel=1e-12)


def test_kelly_cap_applied():
    f = kelly_fraction(q=0.99, p=0.01, fraction=1.0, cap=0.25)
    assert f == pytest.approx(0.25, rel=1e-12)


def test_kelly_zero_when_q_below_p():
    assert kelly_fraction(q=0.4, p=0.5) == 0.0


def test_kelly_rejects_out_of_bounds():
    assert kelly_fraction(q=0.6, p=0.0) == 0.0
    assert kelly_fraction(q=0.6, p=1.0) == 0.0
    assert kelly_fraction(q=-0.1, p=0.5) == 0.0
    assert kelly_fraction(q=1.1, p=0.5) == 0.0


def test_shrink_toward_market_low_confidence():
    assert shrink_toward_market(q_model=0.7, p_market=0.5, confidence=0.0) == pytest.approx(0.5)


def test_shrink_toward_market_full_confidence():
    assert shrink_toward_market(q_model=0.7, p_market=0.5, confidence=1.0) == pytest.approx(0.7)


def test_shrink_toward_market_half():
    assert shrink_toward_market(q_model=0.7, p_market=0.5, confidence=0.5) == pytest.approx(0.6)


def test_polymarket_taker_fee_peaks_at_half():
    mid = polymarket_taker_fee(0.5)
    assert mid > polymarket_taker_fee(0.1)
    assert mid > polymarket_taker_fee(0.9)
    assert mid == pytest.approx(0.018, rel=1e-9)


def test_polymarket_taker_fee_decays_symmetric():
    assert polymarket_taker_fee(0.2) == pytest.approx(polymarket_taker_fee(0.8), rel=1e-12)
