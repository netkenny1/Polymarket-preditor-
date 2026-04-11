"""Black-Scholes digital option pricing + realized-volatility estimators.

For the Polymarket BTC 5-minute up/down market, every 300-second window
is a cash-or-nothing binary. We use the GBM / Black-Scholes formula to
compute the fair probability that BTC closes above (or below) the window's
starting price, given current spot, time remaining, and realized volatility.

Formula (cash-or-nothing digital, r = 0):

    d2(S, K, T, sigma) = (ln(S/K) - 0.5 * sigma^2 * T) / (sigma * sqrt(T))
    P(S_T > K)  = N(d2)
    P(S_T < K)  = N(-d2) = 1 - N(d2)

We annualize sigma using 525,600 minutes/year (crypto trades 24/7, no
calendar effects). All time inputs to these functions are in YEARS.
All sigma inputs are ANNUALIZED. Mismatches are the #1 bug source — if
you pass T in seconds, you'll silently get garbage.

References:
  - Hull, Options Futures and Other Derivatives, ch. 26 (binary/digital options)
  - RiskMetrics Technical Document (1996): EWMA volatility, lambda = 0.94
  - Parkinson (1980): The extreme value method for estimating variance
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.stats import norm

# 24/7 crypto markets: minutes per year for annualization.
MINUTES_PER_YEAR = 525_600
SECONDS_PER_YEAR = 31_536_000


def digital_call_prob(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = 0.0,
) -> float:
    """Risk-neutral probability of S_T > K (digital call / BTC "Up").

    Args:
        S: Current spot price.
        K: Strike (reference price — for BTC 5-min, this is the window-start BTC price).
        T: Time to expiry in YEARS.
        sigma: ANNUALIZED volatility (e.g. 0.60 = 60%).
        r: Risk-free rate (≈ 0 for 5-minute horizons).

    Returns:
        Probability in [0, 1]. Returns a degenerate {0, 0.5, 1} when T or
        sigma are <= 0 so callers don't have to special-case expiry.
    """
    if T <= 0 or sigma <= 0:
        if S > K:
            return 1.0
        if S < K:
            return 0.0
        return 0.5
    if S <= 0 or K <= 0:
        return 0.5

    s_sqrt_t = sigma * math.sqrt(T)
    d2 = (math.log(S / K) + (r - 0.5 * sigma * sigma) * T) / s_sqrt_t
    return float(norm.cdf(d2))


def digital_put_prob(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = 0.0,
) -> float:
    """Risk-neutral probability of S_T < K (digital put / BTC "Down")."""
    return 1.0 - digital_call_prob(S, K, T, sigma, r)


def years_from_seconds(seconds: float) -> float:
    """Convert seconds to the 'years' unit expected by BS."""
    return seconds / SECONDS_PER_YEAR


# ── Volatility estimators ────────────────────────────────────────────

def realized_vol(
    prices: Sequence[float],
    bars_per_year: int = MINUTES_PER_YEAR,
) -> float:
    """Annualized realized volatility from close-to-close log returns.

    Args:
        prices: Sequence of bar closes (e.g. 1-minute BTC closes).
        bars_per_year: 525,600 for 1-min crypto bars.

    Returns:
        Annualized sigma. Zero if fewer than 2 prices.
    """
    if len(prices) < 2:
        return 0.0
    arr = np.asarray(prices, dtype=float)
    # Guard against zeros / bad data
    if (arr <= 0).any():
        return 0.0
    logret = np.diff(np.log(arr))
    if logret.size < 1:
        return 0.0
    sigma_bar = float(np.std(logret, ddof=1)) if logret.size > 1 else float(np.abs(logret[0]))
    return sigma_bar * math.sqrt(bars_per_year)


def ewma_vol(
    prices: Sequence[float],
    lam: float = 0.94,
    bars_per_year: int = MINUTES_PER_YEAR,
) -> float:
    """RiskMetrics EWMA volatility from a price series.

    sigma_t^2 = lam * sigma_{t-1}^2 + (1 - lam) * r_t^2

    Reacts faster than simple rolling std to vol regime changes, which is
    what we want for 5-minute markets. Default lambda = 0.94 (daily) is
    fine for 1-minute bars — don't over-tune.

    Args:
        prices: Sequence of bar closes.
        lam: Decay factor in [0, 1). 0.94 is the RiskMetrics default.
        bars_per_year: 525,600 for 1-min crypto bars.

    Returns:
        Annualized sigma.
    """
    if len(prices) < 2:
        return 0.0
    arr = np.asarray(prices, dtype=float)
    if (arr <= 0).any():
        return 0.0
    logret = np.diff(np.log(arr))
    if logret.size < 1:
        return 0.0

    var = float(logret[0] ** 2)
    for r in logret[1:]:
        var = lam * var + (1.0 - lam) * (r * r)
    return math.sqrt(var * bars_per_year)


def parkinson_vol(
    highs: Sequence[float],
    lows: Sequence[float],
    bars_per_year: int = MINUTES_PER_YEAR,
) -> float:
    """Parkinson (1980) high-low volatility estimator.

    sigma^2 = (1 / (4 ln 2)) * mean(ln(H/L)^2)

    ~5x more statistically efficient than close-to-close for the same
    sample size. Requires high/low data per bar.
    """
    if len(highs) == 0 or len(highs) != len(lows):
        return 0.0
    h = np.asarray(highs, dtype=float)
    l = np.asarray(lows, dtype=float)
    if (h <= 0).any() or (l <= 0).any():
        return 0.0
    log_hl = np.log(h / l)
    var_bar = float(np.mean(log_hl ** 2) / (4.0 * math.log(2.0)))
    return math.sqrt(var_bar * bars_per_year)


def blended_vol(
    closes: Sequence[float],
    highs: Sequence[float] | None = None,
    lows: Sequence[float] | None = None,
    lam: float = 0.94,
    bars_per_year: int = MINUTES_PER_YEAR,
) -> float:
    """50/50 blend of EWMA and Parkinson for robustness.

    Falls back to pure EWMA if highs/lows are not provided.
    """
    ewma = ewma_vol(closes, lam=lam, bars_per_year=bars_per_year)
    if not highs or not lows:
        return ewma
    park = parkinson_vol(highs, lows, bars_per_year=bars_per_year)
    if ewma <= 0:
        return park
    if park <= 0:
        return ewma
    return 0.5 * (ewma + park)
