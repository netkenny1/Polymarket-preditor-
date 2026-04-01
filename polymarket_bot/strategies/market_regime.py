"""Market regime detection for adaptive strategy selection.

Classifies each market into one of three regimes:
- TRENDING: Strong directional movement, favor momentum strategy
- MEAN_REVERTING: Choppy/range-bound, favor contrarian strategy
- VOLATILE: High uncertainty, reduce position sizes

Uses Hurst exponent estimation and volatility regime classification.
The Hurst exponent H tells us:
- H > 0.55: Trending (persistent) -> use momentum
- H < 0.45: Mean-reverting (anti-persistent) -> use contrarian
- 0.45 <= H <= 0.55: Random walk -> reduce confidence

This is critical for profitability: applying the wrong strategy to the
wrong regime is the #1 way to lose money in prediction markets.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger()


class Regime(str, Enum):
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


@dataclass
class RegimeState:
    """Current regime classification for a market."""

    regime: Regime
    hurst_exponent: float
    volatility: float
    volatility_percentile: float  # vs historical
    trend_strength: float  # 0 to 1
    confidence: float  # How confident we are in the classification


class RegimeDetector:
    """Classify market regimes using statistical properties of price series."""

    def __init__(
        self,
        min_history: int = 30,
        hurst_trending_threshold: float = 0.55,
        hurst_reverting_threshold: float = 0.45,
        high_vol_percentile: float = 0.80,
    ) -> None:
        self.min_history = min_history
        self.hurst_trending = hurst_trending_threshold
        self.hurst_reverting = hurst_reverting_threshold
        self.high_vol_pctl = high_vol_percentile

    def detect(self, price_history: list[float]) -> RegimeState:
        """Classify the current market regime from price history."""
        if len(price_history) < self.min_history:
            return RegimeState(
                regime=Regime.UNKNOWN, hurst_exponent=0.5,
                volatility=0.0, volatility_percentile=0.5,
                trend_strength=0.0, confidence=0.0,
            )

        arr = np.array(price_history)

        # ── Hurst exponent (simplified R/S method) ───────────────
        hurst = self._estimate_hurst(arr)

        # ── Volatility analysis ──────────────────────────────────
        returns = np.diff(arr)
        current_vol = np.std(returns[-10:]) if len(returns) >= 10 else np.std(returns)

        # Rolling volatility for percentile calculation
        vol_window = 10
        rolling_vols = []
        for i in range(vol_window, len(returns)):
            rolling_vols.append(np.std(returns[i - vol_window:i]))

        if rolling_vols:
            vol_percentile = float(np.mean(np.array(rolling_vols) <= current_vol))
        else:
            vol_percentile = 0.5

        # ── Trend strength (ADX-like) ────────────────────────────
        if len(arr) >= 14:
            up_moves = np.maximum(np.diff(arr), 0)
            down_moves = np.maximum(-np.diff(arr), 0)

            avg_up = np.mean(up_moves[-14:])
            avg_down = np.mean(down_moves[-14:])
            total = avg_up + avg_down
            trend_strength = abs(avg_up - avg_down) / total if total > 0 else 0.0
        else:
            trend_strength = 0.0

        # ── Regime classification ────────────────────────────────
        if vol_percentile >= self.high_vol_pctl:
            regime = Regime.VOLATILE
            confidence = vol_percentile
        elif hurst > self.hurst_trending and trend_strength > 0.3:
            regime = Regime.TRENDING
            confidence = min(0.9, (hurst - 0.5) * 4 + trend_strength)
        elif hurst < self.hurst_reverting:
            regime = Regime.MEAN_REVERTING
            confidence = min(0.9, (0.5 - hurst) * 4)
        else:
            # Ambiguous: use trend strength as tiebreaker
            if trend_strength > 0.4:
                regime = Regime.TRENDING
                confidence = trend_strength * 0.7
            else:
                regime = Regime.MEAN_REVERTING
                confidence = (1 - trend_strength) * 0.5

        return RegimeState(
            regime=regime,
            hurst_exponent=float(hurst),
            volatility=float(current_vol),
            volatility_percentile=float(vol_percentile),
            trend_strength=float(trend_strength),
            confidence=float(confidence),
        )

    def _estimate_hurst(self, series: np.ndarray) -> float:
        """Estimate Hurst exponent using the R/S (rescaled range) method.

        H > 0.5: Persistent (trending)
        H = 0.5: Random walk
        H < 0.5: Anti-persistent (mean reverting)
        """
        n = len(series)
        if n < 20:
            return 0.5

        returns = np.diff(series)
        if len(returns) < 10:
            return 0.5

        # Use multiple window sizes for more robust estimate
        max_k = min(len(returns) // 2, 50)
        if max_k < 4:
            return 0.5

        window_sizes = []
        rs_values = []

        for k in range(4, max_k + 1, 2):
            rs_list = []
            for start in range(0, len(returns) - k + 1, k):
                window = returns[start:start + k]
                mean_r = np.mean(window)
                deviate = np.cumsum(window - mean_r)
                r = np.max(deviate) - np.min(deviate)
                s = np.std(window, ddof=1)
                if s > 1e-10:
                    rs_list.append(r / s)

            if rs_list:
                window_sizes.append(k)
                rs_values.append(np.mean(rs_list))

        if len(window_sizes) < 3:
            return 0.5

        # Log-log regression to estimate H
        log_n = np.log(window_sizes)
        log_rs = np.log(np.maximum(rs_values, 1e-10))

        # Simple linear regression
        slope = np.polyfit(log_n, log_rs, 1)[0]

        # Clamp to reasonable range
        return float(max(0.1, min(0.9, slope)))

    def classify_all_markets(
        self, price_histories: dict[str, list[float]]
    ) -> dict[str, RegimeState]:
        """Classify regimes for multiple markets."""
        results = {}
        for market_id, history in price_histories.items():
            results[market_id] = self.detect(history)
        return results
