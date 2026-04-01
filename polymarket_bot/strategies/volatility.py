"""Volatility clustering strategy for Polymarket.

Core idea: Volatility tends to cluster — high-vol periods follow
high-vol periods, and low-vol periods follow low-vol periods. This
persistence allows us to forecast near-term volatility and position
accordingly.

Uses a GARCH-like framework:
- Compare short-term realized vol (10-step) to long-term (30-step)
- When short vol >> long vol, expect mean-reversion in vol (sell premium)
- When short vol << long vol, expect vol expansion (ride momentum)

Bollinger Band width acts as a secondary regime indicator:
- Narrow bands (low percentile) → compression, breakout imminent
- Wide bands (high percentile) → expansion exhaustion, reversion likely

Profitable because:
1. Prediction markets exhibit volatility clustering like all markets
2. Retail participants over-react during high-vol and under-react during low-vol
3. Combining vol regime detection with price position (near bands)
   gives directional edge others miss
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from polymarket_bot.data.models import Market, OrderBook, Side, Signal
from polymarket_bot.strategies.base import BaseStrategy
from polymarket_bot.utils.helpers import clamp

logger = structlog.get_logger()


class VolatilityStrategy(BaseStrategy):
    """Exploit volatility clustering to find mispriced outcomes."""

    name = "volatility"

    def __init__(
        self,
        short_window: int = 10,
        long_window: int = 30,
        min_history: int = 30,
        vol_ratio_threshold: float = 1.5,
        bb_period: int = 20,
        bb_std_dev: float = 2.0,
        bb_width_pctl_low: float = 0.25,
        bb_width_pctl_high: float = 0.75,
        min_edge: float = 0.03,
    ) -> None:
        self.short_window = short_window
        self.long_window = long_window
        self.min_history = min_history
        self.vol_ratio_threshold = vol_ratio_threshold
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        self.bb_width_pctl_low = bb_width_pctl_low
        self.bb_width_pctl_high = bb_width_pctl_high
        self.min_edge = min_edge

    def generate_signals(
        self,
        markets: list[Market],
        order_books: dict[str, OrderBook],
        context: dict[str, Any],
    ) -> list[Signal]:
        signals = []
        tradeable = self.filter_tradeable_markets(markets)

        for market in tradeable:
            price_history = context.get(f"price_history_{market.condition_id}", [])
            if len(price_history) < self.min_history:
                continue

            try:
                signal = self._analyze(market, price_history, order_books)
                if signal is not None:
                    signals.append(signal)
            except Exception as e:
                logger.error("volatility_error", market=market.condition_id, error=str(e))

        return signals

    # ── Internal analysis ────────────────────────────────────────

    def _analyze(
        self,
        market: Market,
        price_history: list[float],
        order_books: dict[str, OrderBook],
    ) -> Signal | None:
        arr = np.array(price_history)
        current = arr[-1]

        # ── Realized volatility (rolling std of log returns) ─────
        returns = np.diff(np.log(np.clip(arr, 1e-8, None)))
        short_vol = float(np.std(returns[-self.short_window:]))
        long_vol = float(np.std(returns[-self.long_window:]))

        if long_vol < 1e-8:
            return None  # Market is flat — no signal

        vol_ratio = short_vol / long_vol

        # ── GARCH-like regime classification ─────────────────────
        # vol_ratio > threshold → short vol elevated → expect mean-reversion
        # vol_ratio < 1/threshold → short vol suppressed → expect expansion
        vol_contracting = vol_ratio < (1.0 / self.vol_ratio_threshold)
        vol_expanding = vol_ratio > self.vol_ratio_threshold

        # ── Bollinger Bands ──────────────────────────────────────
        bb_slice = arr[-self.bb_period:]
        bb_mean = float(np.mean(bb_slice))
        bb_std = float(np.std(bb_slice))

        if bb_std < 1e-8:
            return None

        upper_band = bb_mean + self.bb_std_dev * bb_std
        lower_band = bb_mean - self.bb_std_dev * bb_std
        bb_width = upper_band - lower_band

        # Percentile rank of current BB width vs recent history
        bb_width_history = self._rolling_bb_width(arr)
        if len(bb_width_history) < 2:
            return None
        bb_width_pctl = float(np.mean(np.array(bb_width_history) <= bb_width))

        # Price position within bands (0 = lower, 1 = upper)
        band_position = (current - lower_band) / max(bb_width, 1e-8)
        band_position = clamp(band_position, 0.0, 1.0)

        # ── Detect recent price recovery after a drop ────────────
        recent_min = float(np.min(arr[-self.short_window:]))
        recovering = current > recent_min and (current - recent_min) > 0.5 * bb_std

        # ── Signal logic ─────────────────────────────────────────
        direction: float | None = None  # positive → buy YES, negative → buy NO

        # (a) Vol contracting + price near lower band → breakout up
        if vol_contracting and band_position < 0.3 and bb_width_pctl < self.bb_width_pctl_low:
            direction = 1.0

        # (b) Vol expanding after drop + price recovering → vol-adjusted value
        elif vol_expanding and recovering and band_position < 0.6:
            direction = 0.6

        # (c) Vol contracting + price near upper band → breakout down
        elif vol_contracting and band_position > 0.7 and bb_width_pctl < self.bb_width_pctl_low:
            direction = -1.0

        if direction is None:
            return None

        # ── Fair value estimation ────────────────────────────────
        # Mean-reversion target adjusted by vol regime
        if direction > 0:
            # Expect price to revert toward mean, biased upward by compression
            vol_adjustment = bb_std * clamp(1.0 / vol_ratio, 0.5, 2.0)
            fair_value = clamp(bb_mean + vol_adjustment * 0.5, 0.05, 0.95)
        else:
            # Expect price to revert toward mean, biased downward
            vol_adjustment = bb_std * clamp(1.0 / vol_ratio, 0.5, 2.0)
            fair_value = clamp(bb_mean - vol_adjustment * 0.5, 0.05, 0.95)

        edge = abs(fair_value - current)
        if edge < self.min_edge:
            return None

        # ── Confidence from vol ratio strength ───────────────────
        # Stronger divergence between short/long vol → higher confidence
        vol_divergence = abs(np.log(vol_ratio))  # Symmetric in log-space
        confidence = clamp(0.3 + vol_divergence * 0.4, 0.2, 0.85)

        # Boost confidence when BB width percentile confirms the regime
        if vol_contracting and bb_width_pctl < self.bb_width_pctl_low:
            confidence = min(0.85, confidence * 1.15)
        elif vol_expanding and bb_width_pctl > self.bb_width_pctl_high:
            confidence = min(0.85, confidence * 1.15)

        # ── Build signal ─────────────────────────────────────────
        yes_token = next((t for t in market.tokens if t.outcome == "Yes"), None)
        if yes_token is None:
            return None

        if direction > 0:
            token = yes_token
            market_price = current
        else:
            token = next((t for t in market.tokens if t.outcome == "No"), None)
            if token is None:
                return None
            fair_value = 1.0 - fair_value
            market_price = 1.0 - current

        return Signal(
            market_condition_id=market.condition_id,
            token_id=token.token_id,
            side=Side.BUY,
            outcome=token.outcome,
            estimated_fair_value=fair_value,
            market_price=market_price,
            edge=edge,
            confidence=confidence,
            strategy=self.name,
            metadata={
                "short_vol": round(float(short_vol), 6),
                "long_vol": round(float(long_vol), 6),
                "vol_ratio": round(float(vol_ratio), 4),
                "vol_regime": "contracting" if vol_contracting else "expanding" if vol_expanding else "normal",
                "bb_width": round(float(bb_width), 4),
                "bb_width_pctl": round(float(bb_width_pctl), 4),
                "band_position": round(float(band_position), 4),
                "recovering": recovering,
            },
        )

    def _rolling_bb_width(self, arr: np.ndarray) -> list[float]:
        """Compute rolling Bollinger Band width over the array.

        Returns a list of BB widths for each window ending position,
        used to determine the percentile rank of the current width.
        """
        widths: list[float] = []
        n = len(arr)
        start = max(0, n - self.long_window - self.bb_period)
        for i in range(start + self.bb_period, n + 1):
            window = arr[i - self.bb_period : i]
            std = float(np.std(window))
            widths.append(2 * self.bb_std_dev * std)
        return widths
