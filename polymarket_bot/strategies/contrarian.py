"""Contrarian / mean-reversion strategy for Polymarket.

Core idea: After sharp price moves, prediction markets tend to
overshoot the true probability and then revert. This is because:

1. Retail traders panic-buy or panic-sell on news
2. Herding behavior amplifies moves beyond fair value
3. Market makers withdraw during volatile periods, reducing liquidity
4. Once volatility subsides, informed traders push prices back

Strategy:
- Detect large price dislocations (z-score > 2 from recent mean)
- Confirm with contrarian sentiment (crowd is panicking = opportunity)
- Fade the move with Kelly-sized positions
- Use tight time-based exits (mean reversion happens in hours, not days)

This is the complement to the momentum strategy. The key is knowing
WHEN to follow momentum vs WHEN to fade it. We use regime detection
for this (see market_regime module).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from polymarket_bot.data.models import Market, OrderBook, Side, Signal
from polymarket_bot.strategies.base import BaseStrategy
from polymarket_bot.utils.helpers import clamp

logger = structlog.get_logger()


class ContrarianStrategy(BaseStrategy):
    """Fade extreme price moves with mean-reversion expectations."""

    name = "contrarian"

    def __init__(
        self,
        lookback: int = 30,
        z_threshold: float = 1.3,
        min_edge: float = 0.02,
        max_reversion_pct: float = 0.50,
    ) -> None:
        self.lookback = lookback
        self.z_threshold = z_threshold
        self.min_edge = min_edge
        self.max_reversion_pct = max_reversion_pct

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
            if len(price_history) < self.lookback + 1:
                continue

            # Skip only very strong trends (allow contrarian in weak trends)
            regime = context.get(f"regime_{market.condition_id}", "unknown")
            if regime == "trending":
                # Check if trend is strong via price history slope
                ph = context.get(f"price_history_{market.condition_id}", [])
                if len(ph) >= 10:
                    slope = abs(ph[-1] - ph[-10]) / 10
                    if slope > 0.01:  # Strong trend — skip
                        continue

            try:
                signal = self._analyze(market, price_history, order_books, context)
                if signal is not None:
                    signals.append(signal)
            except Exception as e:
                logger.error("contrarian_error", market=market.condition_id, error=str(e))

        return signals

    def _analyze(
        self,
        market: Market,
        price_history: list[float],
        order_books: dict[str, OrderBook],
        context: dict[str, Any],
    ) -> Signal | None:
        arr = np.array(price_history)
        current = arr[-1]

        # ── Z-score calculation ──────────────────────────────────
        lookback_data = arr[-(self.lookback + 1):-1]  # Exclude current
        mean = np.mean(lookback_data)
        std = np.std(lookback_data, ddof=1)

        if std < 0.01:
            return None  # Not enough volatility

        z_score = (current - mean) / std

        if abs(z_score) < self.z_threshold:
            return None  # Not extreme enough

        # ── Speed of move ────────────────────────────────────────
        # Fast moves are more likely to revert than slow grinds
        recent_5 = arr[-5:]
        move_speed = abs(recent_5[-1] - recent_5[0]) / 5
        avg_move = np.mean(np.abs(np.diff(lookback_data)))
        speed_ratio = move_speed / max(avg_move, 0.001)

        if speed_ratio < 1.2:
            return None  # Move was too gradual - more likely a real shift

        # ── Estimate reversion target ────────────────────────────
        # Expect partial reversion toward the mean
        reversion_amount = (current - mean) * self.max_reversion_pct
        target = current - reversion_amount
        target = clamp(target, 0.05, 0.95)

        edge = abs(target - current)
        if edge < self.min_edge:
            return None

        # ── Contrarian sentiment confirmation ────────────────────
        # If everyone is panicking in one direction, the reversion is stronger
        sentiment = context.get(f"sentiment_{market.condition_id}")
        sentiment_confirms = False
        if sentiment and sentiment.tweet_count >= 5:
            # If price spiked up but sentiment is extremely bullish,
            # the crowd is over-excited -> fade
            if z_score > 0 and sentiment.bullish_pct > 0.7:
                sentiment_confirms = True
            elif z_score < 0 and sentiment.bearish_pct > 0.7:
                sentiment_confirms = True

        # ── Build signal ─────────────────────────────────────────
        if z_score > 0:
            # Price spiked up -> expect reversion down -> buy NO
            token = next((t for t in market.tokens if t.outcome == "No"), None)
            if token is None:
                return None
            fair_value = 1.0 - target
            market_price = 1.0 - current
        else:
            # Price dropped -> expect reversion up -> buy YES
            token = next((t for t in market.tokens if t.outcome == "Yes"), None)
            if token is None:
                return None
            fair_value = target
            market_price = current

        edge = abs(fair_value - market_price)

        # Confidence based on z-score extremity, speed, and sentiment
        conf = min(0.85, abs(z_score) / 4 + speed_ratio / 10)
        if sentiment_confirms:
            conf = min(0.90, conf + 0.15)

        return Signal(
            market_condition_id=market.condition_id,
            token_id=token.token_id,
            side=Side.BUY,
            outcome=token.outcome,
            estimated_fair_value=fair_value,
            market_price=market_price,
            edge=edge,
            confidence=conf,
            strategy=self.name,
            metadata={
                "z_score": round(float(z_score), 2),
                "speed_ratio": round(float(speed_ratio), 2),
                "reversion_target": round(float(target), 3),
                "sentiment_confirms": sentiment_confirms,
            },
        )
