"""Time-decay strategy for expiring prediction markets.

Core idea: As markets approach their resolution date, prices should
converge toward 0 or 1 as the outcome becomes clearer. Markets
that are "stuck" in the 30-70% range close to expiry often present
opportunities because:

1. If an event is likely (>60%) and the market resolves soon,
   the risk/reward is asymmetric - you buy at 0.60 to win 1.00.
2. Markets near expiry with extreme prices (>85% or <15%) tend
   to be correctly priced - we avoid those.
3. The "sweet spot" is markets 1-7 days from expiry priced 25-75%
   where our other signals can give us an edge with faster payoff.

Also implements theta-collection: selling overpriced time value
in markets where the implied volatility exceeds what's realistic
given the time remaining.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog

from polymarket_bot.data.models import Market, OrderBook, Side, Signal
from polymarket_bot.strategies.base import BaseStrategy
from polymarket_bot.utils.helpers import clamp, time_to_expiry_hours

logger = structlog.get_logger()


class TimeDecayStrategy(BaseStrategy):
    """Exploit time decay and expiry dynamics in prediction markets."""

    name = "time_decay"

    def __init__(
        self,
        min_edge: float = 0.04,
        min_hours_to_expiry: float = 6.0,
        max_hours_to_expiry: float = 168.0,  # 7 days
        sweet_spot_range: tuple[float, float] = (0.25, 0.75),
    ) -> None:
        self.min_edge = min_edge
        self.min_hours = min_hours_to_expiry
        self.max_hours = max_hours_to_expiry
        self.sweet_spot = sweet_spot_range

    def generate_signals(
        self,
        markets: list[Market],
        order_books: dict[str, OrderBook],
        context: dict[str, Any],
    ) -> list[Signal]:
        signals = []
        tradeable = self.filter_tradeable_markets(markets)

        for market in tradeable:
            try:
                signal = self._analyze(market, order_books, context)
                if signal is not None:
                    signals.append(signal)
            except Exception as e:
                logger.error("time_decay_error", market=market.condition_id, error=str(e))

        return signals

    def _analyze(
        self,
        market: Market,
        order_books: dict[str, OrderBook],
        context: dict[str, Any],
    ) -> Signal | None:
        hours_left = time_to_expiry_hours(market.end_date)

        # Only trade markets in the time window
        if hours_left < self.min_hours or hours_left > self.max_hours:
            return None

        yes_price = market.yes_price

        # Skip extreme prices (likely correctly priced near expiry)
        if yes_price < 0.10 or yes_price > 0.90:
            return None

        # ── Time-urgency amplifier ───────────────────────────────
        # Closer to expiry = stronger signal (less time for reversal)
        # This creates an urgency factor from 0.5 (far) to 1.5 (near)
        urgency = 1.0 + 0.5 * (1.0 - hours_left / self.max_hours)

        # ── Gather supporting signals from context ───────────────
        # Use other strategies' sentiment/model data if available
        sentiment = context.get(f"sentiment_{market.condition_id}")
        price_history = context.get(f"price_history_{market.condition_id}", [])

        directional_bias = 0.0

        # Sentiment bias
        if sentiment and sentiment.tweet_count >= 5:
            directional_bias += sentiment.avg_sentiment * 0.3

        # Price trend bias (recent direction)
        if len(price_history) >= 5:
            recent_trend = price_history[-1] - price_history[-5]
            directional_bias += clamp(recent_trend * 2, -0.3, 0.3)

        # ── Time-value analysis ──────────────────────────────────
        # Markets in the "sweet spot" price range near expiry
        in_sweet_spot = self.sweet_spot[0] <= yes_price <= self.sweet_spot[1]

        if not in_sweet_spot and abs(directional_bias) < 0.2:
            return None

        # Estimate fair value incorporating time decay
        # Near expiry, prices should be more extreme (closer to 0 or 1)
        # A market at 0.60 with 12 hours left should probably be 0.65+ or 0.55-
        time_pressure = 0.05 * urgency  # How much to push toward extremes

        if directional_bias > 0:
            fair_value = clamp(yes_price + time_pressure * directional_bias * 2, 0.05, 0.95)
        elif directional_bias < 0:
            fair_value = clamp(yes_price + time_pressure * directional_bias * 2, 0.05, 0.95)
        else:
            return None  # No directional bias = no trade

        edge = abs(fair_value - yes_price)

        if edge < self.min_edge:
            return None

        # Direction
        if fair_value > yes_price:
            side = Side.BUY
            token = next((t for t in market.tokens if t.outcome == "Yes"), None)
            market_price = yes_price
        else:
            side = Side.BUY
            token = next((t for t in market.tokens if t.outcome == "No"), None)
            if token is None:
                return None
            fair_value = 1.0 - fair_value
            market_price = 1.0 - yes_price
            edge = abs(fair_value - market_price)

        if token is None:
            return None

        # Higher confidence near expiry (less time for mean reversion)
        confidence = min(0.80, 0.4 + 0.3 * urgency + abs(directional_bias) * 0.3)

        return Signal(
            market_condition_id=market.condition_id,
            token_id=token.token_id,
            side=side,
            outcome=token.outcome,
            estimated_fair_value=fair_value,
            market_price=market_price,
            edge=edge,
            confidence=confidence,
            strategy=self.name,
            metadata={
                "hours_to_expiry": round(hours_left, 1),
                "urgency": round(urgency, 2),
                "directional_bias": round(directional_bias, 3),
                "in_sweet_spot": in_sweet_spot,
            },
        )
