"""Event catalyst strategy for scheduled-event-driven markets.

Core idea: Markets systematically misprice event risk around scheduled
catalysts (elections, earnings, court rulings, etc.) in two ways:

1. Pre-event complacency: Before a known event, implied volatility is
   often too LOW. People anchor on the current price and underestimate
   how much a binary outcome can move things. If the market is near
   50/50, a large move is coming -- buy the cheaper side. If momentum
   is already pushing toward an extreme, ride it.

2. Post-event overreaction: After an event resolves, markets often
   overshoot. A price that rockets to 0.90 on initial news may settle
   back to 0.80 as the dust clears. We fade these overreactions
   partially, capturing the reversion.

Category-aware adjustments reflect that political events are more
binary (higher confidence in extremes), sports events are more
frequent (smaller edge), and crypto events are less scheduled
(lower overall confidence).
"""

from __future__ import annotations

from typing import Any

import structlog

from polymarket_bot.data.models import Market, MarketCategory, OrderBook, Side, Signal
from polymarket_bot.strategies.base import BaseStrategy
from polymarket_bot.utils.helpers import clamp, time_to_expiry_hours

logger = structlog.get_logger()

# Category-specific confidence multipliers
_CATEGORY_MULTIPLIERS: dict[MarketCategory, float] = {
    MarketCategory.POLITICS: 1.2,   # Binary outcomes, higher confidence
    MarketCategory.SPORTS: 0.8,     # Frequent events, smaller edge
    MarketCategory.CRYPTO: 0.6,     # Less scheduled, lower confidence
    MarketCategory.POP_CULTURE: 0.9,
    MarketCategory.SCIENCE: 1.0,
    MarketCategory.OTHER: 0.9,
}


class EventCatalystStrategy(BaseStrategy):
    """Capitalize on mispricings around scheduled event catalysts."""

    name = "event_catalyst"

    def __init__(
        self,
        pre_event_days: tuple[float, float] = (1.0, 3.0),
        overreaction_threshold: float = 0.15,
        min_edge: float = 0.03,
    ) -> None:
        self.pre_event_days = pre_event_days
        self.overreaction_threshold = overreaction_threshold
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
            try:
                signal = self._analyze(market, order_books, context)
                if signal is not None:
                    signals.append(signal)
            except Exception as e:
                logger.error("event_catalyst_error", market=market.condition_id, error=str(e))

        return signals

    def _analyze(
        self,
        market: Market,
        order_books: dict[str, OrderBook],
        context: dict[str, Any],
    ) -> Signal | None:
        hours_left = time_to_expiry_hours(market.end_date)
        yes_price = market.yes_price
        price_history = context.get(f"price_history_{market.condition_id}", [])

        # Detect post-event overreaction first (takes priority)
        signal = self._check_post_event(market, price_history, yes_price)
        if signal is not None:
            return signal

        # Then check pre-event opportunity
        return self._check_pre_event(market, hours_left, yes_price)

    # ── Pre-event logic ─────────────────────────────────────────

    def _check_pre_event(
        self,
        market: Market,
        hours_left: float,
        yes_price: float,
    ) -> Signal | None:
        min_hours = self.pre_event_days[0] * 24
        max_hours = self.pre_event_days[1] * 24

        if hours_left < min_hours or hours_left > max_hours:
            return None

        # Urgency: closer to event = stronger signal (1.0 at max, 1.5 at min)
        urgency = 1.0 + 0.5 * (1.0 - (hours_left - min_hours) / (max_hours - min_hours))

        category_mult = _CATEGORY_MULTIPLIERS.get(market.category, 0.9)

        # ── Near 50/50: expect a large move, buy the cheaper side ───
        if 0.30 <= yes_price <= 0.70:
            # The cheaper side has better risk/reward before a catalyst
            if yes_price <= 0.50:
                # "No" is more expensive; buy "Yes" (the cheaper side)
                fair_value = clamp(yes_price + 0.05 * urgency, 0.05, 0.95)
                return self._build_signal(
                    market, fair_value, yes_price, urgency, category_mult,
                    regime="pre_event_uncertain",
                )
            else:
                # "Yes" is more expensive; buy "No" (the cheaper side)
                no_price = 1.0 - yes_price
                fair_value = clamp(no_price + 0.05 * urgency, 0.05, 0.95)
                return self._build_signal_no(
                    market, fair_value, no_price, urgency, category_mult,
                    regime="pre_event_uncertain",
                )

        # ── Drifting toward extreme: follow momentum ────────────────
        if yes_price > 0.70:
            # Momentum toward Yes, push fair value higher
            fair_value = clamp(yes_price + 0.04 * urgency, 0.05, 0.95)
            return self._build_signal(
                market, fair_value, yes_price, urgency, category_mult,
                regime="pre_event_momentum",
            )

        if yes_price < 0.30:
            # Momentum toward No, push No fair value higher
            no_price = 1.0 - yes_price
            fair_value = clamp(no_price + 0.04 * urgency, 0.05, 0.95)
            return self._build_signal_no(
                market, fair_value, no_price, urgency, category_mult,
                regime="pre_event_momentum",
            )

        return None

    # ── Post-event logic ────────────────────────────────────────

    def _check_post_event(
        self,
        market: Market,
        price_history: list[float],
        yes_price: float,
    ) -> Signal | None:
        if len(price_history) < 3:
            return None

        # Detect a large recent move (proxy for event just happened)
        recent_move = yes_price - price_history[-3]

        if abs(recent_move) < self.overreaction_threshold:
            return None

        category_mult = _CATEGORY_MULTIPLIERS.get(market.category, 0.9)

        # Post-event: use lower urgency (the event already happened)
        urgency = 0.8

        # ── Spiked to extreme high: fade the overreaction ───────────
        if recent_move > 0 and yes_price > 0.85:
            # Fair value is somewhat below current price
            reversion = recent_move * 0.3  # expect ~30% reversion
            fair_value = clamp(yes_price - reversion, 0.05, 0.95)
            no_price = 1.0 - yes_price
            no_fair = 1.0 - fair_value
            edge = abs(no_fair - no_price)
            if edge < self.min_edge:
                return None
            return self._build_signal_no(
                market, no_fair, no_price, urgency, category_mult,
                regime="post_event_fade_high",
            )

        # ── Dropped to extreme low: fade the overreaction ───────────
        if recent_move < 0 and yes_price < 0.15:
            reversion = abs(recent_move) * 0.3
            fair_value = clamp(yes_price + reversion, 0.05, 0.95)
            edge = abs(fair_value - yes_price)
            if edge < self.min_edge:
                return None
            return self._build_signal(
                market, fair_value, yes_price, urgency, category_mult,
                regime="post_event_fade_low",
            )

        return None

    # ── Signal builders ─────────────────────────────────────────

    def _build_signal(
        self,
        market: Market,
        fair_value: float,
        market_price: float,
        urgency: float,
        category_mult: float,
        regime: str,
    ) -> Signal | None:
        edge = abs(fair_value - market_price)
        if edge < self.min_edge:
            return None

        token = next((t for t in market.tokens if t.outcome == "Yes"), None)
        if token is None:
            return None

        confidence = self._compute_confidence(edge, urgency, category_mult)

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
                "regime": regime,
                "urgency": round(urgency, 2),
                "category": market.category.value,
                "category_mult": category_mult,
            },
        )

    def _build_signal_no(
        self,
        market: Market,
        fair_value: float,
        market_price: float,
        urgency: float,
        category_mult: float,
        regime: str,
    ) -> Signal | None:
        edge = abs(fair_value - market_price)
        if edge < self.min_edge:
            return None

        token = next((t for t in market.tokens if t.outcome == "No"), None)
        if token is None:
            return None

        confidence = self._compute_confidence(edge, urgency, category_mult)

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
                "regime": regime,
                "urgency": round(urgency, 2),
                "category": market.category.value,
                "category_mult": category_mult,
            },
        )

    @staticmethod
    def _compute_confidence(edge: float, urgency: float, category_mult: float) -> float:
        """Confidence = base * category_multiplier * urgency_factor.

        Base confidence scales with edge size (bigger edge = more confident).
        """
        base = clamp(0.3 + edge * 3.0, 0.2, 0.7)
        return min(0.85, base * category_mult * (urgency / 1.25))
