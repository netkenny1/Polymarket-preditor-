"""Momentum / trend-following strategy for Polymarket.

Core idea: Markets that are trending in a direction tend to continue
trending. This is because information gets absorbed gradually -
when a market starts moving, it often hasn't finished moving yet.

Uses multiple timeframe momentum:
- Short-term (5-step): Captures immediate price action
- Medium-term (20-step): Captures developing trends
- Long-term (50-step): Captures major sentiment shifts

Combines momentum signals with volume confirmation and
rate-of-change acceleration to avoid chasing stale moves.

Profitable because:
1. Prediction market participants update beliefs slowly
2. New information (polls, events) takes time to fully price in
3. Momentum is the strongest factor in most financial markets
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from polymarket_bot.data.models import Market, OrderBook, Side, Signal
from polymarket_bot.strategies.base import BaseStrategy
from polymarket_bot.utils.helpers import clamp

logger = structlog.get_logger()


class MomentumStrategy(BaseStrategy):
    """Trade in the direction of price momentum with volume confirmation."""

    name = "momentum"

    def __init__(
        self,
        short_window: int = 5,
        medium_window: int = 20,
        long_window: int = 50,
        min_edge: float = 0.02,
        acceleration_threshold: float = 0.002,
    ) -> None:
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        self.min_edge = min_edge
        self.acceleration_threshold = acceleration_threshold

    def generate_signals(
        self,
        markets: list[Market],
        order_books: dict[str, OrderBook],
        context: dict[str, Any],
    ) -> list[Signal]:
        signals = []
        tradeable = self.filter_tradeable_markets(markets, order_books)

        for market in tradeable:
            price_history = context.get(f"price_history_{market.condition_id}", [])
            if len(price_history) < self.medium_window + 2:
                continue

            try:
                signal = self._analyze(market, price_history, order_books)
                if signal is not None:
                    signals.append(signal)
            except Exception as e:
                logger.error("momentum_error", market=market.condition_id, error=str(e))

        return signals

    def _analyze(
        self,
        market: Market,
        price_history: list[float],
        order_books: dict[str, OrderBook],
    ) -> Signal | None:
        arr = np.array(price_history)
        current = arr[-1]

        # ── Multi-timeframe momentum ─────────────────────────────
        short_ma = np.mean(arr[-self.short_window:])
        medium_ma = np.mean(arr[-self.medium_window:])

        # Use available history for long MA (may be shorter than window)
        long_len = min(self.long_window, len(arr))
        long_ma = np.mean(arr[-long_len:])

        if long_len < self.long_window * 0.7:
            return None  # Not enough history for reliable long MA

        # Rate of change (normalized)
        short_roc = (current - arr[-self.short_window]) / max(arr[-self.short_window], 0.01)
        medium_roc = (current - arr[-self.medium_window]) / max(arr[-self.medium_window], 0.01)

        # Acceleration: is momentum increasing?
        if len(arr) > self.short_window + 1:
            prev_short_roc = (arr[-2] - arr[-(self.short_window + 1)]) / max(arr[-(self.short_window + 1)], 0.01)
            acceleration = short_roc - prev_short_roc
        else:
            acceleration = 0.0

        # ── Trend strength scoring ───────────────────────────────
        # Score from -1 (strong bearish) to +1 (strong bullish)
        trend_score = 0.0

        # MA alignment (bullish: short > medium > long)
        if short_ma > medium_ma > long_ma:
            trend_score += 0.4
        elif short_ma < medium_ma < long_ma:
            trend_score -= 0.4

        # Short-term momentum
        trend_score += clamp(short_roc * 5, -0.3, 0.3)

        # Medium-term momentum
        trend_score += clamp(medium_roc * 3, -0.2, 0.2)

        # Acceleration bonus
        if abs(acceleration) > self.acceleration_threshold:
            trend_score += 0.1 * np.sign(acceleration)

        # ── Volume confirmation ──────────────────────────────────
        # Check if order book shows supportive depth
        yes_token = next((t for t in market.tokens if t.outcome == "Yes"), None)
        if yes_token is None:
            return None

        book = order_books.get(yes_token.token_id)

        # Prefer order book mid_price over last-trade price for fresher data
        if book and book.mid_price is not None:
            current = book.mid_price

        if book and book.bid_depth > 0 and book.ask_depth > 0:
            depth_ratio = book.bid_depth / (book.bid_depth + book.ask_depth)
            # If price is trending up, expect more bid depth (support)
            if trend_score > 0 and depth_ratio > 0.55:
                trend_score *= 1.2  # Volume confirms the trend
            elif trend_score < 0 and depth_ratio < 0.45:
                trend_score *= 1.2

        # ── Signal generation ────────────────────────────────────
        if abs(trend_score) < 0.12:
            return None  # No strong trend

        # Project fair value based on momentum continuation
        momentum_adjustment = trend_score * 0.10  # Max 10% from momentum
        fair_value = clamp(current + momentum_adjustment, 0.05, 0.95)
        edge = abs(fair_value - current)

        if edge < self.min_edge:
            return None

        if trend_score > 0:
            side = Side.BUY
            token = yes_token
            market_price = current
        else:
            side = Side.BUY
            token = next((t for t in market.tokens if t.outcome == "No"), None)
            if token is None:
                return None
            fair_value = 1.0 - fair_value
            market_price = 1.0 - current

        confidence = min(0.85, abs(trend_score))

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
                "trend_score": round(float(trend_score), 3),
                "short_roc": round(float(short_roc), 4),
                "medium_roc": round(float(medium_roc), 4),
                "acceleration": round(float(acceleration), 4),
                "ma_alignment": "bullish" if short_ma > medium_ma > long_ma else "bearish" if short_ma < medium_ma < long_ma else "mixed",
            },
        )
