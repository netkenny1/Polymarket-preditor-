"""Order book microstructure analysis strategy.

Core idea: The order book reveals hidden information about where
prices are going before the price actually moves. By analyzing
the structure and dynamics of the book, we can predict short-term
price direction.

Signals:
1. Book Imbalance Momentum: Large bid/ask imbalance predicts
   price movement in the direction of the heavier side
2. Large Order Detection: Institutional-size orders signal
   informed trading
3. Spread Compression: Narrowing spread indicates an imminent
   directional move
4. Depth Cascade: Multiple levels of heavy support/resistance
   are stronger signals than a single level

This is the closest thing to HFT on Polymarket's CLOB.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from polymarket_bot.data.models import Market, OrderBook, Side, Signal
from polymarket_bot.strategies.base import BaseStrategy
from polymarket_bot.utils.helpers import clamp

logger = structlog.get_logger()


class MicrostructureStrategy(BaseStrategy):
    """Trade based on order book microstructure analysis."""

    name = "microstructure"

    def __init__(
        self,
        imbalance_threshold: float = 0.25,
        large_order_multiple: float = 3.0,
        min_depth_usd: float = 200.0,
        min_edge: float = 0.03,
    ) -> None:
        self.imbalance_threshold = imbalance_threshold
        self.large_order_multiple = large_order_multiple
        self.min_depth = min_depth_usd
        self.min_edge = min_edge
        # Track historical imbalances for momentum
        self._imbalance_history: dict[str, list[float]] = {}

    def generate_signals(
        self,
        markets: list[Market],
        order_books: dict[str, OrderBook],
        context: dict[str, Any],
    ) -> list[Signal]:
        signals = []
        tradeable = self.filter_tradeable_markets(markets, order_books)

        for market in tradeable:
            for token in market.tokens:
                book = order_books.get(token.token_id)
                if book is None:
                    continue

                try:
                    signal = self._analyze_book(market, token, book)
                    if signal is not None:
                        signals.append(signal)
                except Exception as e:
                    logger.error("microstructure_error", error=str(e))

        return signals

    def _analyze_book(
        self, market: Market, token: Any, book: OrderBook,
    ) -> Signal | None:
        """Analyze a single order book for microstructure signals."""
        if not book.bids or not book.asks:
            return None

        bid_depth = book.bid_depth
        ask_depth = book.ask_depth
        total_depth = bid_depth + ask_depth

        if total_depth < self.min_depth:
            return None

        mid = book.mid_price
        if mid is None:
            return None

        # ── Signal 1: Book Imbalance ─────────────────────────────
        imbalance = (bid_depth - ask_depth) / total_depth  # -1 to 1

        # Track imbalance history for imbalance momentum
        hist = self._imbalance_history.setdefault(token.token_id, [])
        hist.append(imbalance)
        if len(hist) > 20:
            hist.pop(0)

        # ── Signal 2: Large Orders ───────────────────────────────
        avg_level_size = total_depth / max(len(book.bids) + len(book.asks), 1)
        large_bid = any(
            level.size > avg_level_size * self.large_order_multiple
            for level in book.bids[:3]
        )
        large_ask = any(
            level.size > avg_level_size * self.large_order_multiple
            for level in book.asks[:3]
        )

        # ── Signal 3: Depth Cascade ──────────────────────────────
        # Check if multiple bid levels are heavy (wall of support)
        bid_cascade = sum(
            1 for level in book.bids[:3]
            if level.size > avg_level_size * 1.5
        )
        ask_cascade = sum(
            1 for level in book.asks[:3]
            if level.size > avg_level_size * 1.5
        )

        # ── Signal 4: Imbalance Momentum ─────────────────────────
        if len(hist) >= 5:
            recent_avg = np.mean(hist[-5:])
            older_avg = np.mean(hist[:-5]) if len(hist) > 5 else 0
            imbalance_momentum = recent_avg - older_avg
        else:
            imbalance_momentum = 0.0

        # ── Composite Score ──────────────────────────────────────
        score = 0.0

        # Imbalance signal (strongest)
        if abs(imbalance) > self.imbalance_threshold:
            score += imbalance * 0.5

        # Large order signal
        if large_bid and not large_ask:
            score += 0.2
        elif large_ask and not large_bid:
            score -= 0.2

        # Depth cascade
        if bid_cascade >= 2 and ask_cascade < 2:
            score += 0.15
        elif ask_cascade >= 2 and bid_cascade < 2:
            score -= 0.15

        # Imbalance momentum
        score += clamp(imbalance_momentum * 2, -0.15, 0.15)

        if abs(score) < 0.2:
            return None

        # ── Generate Signal ──────────────────────────────────────
        adjustment = score * 0.06  # Max ~6% adjustment from microstructure
        fair_value = clamp(mid + adjustment, 0.05, 0.95)
        edge = abs(fair_value - mid)

        if edge < self.min_edge:
            return None

        if score > 0:
            # Expect price to go up
            if token.outcome == "Yes":
                side = Side.BUY
                market_price = mid
            else:
                return None  # Don't generate opposing signal
        else:
            # Expect price to go down
            if token.outcome == "No":
                side = Side.BUY
                fair_value = 1.0 - fair_value
                market_price = 1.0 - mid
                edge = abs(fair_value - market_price)
            else:
                return None

        confidence = min(0.70, abs(score))

        if total_depth < 500.0:
            confidence *= 0.5

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
                "imbalance": round(float(imbalance), 3),
                "large_bid": large_bid,
                "large_ask": large_ask,
                "bid_cascade": bid_cascade,
                "ask_cascade": ask_cascade,
                "composite_score": round(float(score), 3),
            },
        )
