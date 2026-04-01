"""Base strategy interface for all trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from polymarket_bot.data.models import Market, OrderBook, Signal


class BaseStrategy(ABC):
    """Abstract base class for trading strategies.

    Each strategy implements `generate_signals()` which analyzes
    markets and returns a list of trading signals with estimated
    fair values and confidence levels.
    """

    name: str = "base"

    @abstractmethod
    def generate_signals(
        self,
        markets: list[Market],
        order_books: dict[str, OrderBook],
        context: dict[str, Any],
    ) -> list[Signal]:
        """Generate trading signals for the given markets.

        Args:
            markets: List of active markets to analyze.
            order_books: Order books keyed by token_id.
            context: Additional context (sentiment data, external odds, etc.)

        Returns:
            List of Signal objects with trade recommendations.
        """
        ...

    def filter_tradeable_markets(
        self,
        markets: list[Market],
        min_liquidity: float = 500.0,
        max_spread: float = 0.10,
    ) -> list[Market]:
        """Filter markets that meet minimum trading criteria."""
        tradeable = []
        for m in markets:
            if not m.active:
                continue
            if m.liquidity < min_liquidity:
                continue
            if m.spread > max_spread:
                continue
            tradeable.append(m)
        return tradeable
