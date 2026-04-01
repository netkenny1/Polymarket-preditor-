"""Market making strategy for Polymarket.

Core idea: Provide liquidity on both sides of the order book and
profit from the bid-ask spread. Skew quotes based on inventory to
manage risk.

This is a classic HFT strategy adapted for prediction markets:
1. Place limit orders on both sides (bid and ask)
2. When both sides fill, we profit the spread
3. Skew prices based on current inventory to avoid accumulating
   too much directional risk
4. Only market-make in liquid, stable markets (not trending)

Profitability driver: Prediction markets have wider spreads than
traditional markets, so there is more room for market makers.
"""

from __future__ import annotations

from typing import Any

import structlog

from polymarket_bot.config import MarketMakerConfig
from polymarket_bot.data.models import Market, OrderBook, Position, Side, Signal
from polymarket_bot.strategies.base import BaseStrategy
from polymarket_bot.utils.helpers import clamp, round_price

logger = structlog.get_logger()


class MarketMakerStrategy(BaseStrategy):
    """Provide liquidity and earn the spread.

    Places bid and ask orders around the mid price, skewed by
    current inventory. Avoids trending markets where directional
    risk is high.
    """

    name = "market_maker"

    def __init__(self, config: MarketMakerConfig) -> None:
        self.config = config

    def generate_signals(
        self,
        markets: list[Market],
        order_books: dict[str, OrderBook],
        context: dict[str, Any],
    ) -> list[Signal]:
        signals = []
        positions: dict[str, Position] = context.get("positions", {})
        tradeable = self.filter_tradeable_markets(markets, min_liquidity=1000.0)

        for market in tradeable:
            try:
                market_signals = self._quote_market(market, order_books, positions)
                signals.extend(market_signals)
            except Exception as e:
                logger.error("mm_error", market=market.condition_id, error=str(e))

        return signals

    def _quote_market(
        self,
        market: Market,
        order_books: dict[str, OrderBook],
        positions: dict[str, Position],
    ) -> list[Signal]:
        """Generate bid and ask quotes for a market."""
        signals = []

        for token in market.tokens:
            book = order_books.get(token.token_id)
            if book is None or book.mid_price is None:
                continue

            # Skip if book is too thin
            if book.bid_depth < self.config.min_book_depth_usd:
                continue
            if book.ask_depth < self.config.min_book_depth_usd:
                continue

            mid = book.mid_price
            half_spread = self.config.spread / 2

            # ── Inventory Skew ───────────────────────────────────
            # If we're long, skew asks lower (eager to sell)
            # If we're short, skew bids higher (eager to buy)
            inventory = 0.0
            pos = positions.get(token.token_id)
            if pos is not None:
                inventory = pos.size
                # Check inventory limits
                if abs(inventory * mid) > self.config.max_inventory:
                    # Only quote on the reducing side
                    if inventory > 0:
                        # Only ask (sell)
                        ask_price = round_price(mid + half_spread * 0.5)
                        ask_price = clamp(ask_price, 0.02, 0.98)
                        signals.append(self._make_signal(
                            market, token, Side.SELL, ask_price, mid, "reduce_inventory"
                        ))
                    continue

            skew = self.config.inventory_skew_factor * (inventory * mid / max(self.config.max_inventory, 1.0))

            bid_price = round_price(mid - half_spread + skew)
            ask_price = round_price(mid + half_spread + skew)

            bid_price = clamp(bid_price, 0.01, 0.99)
            ask_price = clamp(ask_price, 0.01, 0.99)

            # Ensure bid < ask
            if bid_price >= ask_price:
                continue

            # ── Generate Bid Signal ──────────────────────────────
            signals.append(self._make_signal(
                market, token, Side.BUY, bid_price, mid, "mm_bid"
            ))

            # ── Generate Ask Signal ──────────────────────────────
            signals.append(self._make_signal(
                market, token, Side.SELL, ask_price, mid, "mm_ask"
            ))

        return signals

    def _make_signal(
        self,
        market: Market,
        token: Any,
        side: Side,
        price: float,
        mid: float,
        reason: str,
    ) -> Signal:
        edge = abs(mid - price)
        return Signal(
            market_condition_id=market.condition_id,
            token_id=token.token_id,
            side=side,
            outcome=token.outcome,
            estimated_fair_value=mid,
            market_price=price,
            edge=edge,
            confidence=0.6,  # MM signals have moderate confidence
            strategy=self.name,
            metadata={
                "reason": reason,
                "spread": self.config.spread,
                "order_size": self.config.order_size_usd,
            },
        )
