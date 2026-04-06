"""Latency arbitrage: exploit the ~2.7s BTC price delay from Binance to Polymarket.

When BTC moves on Binance, Polymarket market makers take ~2.7 seconds to
update their quotes. This strategy detects the move on Binance first, identifies
stale Polymarket prices, and trades before the update propagates.

The edge is structural: Polymarket's market makers poll centralized exchange
prices with a delay, creating a persistent information asymmetry window.

Dual-mode architecture:
  Fast path (event-driven): 100ms tick loop, bypasses normal aggregation
  Slow path (poll-based):   Standard generate_signals() for residual staleness
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from typing import Any

import structlog

from polymarket_bot.clients.binance_feed import BinancePriceTracker
from polymarket_bot.config import LatencyArbitrageConfig
from polymarket_bot.data.models import Market, OrderBook, Side, Signal
from polymarket_bot.strategies.base import BaseStrategy
from polymarket_bot.utils.helpers import clamp

logger = structlog.get_logger()

_BTC_KEYWORDS = {"btc", "bitcoin"}


class LatencyArbitrageStrategy(BaseStrategy):
    """Exploit Binance-to-Polymarket BTC price propagation delay."""

    name = "latency_arb"

    def __init__(
        self,
        config: LatencyArbitrageConfig,
        price_tracker: BinancePriceTracker,
        polymarket_client: Any = None,
    ) -> None:
        self.config = config
        self.tracker = price_tracker
        self.pm_client = polymarket_client

        self._btc_markets: list[Market] = []
        self._last_trade_time: dict[str, float] = {}
        self._daily_trade_count: int = 0
        self._daily_pnl: float = 0.0
        self._last_day_reset: float = time.monotonic()

    # ── Fast Path: Event-Driven Execution ─────────────────────

    async def run_fast_loop(self) -> None:
        """Main async loop — checks for opportunities every 100ms."""
        while True:
            try:
                self._maybe_reset_daily()

                if self._daily_trade_count >= self.config.max_daily_trades:
                    await asyncio.sleep(60)
                    continue

                move_pct = self.tracker.get_move_pct()
                if abs(move_pct) >= self.config.move_threshold_pct / 100:
                    await self._scan_and_trade(move_pct)

                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error("latency_arb_fast_loop_error", error=str(e))
                await asyncio.sleep(1.0)

    async def _scan_and_trade(self, move_pct: float) -> None:
        direction = 1 if move_pct > 0 else -1

        for market in self._btc_markets:
            if not self._can_trade_market(market):
                continue

            signal = await self._check_staleness(market, direction, move_pct)
            if signal is not None:
                await self._fast_execute(signal)

    async def _check_staleness(
        self, market: Market, direction: int, move_pct: float,
    ) -> Signal | None:
        """Check if a Polymarket BTC market has stale prices."""
        yes_token = next((t for t in market.tokens if t.outcome == "Yes"), None)
        no_token = next((t for t in market.tokens if t.outcome == "No"), None)
        if not yes_token or not no_token:
            return None

        if self.pm_client is None:
            return None

        book = await asyncio.to_thread(
            self.pm_client.get_order_book, yes_token.token_id
        )
        if book.mid_price is None or book.spread is None:
            return None

        polymarket_mid = book.mid_price
        polymarket_spread = book.spread

        threshold = self._extract_btc_threshold(market)
        if threshold is None:
            return None

        binance_price = self.tracker.current_price
        current_implied = self._btc_to_implied_prob(binance_price, threshold)

        stale_edge = current_implied - polymarket_mid
        if direction > 0 and stale_edge <= 0:
            return None
        if direction < 0 and stale_edge >= 0:
            return None

        net_edge = abs(stale_edge) - polymarket_spread / 2 - 0.002
        if net_edge < self.config.min_edge_cents:
            return None

        if stale_edge > 0:
            token = yes_token
            fair_value = current_implied
            market_price = book.best_ask or polymarket_mid
        else:
            token = no_token
            fair_value = 1.0 - current_implied
            market_price = 1.0 - (book.best_bid or polymarket_mid)

        confidence = self._calculate_confidence(move_pct, net_edge, book)

        return Signal(
            market_condition_id=market.condition_id,
            token_id=token.token_id,
            side=Side.BUY,
            outcome=token.outcome,
            estimated_fair_value=fair_value,
            market_price=market_price,
            edge=net_edge,
            confidence=confidence,
            strategy=self.name,
            metadata={
                "binance_price": binance_price,
                "binance_move_pct": round(move_pct * 100, 3),
                "btc_threshold": threshold,
                "polymarket_mid": polymarket_mid,
                "polymarket_spread": polymarket_spread,
                "stale_edge_raw": round(abs(stale_edge), 4),
                "net_edge": round(net_edge, 4),
                "velocity": round(self.tracker.get_velocity(), 2),
            },
        )

    async def _fast_execute(self, signal: Signal) -> None:
        """Execute immediately, bypassing normal aggregation for speed."""
        if self.pm_client is None:
            return

        price = signal.market_price + self.config.price_aggression
        price = max(0.01, min(0.99, round(price, 2)))

        size_usd = min(
            self.config.max_position_usd,
            signal.edge * signal.confidence * 500,
        )
        shares = round(size_usd / price, 2) if price > 0 else 0

        if shares <= 0:
            return

        logger.info(
            "latency_arb_fast_execute",
            market=signal.market_condition_id,
            side=signal.side.value,
            price=price,
            shares=shares,
            edge=round(signal.edge, 4),
            binance_price=signal.metadata.get("binance_price"),
        )

        result = await asyncio.to_thread(
            self.pm_client.place_order,
            token_id=signal.token_id,
            side=signal.side,
            price=price,
            size=shares,
            market_condition_id=signal.market_condition_id,
            strategy=self.name,
        )

        if result and result.success:
            self._last_trade_time[signal.market_condition_id] = time.monotonic()
            self._daily_trade_count += 1

    # ── Slow Path: Standard Strategy Interface ────────────────

    def generate_signals(
        self,
        markets: list[Market],
        order_books: dict[str, OrderBook],
        context: dict[str, Any],
    ) -> list[Signal]:
        """Standard interface for the normal bot loop (residual sweep)."""
        signals: list[Signal] = []

        binance_price = self.tracker.current_price
        if binance_price <= 0:
            return signals

        self._btc_markets = [
            m for m in markets if m.active and self._is_btc_market(m)
        ]

        for market in self._btc_markets:
            yes_token = next((t for t in market.tokens if t.outcome == "Yes"), None)
            if not yes_token:
                continue

            book = order_books.get(yes_token.token_id)
            if book is None or book.mid_price is None:
                continue

            threshold = self._extract_btc_threshold(market)
            if threshold is None:
                continue

            implied = self._btc_to_implied_prob(binance_price, threshold)
            edge = abs(implied - book.mid_price) - (book.spread or 0.02) / 2 - 0.002

            if edge < self.config.min_edge_cents:
                continue

            if implied > book.mid_price:
                token = yes_token
                fair_value = implied
                market_price = book.best_ask or book.mid_price
            else:
                no_token = next((t for t in market.tokens if t.outcome == "No"), None)
                if not no_token:
                    continue
                token = no_token
                fair_value = 1.0 - implied
                market_price = 1.0 - (book.best_bid or book.mid_price)

            signals.append(Signal(
                market_condition_id=market.condition_id,
                token_id=token.token_id,
                side=Side.BUY,
                outcome=token.outcome,
                estimated_fair_value=fair_value,
                market_price=market_price,
                edge=edge,
                confidence=0.60,
                strategy=self.name,
                metadata={
                    "source": "slow_path",
                    "binance_price": binance_price,
                    "btc_threshold": threshold,
                },
            ))

        return signals

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _is_btc_market(market: Market) -> bool:
        text = market.question.lower()
        return any(kw in text for kw in _BTC_KEYWORDS)

    @staticmethod
    def _extract_btc_threshold(market: Market) -> float | None:
        """Parse BTC price threshold from market question text."""
        text = market.question.replace(",", "")
        match = re.search(r'\$?([\d]+(?:\.\d+)?)', text)
        if match:
            val = float(match.group(1))
            if val > 1000:
                return val
        return None

    @staticmethod
    def _btc_to_implied_prob(btc_price: float, threshold: float) -> float:
        """Convert BTC spot price to implied probability of closing above threshold.

        Uses a logistic model calibrated to ~2% daily BTC volatility.
        """
        daily_vol = 0.02 * threshold
        if daily_vol == 0:
            return 0.5
        z = (btc_price - threshold) / daily_vol
        steepness = 3.0
        prob = 1.0 / (1.0 + math.exp(-steepness * z))
        return clamp(prob, 0.02, 0.98)

    def _calculate_confidence(
        self, move_pct: float, net_edge: float, book: OrderBook,
    ) -> float:
        conf = 0.50
        if abs(move_pct) > 0.003:
            conf += 0.10
        if abs(move_pct) > 0.005:
            conf += 0.10
        if net_edge > 0.03:
            conf += 0.05
        if net_edge > 0.05:
            conf += 0.05
        depth = book.bid_depth + book.ask_depth
        if depth > 1000:
            conf += 0.05
        velocity = abs(self.tracker.get_velocity())
        if velocity > 50:
            conf += 0.05
        return clamp(conf, 0.30, 0.85)

    def _can_trade_market(self, market: Market) -> bool:
        now = time.monotonic()
        last = self._last_trade_time.get(market.condition_id, 0)
        if now - last < self.config.cooldown_seconds:
            return False
        if self._daily_trade_count >= self.config.max_daily_trades:
            return False
        return True

    def _maybe_reset_daily(self) -> None:
        now = time.monotonic()
        if now - self._last_day_reset > 86400:
            self._daily_trade_count = 0
            self._daily_pnl = 0.0
            self._last_day_reset = now

    def refresh_markets(self, markets: list[Market]) -> None:
        self._btc_markets = [
            m for m in markets if m.active and self._is_btc_market(m)
        ]
        logger.info("latency_arb_markets_refreshed", count=len(self._btc_markets))
