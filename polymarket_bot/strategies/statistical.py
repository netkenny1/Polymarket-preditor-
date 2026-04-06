"""Statistical modeling strategy for Polymarket.

Core idea: Build probability models for events using real data
(polls, ELO ratings, historical patterns) and trade when our
model disagrees with the market price.

Approaches by category:
- Politics: Polling aggregation with recency/quality weighting,
  fundamentals-based adjustments
- Sports: ELO ratings, head-to-head records, home advantage
- Crypto: Mean reversion of price prediction markets, on-chain metrics

This is the bread-and-butter strategy for serious Polymarket traders.
Nate Silver-style modeling applied to prediction markets.
"""

from __future__ import annotations

from typing import Any

import structlog

from polymarket_bot.clients.odds_sources import EloRating, OddsAggregator, PollData
from polymarket_bot.data.models import Market, MarketCategory, OrderBook, Side, Signal
from polymarket_bot.strategies.base import BaseStrategy
from polymarket_bot.utils.helpers import clamp

logger = structlog.get_logger()


class StatisticalStrategy(BaseStrategy):
    """Trade based on statistical models vs market prices."""

    name = "statistical"

    def __init__(self, odds_aggregator: OddsAggregator, min_edge: float = 0.05) -> None:
        self.odds = odds_aggregator
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
                if market.category == MarketCategory.POLITICS:
                    signal = self._analyze_political(market, context)
                elif market.category == MarketCategory.SPORTS:
                    signal = self._analyze_sports(market, context)
                elif market.category == MarketCategory.CRYPTO:
                    signal = self._analyze_crypto(market, order_books, context)
                else:
                    signal = self._analyze_generic(market, order_books, context)

                if signal is not None:
                    signals.append(signal)
            except Exception as e:
                logger.error("statistical_error", market=market.condition_id, error=str(e))

        return signals

    def _analyze_political(self, market: Market, context: dict[str, Any]) -> Signal | None:
        """Analyze political markets using polling data.

        Aggregates polls with recency and quality weighting, then compares
        the model probability to the market price.
        """
        polls: list[PollData] = context.get(f"polls_{market.condition_id}", [])
        if not polls:
            return None

        # Extract candidates from market tokens
        candidates = [t.outcome for t in market.tokens]
        polling_avg = self.odds.get_polling_average(candidates, polls)

        # Find the largest edge
        best_edge = 0.0
        best_token = None
        best_fair_value = 0.5

        for token in market.tokens:
            model_prob = polling_avg.get(token.outcome, 0.5)
            edge = model_prob - token.price
            if abs(edge) > abs(best_edge):
                best_edge = edge
                best_token = token
                best_fair_value = model_prob

        if best_token is None or abs(best_edge) < self.min_edge:
            return None

        # On Polymarket's CLOB, buy the complement token instead of selling
        if best_edge > 0:
            side = Side.BUY
            token = best_token
            fair_value = best_fair_value
            market_price = best_token.price
            edge = abs(best_edge)
        else:
            complement = next(
                (t for t in market.tokens if t.token_id != best_token.token_id), None
            )
            if complement is None:
                return None
            side = Side.BUY
            token = complement
            fair_value = 1.0 - best_fair_value
            market_price = complement.price
            edge = abs(best_edge)

        confidence = min(0.9, len(polls) / 20)

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
            metadata={"model": "polling_aggregation", "num_polls": len(polls)},
        )

    def _analyze_sports(self, market: Market, context: dict[str, Any]) -> Signal | None:
        """Analyze sports markets using ELO ratings.

        Computes win probabilities from ELO ratings and compares to
        market prices.
        """
        elo_ratings: dict[str, EloRating] = context.get(f"elo_{market.condition_id}", {})
        if len(elo_ratings) < 2:
            return None

        teams = list(elo_ratings.values())[:2]
        probs = self.odds.get_elo_fair_value(teams[0], teams[1])

        best_edge = 0.0
        best_token = None
        best_fair_value = 0.5

        for token in market.tokens:
            model_prob = probs.get(token.outcome, 0.5)
            edge = model_prob - token.price
            if abs(edge) > abs(best_edge):
                best_edge = edge
                best_token = token
                best_fair_value = model_prob

        if best_token is None or abs(best_edge) < self.min_edge:
            return None

        if best_edge > 0:
            side = Side.BUY
            token = best_token
            fair_value = best_fair_value
            market_price = best_token.price
            edge = abs(best_edge)
        else:
            complement = next(
                (t for t in market.tokens if t.token_id != best_token.token_id), None
            )
            if complement is None:
                return None
            side = Side.BUY
            token = complement
            fair_value = 1.0 - best_fair_value
            market_price = complement.price
            edge = abs(best_edge)

        confidence = 0.7

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
            metadata={"model": "elo", "ratings": {t.team: t.rating for t in teams}},
        )

    def _analyze_crypto(
        self, market: Market, order_books: dict[str, OrderBook], context: dict[str, Any]
    ) -> Signal | None:
        """Analyze crypto markets using mean reversion and momentum.

        Crypto prediction markets (e.g., "Will BTC be above $X by date Y?")
        tend to overreact to short-term price moves. We use mean reversion
        combined with volatility-adjusted fair value estimation.
        """
        price_history: list[float] = context.get(f"price_history_{market.condition_id}", [])
        if len(price_history) < 5:
            return None

        # Simple mean reversion: compare current price to moving average
        current = price_history[-1]
        ma = sum(price_history[-20:]) / min(len(price_history), 20)

        # Z-score of current price relative to recent history
        import numpy as np
        arr = np.array(price_history[-20:])
        std = np.std(arr)
        if std < 0.01:
            return None

        z_score = (current - ma) / std

        # Market's YES token
        yes_token = next((t for t in market.tokens if t.outcome == "Yes"), None)
        if yes_token is None:
            return None

        market_price = yes_token.price

        # If price has spiked up (z > 1.5), market probably overreacted -> YES overpriced
        # If price has dropped (z < -1.5), market probably overreacted -> YES underpriced
        if abs(z_score) < 1.0:
            return None  # Not enough deviation

        # Estimate fair value using mean reversion expectation
        reversion_factor = 0.3  # Expect 30% reversion
        if z_score > 0:
            # Price spiked up, YES probably overpriced
            adjustment = -reversion_factor * (z_score / 3.0) * 0.15
        else:
            # Price dropped, YES probably underpriced
            adjustment = -reversion_factor * (z_score / 3.0) * 0.15

        fair_value = clamp(market_price + adjustment, 0.05, 0.95)
        edge = fair_value - market_price

        if abs(edge) < self.min_edge:
            return None

        side = Side.BUY if edge > 0 else Side.SELL
        token = yes_token if edge > 0 else next((t for t in market.tokens if t.outcome == "No"), yes_token)

        if edge < 0:
            fair_value = 1.0 - fair_value
            market_price = 1.0 - market_price
            edge = abs(edge)
            side = Side.BUY

        return Signal(
            market_condition_id=market.condition_id,
            token_id=token.token_id,
            side=side,
            outcome=token.outcome,
            estimated_fair_value=fair_value,
            market_price=market_price,
            edge=edge,
            confidence=min(0.8, abs(z_score) / 3.0),
            strategy=self.name,
            metadata={"model": "mean_reversion", "z_score": float(z_score)},
        )

    def _analyze_generic(
        self, market: Market, order_books: dict[str, OrderBook], context: dict[str, Any]
    ) -> Signal | None:
        """Analyze markets without specialized models.

        Uses order book imbalance as a signal: if there is significantly
        more buying pressure than selling pressure, the price likely moves up.
        """
        yes_token = next((t for t in market.tokens if t.outcome == "Yes"), None)
        if yes_token is None:
            return None

        book = order_books.get(yes_token.token_id)
        if book is None or book.mid_price is None:
            return None

        # Order book imbalance
        bid_depth = book.bid_depth
        ask_depth = book.ask_depth
        total_depth = bid_depth + ask_depth

        if total_depth < 100:
            return None

        imbalance = (bid_depth - ask_depth) / total_depth  # -1 to 1

        if abs(imbalance) < 0.3:
            return None  # Not significant enough

        # Strong bid imbalance -> price likely to increase -> buy YES
        adjustment = imbalance * 0.08
        market_price = yes_token.price
        fair_value = clamp(market_price + adjustment, 0.05, 0.95)
        edge = fair_value - market_price

        if abs(edge) < self.min_edge:
            return None

        side = Side.BUY if edge > 0 else Side.SELL
        token = yes_token if edge > 0 else next((t for t in market.tokens if t.outcome == "No"), yes_token)
        if edge < 0:
            fair_value = 1.0 - fair_value
            market_price = 1.0 - market_price
            edge = abs(edge)
            side = Side.BUY

        return Signal(
            market_condition_id=market.condition_id,
            token_id=token.token_id,
            side=side,
            outcome=token.outcome,
            estimated_fair_value=fair_value,
            market_price=market_price,
            edge=edge,
            confidence=min(0.6, abs(imbalance)),
            strategy=self.name,
            metadata={"model": "book_imbalance", "imbalance": float(imbalance)},
        )
