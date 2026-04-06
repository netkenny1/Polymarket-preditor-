"""Arbitrage strategy for Polymarket.

Two types of arbitrage:

1. Complement Arbitrage: When outcomes in a multi-outcome market don't
   sum to 1.0, you can buy/sell all outcomes for a guaranteed profit.
   This happens frequently on Polymarket due to fragmented liquidity.

2. Cross-Market Arbitrage: When the same event is priced differently
   on Polymarket vs external sources (other prediction markets,
   sportsbooks, polling aggregators).

These are the lowest-risk strategies - pure arbitrage has no directional
risk when executed correctly.
"""

from __future__ import annotations

from typing import Any, Optional

import structlog

from polymarket_bot.clients.odds_sources import OddsAggregator, ExternalOdds
from polymarket_bot.config import ArbitrageConfig
from polymarket_bot.data.models import Market, OrderBook, Side, Signal
from polymarket_bot.strategies.base import BaseStrategy

logger = structlog.get_logger()


class ArbitrageStrategy(BaseStrategy):
    """Find and exploit arbitrage opportunities."""

    name = "arbitrage"

    def __init__(self, config: ArbitrageConfig, odds_aggregator: OddsAggregator) -> None:
        self.config = config
        self.odds = odds_aggregator

    def generate_signals(
        self,
        markets: list[Market],
        order_books: dict[str, OrderBook],
        context: dict[str, Any],
    ) -> list[Signal]:
        signals = []

        for market in markets:
            if not market.active:
                continue
            try:
                # Check complement arbitrage
                comp_signals = self._check_complement_arb(market, order_books)
                signals.extend(comp_signals)

                # Check cross-market arbitrage
                external_odds = context.get(f"external_odds_{market.condition_id}", [])
                if external_odds:
                    cross_signals = self._check_cross_market_arb(market, external_odds)
                    signals.extend(cross_signals)
            except Exception as e:
                logger.error("arb_error", market=market.condition_id, error=str(e))

        return signals

    def _check_complement_arb(
        self,
        market: Market,
        order_books: dict[str, OrderBook],
    ) -> list[Signal]:
        """Check if outcome prices don't sum to 1.0 (complement arb)."""
        if len(market.tokens) < 2:
            return []

        # Require live order books for every outcome (no token.price fallback).
        outcome_prices: dict[str, float] = {}
        outcome_depths: dict[str, float] = {}
        token_map: dict[str, Any] = {}

        for token in market.tokens:
            book = order_books.get(token.token_id)
            if book is None or book.best_ask is None:
                return []
            outcome_prices[token.outcome] = book.best_ask
            outcome_depths[token.outcome] = book.ask_depth
            token_map[token.outcome] = token

        arb = self.odds.find_complement_arb(
            outcome_prices, tolerance=self.config.complement_tolerance
        )

        if arb is None:
            return []

        min_depth = min(outcome_depths.values())
        if min_depth < 10.0:
            return []

        gross_profit_pct = arb["profit_pct"]
        fee_cost_pct = 0.02 * len(market.tokens)
        net_profit_pct = gross_profit_pct - fee_cost_pct
        if net_profit_pct <= 0.005:
            return []

        logger.info(
            "complement_arb_found",
            market=market.condition_id,
            type=arb["type"],
            profit_pct=gross_profit_pct,
            net_profit_pct=net_profit_pct,
            min_depth=min_depth,
        )

        n_tokens = len(market.tokens)
        edge_per_leg = net_profit_pct / n_tokens
        signals = []
        if arb["type"] == "underpriced_complement":
            # Buy all outcomes
            for outcome, price in arb["outcomes"].items():
                token = token_map.get(outcome)
                if token is None:
                    continue
                signals.append(Signal(
                    market_condition_id=market.condition_id,
                    token_id=token.token_id,
                    side=Side.BUY,
                    outcome=outcome,
                    estimated_fair_value=1.0 / n_tokens,
                    market_price=price,
                    edge=edge_per_leg,
                    confidence=0.95,  # Arb is near-certain
                    strategy=self.name,
                    metadata={
                        "arb_type": "complement",
                        "total_cost": arb["total_cost"],
                        "polymarket_depth_shares": min_depth,
                    },
                ))
        elif arb["type"] == "overpriced_complement":
            # Sell all outcomes
            for outcome, price in arb["outcomes"].items():
                token = token_map.get(outcome)
                if token is None:
                    continue
                signals.append(Signal(
                    market_condition_id=market.condition_id,
                    token_id=token.token_id,
                    side=Side.SELL,
                    outcome=outcome,
                    estimated_fair_value=1.0 / n_tokens,
                    market_price=price,
                    edge=edge_per_leg,
                    confidence=0.95,
                    strategy=self.name,
                    metadata={
                        "arb_type": "complement",
                        "total_revenue": arb["total_revenue"],
                        "polymarket_depth_shares": min_depth,
                    },
                ))

        return signals

    def _check_cross_market_arb(
        self,
        market: Market,
        external_odds: list[ExternalOdds],
    ) -> list[Signal]:
        """Compare Polymarket price vs external sources."""
        signals = []

        for token in market.tokens:
            # Find matching external odds for this outcome
            matching = [o for o in external_odds if o.outcome.lower() == token.outcome.lower()]
            if not matching:
                continue

            result = self.odds.find_cross_market_edge(
                polymarket_price=token.price,
                external_odds=matching,
                min_edge=self.config.min_arb_edge,
            )

            if result is None:
                continue

            logger.info(
                "cross_market_edge",
                market=market.condition_id,
                outcome=token.outcome,
                edge=result["edge"],
                direction=result["direction"],
            )

            side = Side.BUY if result["direction"] == "BUY" else Side.SELL
            signals.append(Signal(
                market_condition_id=market.condition_id,
                token_id=token.token_id,
                side=side,
                outcome=token.outcome,
                estimated_fair_value=result["external_avg"],
                market_price=token.price,
                edge=abs(result["edge"]),
                confidence=result["confidence"],
                strategy=self.name,
                metadata={"arb_type": "cross_market", "sources": result["sources"]},
            ))

        return signals
