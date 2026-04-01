"""Market simulator for backtesting.

Generates realistic simulated market data including price paths,
order books, and sentiment data for testing strategies without
needing live market access.

Uses geometric Brownian motion for price paths with mean-reverting
tendencies (since prediction market prices are bounded 0-1 and
tend toward the true probability over time).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from polymarket_bot.data.models import (
    Market,
    MarketCategory,
    OrderBook,
    OrderBookLevel,
    SentimentData,
    Token,
)


@dataclass
class SimulatedMarket:
    """A simulated market with known true probability."""

    market: Market
    true_probability: float  # The actual probability (known to simulator)
    volatility: float = 0.02  # Daily price volatility
    price_history: list[float] = field(default_factory=list)


class MarketSimulator:
    """Generate realistic simulated market data for backtesting."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.RandomState(seed)
        self.random = random.Random(seed)

    def create_simulated_markets(self, count: int = 10) -> list[SimulatedMarket]:
        """Create a set of simulated markets with known true probabilities."""
        markets = []
        categories = [
            ("crypto", MarketCategory.CRYPTO, [
                "Will Bitcoin be above $100,000 by end of Q2?",
                "Will Ethereum reach $5,000 this month?",
                "Will Solana flip Ethereum in daily volume?",
            ]),
            ("politics", MarketCategory.POLITICS, [
                "Will the incumbent win the next presidential election?",
                "Will the Senate pass the infrastructure bill?",
                "Will the governor win re-election?",
            ]),
            ("sports", MarketCategory.SPORTS, [
                "Will the Lakers win the NBA Championship?",
                "Will Team A beat Team B in the finals?",
                "Will the underdog win the Super Bowl?",
            ]),
        ]

        for i in range(count):
            cat_name, cat_enum, questions = categories[i % len(categories)]
            question = questions[i % len(questions)]

            # True probability (what will actually happen)
            true_prob = self.rng.beta(2, 2)  # Beta distribution, centered around 0.5
            true_prob = max(0.05, min(0.95, true_prob))

            # Initial market price (may differ from true prob - this is the edge)
            noise = self.rng.normal(0, 0.08)
            initial_price = max(0.05, min(0.95, true_prob + noise))

            market = Market(
                condition_id=f"sim_{i:04d}",
                question=question,
                slug=f"sim-market-{i}",
                tokens=[
                    Token(token_id=f"sim_{i:04d}_yes", outcome="Yes", price=initial_price),
                    Token(token_id=f"sim_{i:04d}_no", outcome="No", price=1.0 - initial_price),
                ],
                category=cat_enum,
                end_date=datetime.utcnow() + timedelta(days=30),
                volume_24h=self.rng.uniform(1000, 100000),
                liquidity=self.rng.uniform(500, 50000),
                active=True,
                tags=[cat_name],
            )

            sim = SimulatedMarket(
                market=market,
                true_probability=true_prob,
                volatility=self.rng.uniform(0.01, 0.05),
                price_history=[initial_price],
            )
            markets.append(sim)

        return markets

    def step_prices(self, sim_markets: list[SimulatedMarket]) -> None:
        """Advance prices by one time step.

        Uses mean-reverting geometric Brownian motion:
        - Random walk component (noise)
        - Mean reversion toward true probability (information flow)
        """
        for sim in sim_markets:
            current = sim.price_history[-1]

            # Mean reversion toward true probability
            reversion = 0.02 * (sim.true_probability - current)

            # Random noise
            noise = self.rng.normal(0, sim.volatility)

            # New price
            new_price = current + reversion + noise
            new_price = max(0.02, min(0.98, new_price))

            sim.price_history.append(new_price)

            # Update market tokens
            for token in sim.market.tokens:
                if token.outcome == "Yes":
                    token.price = new_price
                else:
                    token.price = 1.0 - new_price

    def generate_order_book(self, sim: SimulatedMarket, depth: int = 5) -> dict[str, OrderBook]:
        """Generate a realistic order book around current price."""
        books = {}
        for token in sim.market.tokens:
            mid = token.price
            spread = self.rng.uniform(0.01, 0.04)

            bids = []
            asks = []
            for i in range(depth):
                bid_price = max(0.01, mid - spread / 2 - i * 0.01)
                ask_price = min(0.99, mid + spread / 2 + i * 0.01)
                bid_size = self.rng.uniform(10, 200) * (depth - i) / depth
                ask_size = self.rng.uniform(10, 200) * (depth - i) / depth

                bids.append(OrderBookLevel(price=round(bid_price, 2), size=round(bid_size, 2)))
                asks.append(OrderBookLevel(price=round(ask_price, 2), size=round(ask_size, 2)))

            bids.sort(key=lambda x: x.price, reverse=True)
            asks.sort(key=lambda x: x.price)

            books[token.token_id] = OrderBook(
                token_id=token.token_id,
                bids=bids,
                asks=asks,
            )

        return books

    def generate_sentiment(
        self, sim: SimulatedMarket, noise_level: float = 0.3
    ) -> SentimentData:
        """Generate simulated sentiment data.

        Sentiment is correlated with the true probability but with noise.
        """
        # Base sentiment from true probability (>0.5 = bullish)
        base_sentiment = (sim.true_probability - 0.5) * 2

        # Add noise
        noisy_sentiment = base_sentiment + self.rng.normal(0, noise_level)
        noisy_sentiment = max(-1.0, min(1.0, noisy_sentiment))

        tweet_count = int(self.rng.uniform(5, 100))
        volume_ratio = self.rng.uniform(0.5, 3.0)

        bullish = max(0, (noisy_sentiment + 1) / 2)
        bearish = 1.0 - bullish

        return SentimentData(
            query=sim.market.question[:50],
            tweet_count=tweet_count,
            avg_sentiment=noisy_sentiment,
            sentiment_std=noise_level,
            volume_ratio=volume_ratio,
            bullish_pct=bullish,
            bearish_pct=bearish,
        )
