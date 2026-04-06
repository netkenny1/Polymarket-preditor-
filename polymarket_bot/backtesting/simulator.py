"""Market simulator for backtesting.

Generates simulated market data including price paths, order books,
and sentiment data for testing strategies without live market access.

Two simulators are provided:

- MarketSimulator: Original implementation. Contains hindsight bias
  (prices mean-revert toward the true probability; sentiment is
  correlated with the true outcome). Retained for backward compat.

- RealisticMarketSimulator: Bias-free replacement. Price dynamics
  are a bounded random walk with regime shifts and news jumps.
  Sentiment follows recent price trends. Order books are persistent
  across steps. true_probability is used ONLY at resolution time.
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

            if i % 5 == 4:
                expiry_days = self.rng.uniform(1, 5)
            elif i % 5 == 3:
                expiry_days = self.rng.uniform(5, 14)
            else:
                expiry_days = 30

            market = Market(
                condition_id=f"sim_{i:04d}",
                question=question,
                slug=f"sim-market-{i}",
                tokens=[
                    Token(token_id=f"sim_{i:04d}_yes", outcome="Yes", price=initial_price),
                    Token(token_id=f"sim_{i:04d}_no", outcome="No", price=1.0 - initial_price),
                ],
                category=cat_enum,
                end_date=datetime.utcnow() + timedelta(days=expiry_days),
                volume_24h=self.rng.uniform(1000, 100000),
                liquidity=self.rng.uniform(500, 50000),
                active=True,
                tags=[cat_name],
            )

            sim = SimulatedMarket(
                market=market,
                true_probability=true_prob,
                volatility=self.rng.uniform(0.02, 0.08),
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
            reversion = 0.005 * (sim.true_probability - current)

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
            spread = self.rng.uniform(0.01, 0.06)

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


# ---------------------------------------------------------------------------
# Realistic (bias-free) simulator
# ---------------------------------------------------------------------------


@dataclass
class _VolatilityRegime:
    """Tracks temporary volatility regime shifts for a single market."""

    multiplier: float = 1.0
    steps_remaining: int = 0


@dataclass
class _PersistentBook:
    """Mutable order book state that persists across simulation steps."""

    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)


class RealisticMarketSimulator:
    """Bias-free market simulator for honest backtesting.

    Key differences from MarketSimulator:
    - Prices follow a bounded random walk — NO mean-reversion toward
      true_probability.
    - Occasional random jumps simulate news events.
    - Weak mean-reversion toward 0.5 only (uninformed prior).
    - Sentiment is derived from recent price trends, NOT the true outcome.
    - Order books are persistent — orders are partially cancelled and
      replenished each step instead of regenerated from scratch.
    - true_probability is used ONLY in resolve_market() at the end.
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.RandomState(seed)
        self.random = random.Random(seed)

        self._regimes: dict[str, _VolatilityRegime] = {}
        self._books: dict[str, _PersistentBook] = {}

    # ------------------------------------------------------------------
    # Market creation (same interface as MarketSimulator)
    # ------------------------------------------------------------------

    def create_simulated_markets(self, count: int = 10) -> list[SimulatedMarket]:
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

        markets: list[SimulatedMarket] = []
        for i in range(count):
            cat_name, cat_enum, questions = categories[i % len(categories)]
            question = questions[i % len(questions)]

            true_prob = float(np.clip(self.rng.beta(2, 2), 0.05, 0.95))

            # Initial price: intentionally offset from true_prob — this is
            # the market's *starting mispricing*, NOT a signal the bot sees.
            noise = self.rng.normal(0, 0.08)
            initial_price = float(np.clip(true_prob + noise, 0.05, 0.95))

            if i % 5 == 4:
                expiry_days = self.rng.uniform(1, 5)
            elif i % 5 == 3:
                expiry_days = self.rng.uniform(5, 14)
            else:
                expiry_days = 30

            market = Market(
                condition_id=f"sim_{i:04d}",
                question=question,
                slug=f"sim-market-{i}",
                tokens=[
                    Token(token_id=f"sim_{i:04d}_yes", outcome="Yes", price=initial_price),
                    Token(token_id=f"sim_{i:04d}_no", outcome="No", price=1.0 - initial_price),
                ],
                category=cat_enum,
                end_date=datetime.utcnow() + timedelta(days=expiry_days),
                volume_24h=self.rng.uniform(1000, 100000),
                liquidity=self.rng.uniform(500, 50000),
                active=True,
                tags=[cat_name],
            )

            sim = SimulatedMarket(
                market=market,
                true_probability=true_prob,
                volatility=self.rng.uniform(0.02, 0.08),
                price_history=[initial_price],
            )
            markets.append(sim)

            self._regimes[market.condition_id] = _VolatilityRegime()
            self._init_book(sim)

        return markets

    # ------------------------------------------------------------------
    # Price dynamics — bounded random walk, NO true_probability
    # ------------------------------------------------------------------

    @staticmethod
    def _logit(p: float) -> float:
        p = max(1e-6, min(1 - 1e-6, p))
        return float(np.log(p / (1.0 - p)))

    @staticmethod
    def _sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    def step_prices(self, sim_markets: list[SimulatedMarket]) -> None:
        """Advance prices using a logit-space random walk (martingale).

        By evolving log-odds rather than probability directly, we:
        1. Keep prices strictly in (0, 1) without boundary clipping
        2. Maintain the martingale property (E[p_next] ≈ p_current)
        3. Prevent the systematic drift toward 0.5 that plagued
           bounded random walks
        """
        for sim in sim_markets:
            current = sim.price_history[-1]
            cid = sim.market.condition_id
            regime = self._regimes.get(cid, _VolatilityRegime())

            if regime.steps_remaining > 0:
                regime.steps_remaining -= 1
            elif self.rng.random() < 0.005:
                regime.multiplier = self.rng.choice([0.5, 2.0])
                regime.steps_remaining = 20
            else:
                regime.multiplier = 1.0
            self._regimes[cid] = regime

            effective_vol = sim.volatility * regime.multiplier

            logit_p = self._logit(current)

            noise = self.rng.normal(0, effective_vol * 2.0)

            jump = 0.0
            if self.rng.random() < 0.01:
                jump = self.rng.normal(0, 0.5)

            logit_new = logit_p + noise + jump
            logit_new = max(-6.0, min(6.0, logit_new))

            new_price = self._sigmoid(logit_new)
            new_price = max(0.02, min(0.98, new_price))

            sim.price_history.append(new_price)

            for token in sim.market.tokens:
                if token.outcome == "Yes":
                    token.price = new_price
                else:
                    token.price = 1.0 - new_price

    # ------------------------------------------------------------------
    # Persistent order book
    # ------------------------------------------------------------------

    def _init_book(self, sim: SimulatedMarket, levels: int = 10) -> None:
        """Seed a persistent order book for each token in *sim*."""
        for token in sim.market.tokens:
            mid = token.price
            spread = self.rng.uniform(0.01, 0.04)

            bids: list[OrderBookLevel] = []
            asks: list[OrderBookLevel] = []
            for j in range(levels):
                bp = max(0.01, mid - spread / 2 - j * 0.01)
                ap = min(0.99, mid + spread / 2 + j * 0.01)
                bs = float(self.rng.uniform(20, 300) * (levels - j) / levels)
                asv = float(self.rng.uniform(20, 300) * (levels - j) / levels)
                bids.append(OrderBookLevel(price=round(bp, 2), size=round(bs, 2)))
                asks.append(OrderBookLevel(price=round(ap, 2), size=round(asv, 2)))

            bids.sort(key=lambda x: x.price, reverse=True)
            asks.sort(key=lambda x: x.price)
            self._books[token.token_id] = _PersistentBook(bids=bids, asks=asks)

    def generate_order_book(
        self, sim: SimulatedMarket, depth: int = 5
    ) -> dict[str, OrderBook]:
        books: dict[str, OrderBook] = {}
        for token in sim.market.tokens:
            tid = token.token_id
            pb = self._books.get(tid)
            if pb is None:
                self._init_book(sim)
                pb = self._books[tid]

            mid = token.price

            # 1) Cancel 10-30 % of existing orders at random
            cancel_rate = self.rng.uniform(0.10, 0.30)
            pb.bids = [o for o in pb.bids if self.rng.random() > cancel_rate]
            pb.asks = [o for o in pb.asks if self.rng.random() > cancel_rate]

            # 2) Shift surviving orders toward the new mid-price
            if pb.bids:
                old_best_bid = max(o.price for o in pb.bids)
                shift = mid - (old_best_bid + 0.01)
                pb.bids = [
                    OrderBookLevel(
                        price=round(max(0.01, o.price + shift), 2),
                        size=o.size,
                    )
                    for o in pb.bids
                ]
            if pb.asks:
                old_best_ask = min(o.price for o in pb.asks)
                shift = mid - (old_best_ask - 0.01)
                pb.asks = [
                    OrderBookLevel(
                        price=round(min(0.99, o.price + shift), 2),
                        size=o.size,
                    )
                    for o in pb.asks
                ]

            # 3) Add new random orders to replenish depth
            target_levels = 10
            spread = self.rng.uniform(0.01, 0.04)
            while len(pb.bids) < target_levels:
                offset = self.rng.uniform(0.005, 0.10)
                bp = max(0.01, mid - spread / 2 - offset)
                bs = float(self.rng.uniform(10, 200))
                pb.bids.append(OrderBookLevel(price=round(bp, 2), size=round(bs, 2)))
            while len(pb.asks) < target_levels:
                offset = self.rng.uniform(0.005, 0.10)
                ap = min(0.99, mid + spread / 2 + offset)
                asv = float(self.rng.uniform(10, 200))
                pb.asks.append(OrderBookLevel(price=round(ap, 2), size=round(asv, 2)))

            # Sort & trim to requested depth
            pb.bids.sort(key=lambda x: x.price, reverse=True)
            pb.asks.sort(key=lambda x: x.price)

            books[tid] = OrderBook(
                token_id=tid,
                bids=pb.bids[:depth],
                asks=pb.asks[:depth],
            )

        return books

    # ------------------------------------------------------------------
    # Sentiment — follows PRICE TREND, not true_probability
    # ------------------------------------------------------------------

    def generate_sentiment(
        self, sim: SimulatedMarket, noise_level: float = 0.3
    ) -> SentimentData:
        current = sim.price_history[-1]

        lookback = 5
        if len(sim.price_history) >= lookback + 1:
            recent_trend = current - sim.price_history[-(lookback + 1)]
        else:
            recent_trend = 0.0

        base_sentiment = recent_trend * 3.0
        noisy_sentiment = float(
            np.clip(base_sentiment + self.rng.normal(0, noise_level), -1.0, 1.0)
        )

        tweet_count = int(self.rng.uniform(5, 100))
        volume_ratio = float(self.rng.uniform(0.5, 3.0))

        bullish = max(0.0, (noisy_sentiment + 1) / 2)
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

    # ------------------------------------------------------------------
    # Resolution — the ONLY place true_probability is consumed
    # ------------------------------------------------------------------

    def resolve_market(self, sim: SimulatedMarket) -> str:
        """Resolve market based on true probability. Called ONLY at simulation end."""
        return "Yes" if self.rng.random() < sim.true_probability else "No"
