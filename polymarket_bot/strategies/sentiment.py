"""Sentiment-based trading strategy using Twitter/X data.

Core idea: When Twitter sentiment diverges significantly from the market
price, the market is likely mispriced. Strong bullish sentiment with a
low YES price = buy signal. Volume spikes indicate breaking news that
hasn't been priced in yet.

This is one of the most profitable Polymarket strategies because:
1. Social media reacts faster than prediction markets
2. Retail traders on Polymarket are slow to update prices
3. Volume spikes reliably predict price movement direction
"""

from __future__ import annotations

from typing import Any

import structlog

from polymarket_bot.clients.twitter import TwitterClient
from polymarket_bot.config import SentimentConfig
from polymarket_bot.data.models import Market, OrderBook, Side, Signal
from polymarket_bot.strategies.base import BaseStrategy
from polymarket_bot.utils.helpers import clamp

logger = structlog.get_logger()


class SentimentStrategy(BaseStrategy):
    """Trade based on Twitter/X sentiment divergence from market prices.

    Generates BUY signals when:
    - Sentiment is strongly bullish AND market price is low
    - Tweet volume spikes (breaking news) with positive sentiment

    Generates SELL signals when:
    - Sentiment is strongly bearish AND market price is high
    - Tweet volume spikes with negative sentiment
    """

    name = "sentiment"

    def __init__(self, twitter_client: TwitterClient, config: SentimentConfig) -> None:
        self.twitter = twitter_client
        self.config = config

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
                signal = self._analyze_market(market, order_books, context)
                if signal is not None:
                    signals.append(signal)
            except Exception as e:
                logger.error("sentiment_analysis_error", market=market.condition_id, error=str(e))

        return signals

    def _analyze_market(
        self,
        market: Market,
        order_books: dict[str, OrderBook],
        context: dict[str, Any],
    ) -> Signal | None:
        """Analyze a single market for sentiment-based signals."""
        # Get sentiment data (may come from context if pre-fetched)
        sentiment_key = f"sentiment_{market.condition_id}"
        sentiment_data = context.get(sentiment_key)

        if sentiment_data is None:
            keywords = context.get(f"keywords_{market.condition_id}")
            sentiment_data = self.twitter.get_market_sentiment(
                market.question, keywords=keywords
            )

        if sentiment_data.tweet_count < self.config.min_tweets:
            return None  # Not enough data

        avg_sent = sentiment_data.avg_sentiment
        volume_ratio = sentiment_data.volume_ratio

        market_price = market.yes_price

        # ── Core Signal Logic ────────────────────────────────────

        # Estimate fair value from sentiment
        # Sentiment ranges -1 to 1, map to probability adjustment
        sentiment_adjustment = avg_sent * 0.15  # Max 15% adjustment from sentiment alone

        # Volume spike amplifies the signal - breaking news effect
        if volume_ratio >= self.config.volume_spike_threshold:
            sentiment_adjustment *= 1.5
            logger.info(
                "volume_spike_detected",
                market=market.condition_id,
                volume_ratio=volume_ratio,
            )

        # Combine market price with sentiment adjustment
        estimated_fair_value = clamp(market_price + sentiment_adjustment, 0.02, 0.98)

        # Calculate edge
        edge = estimated_fair_value - market_price

        # Determine signal strength and direction
        if abs(edge) < 0.03:  # Less than 3% edge, skip
            return None

        # Confidence based on:
        # 1. Strength of sentiment (how one-sided)
        # 2. Volume (more tweets = more reliable)
        # 3. Consistency (low std = more consistent signal)
        sent_strength = min(1.0, abs(avg_sent) / 0.5)
        volume_conf = min(1.0, sentiment_data.tweet_count / 50)
        consistency = max(0.0, 1.0 - sentiment_data.sentiment_std)
        confidence = (sent_strength * 0.4 + volume_conf * 0.3 + consistency * 0.3)

        if confidence < 0.3:
            return None

        # Direction
        if edge > 0:
            # Market underpriced vs sentiment -> BUY YES
            side = Side.BUY
            token = next((t for t in market.tokens if t.outcome == "Yes"), None)
        else:
            # Market overpriced vs sentiment -> BUY NO
            side = Side.BUY
            token = next((t for t in market.tokens if t.outcome == "No"), None)
            estimated_fair_value = 1.0 - estimated_fair_value
            market_price = 1.0 - market_price
            edge = abs(edge)

        if token is None:
            return None

        return Signal(
            market_condition_id=market.condition_id,
            token_id=token.token_id,
            side=side,
            outcome=token.outcome,
            estimated_fair_value=estimated_fair_value,
            market_price=market_price,
            edge=edge,
            confidence=confidence,
            strategy=self.name,
            metadata={
                "avg_sentiment": avg_sent,
                "tweet_count": sentiment_data.tweet_count,
                "volume_ratio": volume_ratio,
                "bullish_pct": sentiment_data.bullish_pct,
            },
        )
