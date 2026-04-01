"""Twitter/X client for sentiment analysis and news monitoring."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Optional

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from polymarket_bot.config import TwitterConfig
from polymarket_bot.data.models import SentimentData

logger = structlog.get_logger()

# Simple lexicon-based sentiment scoring (no heavy NLP dependency)
POSITIVE_WORDS = frozenset([
    "bullish", "moon", "pump", "surge", "rally", "win", "winning", "victory",
    "up", "rise", "rising", "soar", "spike", "boom", "breakout", "strong",
    "confident", "certain", "definitely", "absolutely", "crushing", "dominating",
    "landslide", "yes", "confirmed", "agreed", "positive", "great", "amazing",
    "excellent", "ahead", "leading", "favored", "likely", "probable", "lock",
])

NEGATIVE_WORDS = frozenset([
    "bearish", "dump", "crash", "plunge", "drop", "lose", "losing", "defeat",
    "down", "fall", "falling", "sink", "collapse", "bust", "breakdown", "weak",
    "uncertain", "unlikely", "no", "denied", "negative", "terrible", "awful",
    "behind", "trailing", "underdog", "doubt", "risky", "failed", "impossible",
    "never", "selloff", "panic", "fear", "worried", "concerned",
])

NEGATION_WORDS = frozenset(["not", "no", "never", "don't", "doesn't", "won't", "isn't", "aren't", "wasn't"])


def score_text(text: str) -> float:
    """Score sentiment of text from -1 (bearish) to +1 (bullish).

    Uses a lexicon approach with negation handling. This is intentionally
    simple and fast - good enough for trading signals without needing
    a large ML model.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0

    score = 0.0
    negate = False

    for word in words:
        if word in NEGATION_WORDS:
            negate = True
            continue

        if word in POSITIVE_WORDS:
            score += -1.0 if negate else 1.0
            negate = False
        elif word in NEGATIVE_WORDS:
            score += 1.0 if negate else -1.0
            negate = False
        else:
            negate = False

    # Normalize by word count to keep in [-1, 1] range
    max_possible = len(words) * 0.5
    if max_possible > 0:
        score = max(-1.0, min(1.0, score / max_possible * 2))

    return score


class TwitterClient:
    """Client for X/Twitter API v2 for sentiment and news monitoring.

    Fetches recent tweets about prediction market topics, scores sentiment,
    and detects volume spikes that may indicate breaking news.
    """

    SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"

    def __init__(self, config: TwitterConfig) -> None:
        self.config = config
        self._client = httpx.Client(
            timeout=30.0,
            headers={
                "Authorization": f"Bearer {config.bearer_token}",
                "Content-Type": "application/json",
            },
        )
        # Baseline tweet volumes for volume spike detection
        self._baselines: dict[str, float] = {}

    def close(self) -> None:
        self._client.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    def search_tweets(
        self,
        query: str,
        max_results: int = 100,
        lookback_minutes: int = 60,
    ) -> list[dict[str, Any]]:
        """Search recent tweets matching a query."""
        if not self.config.bearer_token:
            logger.warning("twitter_no_token", msg="No Twitter bearer token configured")
            return []

        since = datetime.utcnow() - timedelta(minutes=lookback_minutes)
        params = {
            "query": f"{query} -is:retweet lang:en",
            "max_results": min(max_results, self.config.max_tweets_per_query),
            "start_time": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tweet.fields": "created_at,public_metrics,author_id",
        }

        try:
            resp = self._client.get(self.SEARCH_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            tweets = data.get("data", [])
            logger.info("twitter_search", query=query, count=len(tweets))
            return tweets
        except httpx.HTTPError as e:
            logger.error("twitter_search_failed", query=query, error=str(e))
            return []

    def analyze_sentiment(
        self,
        query: str,
        lookback_minutes: int | None = None,
    ) -> SentimentData:
        """Fetch tweets and compute aggregate sentiment for a query."""
        lookback = lookback_minutes or self.config.sentiment_lookback_minutes
        tweets = self.search_tweets(query, lookback_minutes=lookback)

        if not tweets:
            return SentimentData(query=query)

        scores = []
        texts = []
        for tweet in tweets:
            text = tweet.get("text", "")
            texts.append(text)
            scores.append(score_text(text))

        avg_sentiment = sum(scores) / len(scores) if scores else 0.0
        std_sentiment = (
            (sum((s - avg_sentiment) ** 2 for s in scores) / len(scores)) ** 0.5
            if len(scores) > 1
            else 0.0
        )

        bullish = sum(1 for s in scores if s > 0.1) / len(scores) if scores else 0.5
        bearish = sum(1 for s in scores if s < -0.1) / len(scores) if scores else 0.5

        # Volume spike detection
        current_volume = float(len(tweets))
        baseline = self._baselines.get(query, current_volume)
        volume_ratio = current_volume / baseline if baseline > 0 else 1.0
        # Update baseline with exponential moving average
        self._baselines[query] = baseline * 0.9 + current_volume * 0.1

        return SentimentData(
            query=query,
            tweet_count=len(tweets),
            avg_sentiment=avg_sentiment,
            sentiment_std=std_sentiment,
            volume_ratio=volume_ratio,
            bullish_pct=bullish,
            bearish_pct=bearish,
            sample_tweets=texts[:5],
        )

    def get_market_sentiment(self, market_question: str, keywords: list[str] | None = None) -> SentimentData:
        """Get sentiment for a specific market.

        Builds a search query from the market question and optional keywords,
        fetches tweets, and returns aggregated sentiment data.
        """
        # Build query from question keywords
        if keywords:
            query = " OR ".join(keywords[:5])
        else:
            # Extract key terms from the question
            stop_words = {"will", "the", "a", "an", "in", "on", "at", "to", "of", "by", "be", "is", "it", "or", "and", "for", "this", "that", "with", "from", "as", "are", "was", "were"}
            words = re.findall(r'\b\w+\b', market_question.lower())
            key_terms = [w for w in words if w not in stop_words and len(w) > 2][:5]
            query = " OR ".join(key_terms) if key_terms else market_question[:50]

        return self.analyze_sentiment(query)


class MockTwitterClient(TwitterClient):
    """Mock Twitter client for testing without API access."""

    def __init__(self, config: TwitterConfig | None = None) -> None:
        self.config = config or TwitterConfig()
        self._baselines: dict[str, float] = {}
        self._mock_tweets: dict[str, list[dict[str, Any]]] = {}

    def close(self) -> None:
        pass

    def set_mock_tweets(self, query: str, tweets: list[dict[str, Any]]) -> None:
        """Set mock tweet data for a query."""
        self._mock_tweets[query] = tweets

    def search_tweets(
        self,
        query: str,
        max_results: int = 100,
        lookback_minutes: int = 60,
    ) -> list[dict[str, Any]]:
        """Return mock tweets."""
        # Check for exact match first, then partial match
        if query in self._mock_tweets:
            return self._mock_tweets[query][:max_results]
        for key, tweets in self._mock_tweets.items():
            if any(word in query.lower() for word in key.lower().split()):
                return tweets[:max_results]
        return []
