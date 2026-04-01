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

# Weighted lexicon: (word -> sentiment weight)
# Higher absolute weight = stronger signal. This gives more nuance
# than a binary positive/negative classification.
WEIGHTED_LEXICON: dict[str, float] = {
    # Strong positive (1.5-2.0)
    "landslide": 2.0, "crushing": 1.8, "dominating": 1.8, "lock": 1.7,
    "certain": 1.6, "confirmed": 1.5, "absolutely": 1.5,
    # Medium positive (1.0-1.4)
    "bullish": 1.3, "moon": 1.2, "surge": 1.2, "rally": 1.2, "breakout": 1.2,
    "victory": 1.1, "winning": 1.1, "soar": 1.1, "boom": 1.1,
    "excellent": 1.0, "amazing": 1.0, "great": 1.0,
    # Mild positive (0.5-0.9)
    "win": 0.9, "pump": 0.9, "strong": 0.8, "confident": 0.8, "positive": 0.8,
    "favored": 0.8, "likely": 0.7, "probable": 0.7, "ahead": 0.7,
    "leading": 0.7, "rise": 0.6, "rising": 0.6, "up": 0.5, "yes": 0.5,
    "agreed": 0.5, "spike": 0.6, "definitely": 0.8,
    # Strong negative (-1.5 to -2.0)
    "impossible": -2.0, "collapse": -1.8, "crash": -1.7, "panic": -1.6,
    "plunge": -1.5, "disaster": -1.5,
    # Medium negative (-1.0 to -1.4)
    "bearish": -1.3, "dump": -1.2, "selloff": -1.2, "defeat": -1.1,
    "losing": -1.1, "terrible": -1.0, "awful": -1.0, "failed": -1.0,
    # Mild negative (-0.5 to -0.9)
    "lose": -0.9, "weak": -0.8, "fear": -0.8, "worried": -0.7,
    "concerned": -0.7, "uncertain": -0.7, "unlikely": -0.7, "doubt": -0.7,
    "drop": -0.6, "fall": -0.6, "falling": -0.6, "down": -0.5,
    "behind": -0.6, "trailing": -0.6, "underdog": -0.5, "risky": -0.5,
    "denied": -0.6, "negative": -0.6, "bust": -0.8, "breakdown": -0.7,
    "sink": -0.6, "never": -0.6,
}

# Bigram patterns with sentiment scores (captures phrases unigrams miss)
BIGRAM_SCORES: dict[tuple[str, str], float] = {
    ("looking", "good"): 1.0, ("no", "chance"): -1.5, ("no", "way"): -1.3,
    ("for", "sure"): 1.2, ("slam", "dunk"): 1.5, ("long", "shot"): -1.0,
    ("easy", "win"): 1.3, ("big", "win"): 1.2, ("huge", "loss"): -1.3,
    ("dead", "heat"): 0.0, ("too", "close"): 0.0, ("game", "over"): -1.2,
    ("all", "in"): 1.0, ("going", "down"): -0.8, ("going", "up"): 0.8,
    ("blown", "out"): -1.4, ("pulled", "ahead"): 1.0,
    ("falling", "apart"): -1.3, ("coming", "back"): 0.8,
    ("not", "happening"): -1.2, ("guaranteed", "win"): 1.5,
    ("massive", "lead"): 1.4, ("close", "race"): 0.0,
    ("red", "flag"): -0.8, ("green", "light"): 0.8,
}

# Intensifiers amplify the next sentiment word
INTENSIFIERS: dict[str, float] = {
    "very": 1.4, "extremely": 1.6, "incredibly": 1.5, "absolutely": 1.5,
    "totally": 1.3, "completely": 1.4, "really": 1.3, "super": 1.3,
    "hugely": 1.4, "massively": 1.5, "insanely": 1.5,
}

NEGATION_WORDS = frozenset([
    "not", "no", "never", "don't", "doesn't", "won't", "isn't", "aren't",
    "wasn't", "weren't", "can't", "cannot", "hardly", "barely", "neither",
])

# Emoji sentiment (many prediction market tweets use these)
EMOJI_SCORES: dict[str, float] = {
    "🚀": 1.2, "🔥": 0.8, "💪": 0.8, "✅": 0.7, "🎯": 0.8,
    "📈": 1.0, "📉": -1.0, "💀": -0.8, "😱": -0.7, "🐻": -1.0,
    "🐂": 1.0, "⬆️": 0.5, "⬇️": -0.5, "❌": -0.7, "💰": 0.6,
    "🏆": 1.0, "🤡": -0.6, "👑": 0.8, "⚠️": -0.5,
}


def score_text(text: str) -> float:
    """Score sentiment of text from -1 (bearish) to +1 (bullish).

    Uses a weighted lexicon with:
    - Intensity-weighted word scores (not binary)
    - Bigram pattern matching for multi-word phrases
    - Negation handling with scope (2-word window)
    - Intensifier amplification
    - Emoji sentiment scoring
    """
    # Score emojis first (before lowercasing)
    emoji_score = sum(
        EMOJI_SCORES.get(char, 0.0)
        for char in text
        if char in EMOJI_SCORES
    )

    words = re.findall(r'\b\w+\b', text.lower())
    if not words and emoji_score == 0:
        return 0.0

    score = emoji_score
    total_weight = max(len(words), 1)

    # ── Bigram scoring ───────────────────────────────────────
    for i in range(len(words) - 1):
        bigram = (words[i], words[i + 1])
        if bigram in BIGRAM_SCORES:
            score += BIGRAM_SCORES[bigram]

    # ── Unigram scoring with negation and intensifiers ───────
    negate = False
    negation_scope = 0  # Negation affects next 2 words
    intensifier = 1.0

    for word in words:
        if word in NEGATION_WORDS:
            negate = True
            negation_scope = 2
            continue

        if word in INTENSIFIERS:
            intensifier = INTENSIFIERS[word]
            continue

        if word in WEIGHTED_LEXICON:
            word_score = WEIGHTED_LEXICON[word] * intensifier
            if negate:
                word_score *= -0.75  # Negation partially inverts
            score += word_score
            intensifier = 1.0

        # Decay negation scope
        if negation_scope > 0:
            negation_scope -= 1
            if negation_scope == 0:
                negate = False
        intensifier = 1.0  # Reset if not followed by sentiment word

    # Normalize: scale by word count to keep in [-1, 1]
    if total_weight > 0:
        score = score / (total_weight * 0.3)
        score = max(-1.0, min(1.0, score))

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
