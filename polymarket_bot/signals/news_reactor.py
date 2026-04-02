"""News and tweet reactor — maps breaking events to Polymarket trades.

When Trump tweets about tariffs, or crypto news breaks, there's a window
where Polymarket prices haven't fully adjusted. This module detects those
events, maps them to relevant markets, and generates trading signals.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from polymarket_bot.data.models import Market, MarketCategory, Side, Signal
from polymarket_bot.utils.helpers import clamp

logger = structlog.get_logger()

# Breaking news indicators
_BREAKING_PATTERNS = re.compile(
    r"BREAKING|JUST\s+IN|ALERT|URGENT|DEVELOPING|EXCLUSIVE|🚨",
    re.IGNORECASE,
)

# Trump-specific policy patterns
_TRUMP_POLICY = {
    "tariff": ("ECONOMICS", -0.3),      # Tariffs usually negative for markets
    "executive order": ("POLITICS", 0.0),
    "ban": ("POLITICS", -0.2),
    "sanctions": ("FOREIGN_POLICY", -0.2),
    "deal": ("ECONOMICS", 0.3),
    "agreement": ("FOREIGN_POLICY", 0.2),
    "bitcoin": ("CRYPTO", 0.3),          # Trump pro-crypto signals
    "crypto": ("CRYPTO", 0.3),
    "fed": ("ECONOMICS", 0.0),
    "rate": ("ECONOMICS", 0.0),
    "china": ("FOREIGN_POLICY", -0.1),
    "war": ("FOREIGN_POLICY", -0.3),
    "peace": ("FOREIGN_POLICY", 0.3),
    "nomination": ("POLITICS", 0.0),
    "fire": ("POLITICS", -0.1),
    "resign": ("POLITICS", -0.2),
}

# Category impact multipliers
_CATEGORY_IMPACT = {
    MarketCategory.POLITICS: 1.3,
    MarketCategory.CRYPTO: 1.0,
    MarketCategory.SPORTS: 0.3,
    MarketCategory.POP_CULTURE: 0.5,
    MarketCategory.SCIENCE: 0.6,
    MarketCategory.OTHER: 0.8,
}


@dataclass
class NewsEvent:
    """A detected news event from Twitter or other sources."""

    source: str
    content: str
    sentiment: float  # -1 to 1
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    volume: int = 1
    keywords: list[str] = field(default_factory=list)
    breaking: bool = False
    relevance_category: MarketCategory | None = None

    @property
    def age_minutes(self) -> float:
        delta = datetime.now(timezone.utc) - self.timestamp
        return delta.total_seconds() / 60.0

    @property
    def is_fresh(self) -> bool:
        return self.age_minutes < 60  # Fresh within 1 hour


class TrumpTweetAnalyzer:
    """Analyze Trump tweets/posts for market impact.

    Handles Trump's communication patterns:
    - ALL CAPS = more serious/emphatic
    - "FAKE NEWS" / "HOAX" = dismiss/contrarian signal
    - Policy keywords = map to specific market categories
    - Exclamation marks = emphasis multiplier
    """

    def analyze(self, text: str) -> tuple[str, float, MarketCategory | None]:
        """Analyze a Trump tweet for impact.

        Returns:
            (impact_type, direction, category)
            - impact_type: "tariff", "crypto", "foreign_policy", etc.
            - direction: -1 to 1 (negative = bearish, positive = bullish)
            - category: relevant MarketCategory or None
        """
        text_lower = text.lower()

        # Detect emphasis (ALL CAPS ratio, exclamation marks)
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        exclamation_count = text.count("!")
        emphasis = 1.0 + min(caps_ratio * 0.5, 0.3) + min(exclamation_count * 0.05, 0.2)

        # Dismissive patterns reduce impact
        if any(p in text_lower for p in ["fake news", "hoax", "witch hunt", "nothing burger"]):
            emphasis *= 0.5

        # Check policy keywords
        best_match = None
        best_direction = 0.0
        best_category = None

        for keyword, (impact_type, base_dir) in _TRUMP_POLICY.items():
            if keyword in text_lower:
                category_map = {
                    "ECONOMICS": MarketCategory.OTHER,
                    "POLITICS": MarketCategory.POLITICS,
                    "FOREIGN_POLICY": MarketCategory.POLITICS,
                    "CRYPTO": MarketCategory.CRYPTO,
                }
                best_match = impact_type
                best_direction = base_dir * emphasis
                best_category = category_map.get(impact_type)

        if best_match is None:
            # Generic Trump tweet — small political impact
            best_match = "general"
            best_direction = 0.0
            best_category = MarketCategory.POLITICS

        return best_match, clamp(best_direction, -1.0, 1.0), best_category


class NewsReactor:
    """Maps breaking news/tweets to actionable Polymarket trades."""

    def __init__(self, twitter_client: Any = None) -> None:
        self._twitter = twitter_client
        self._trump_analyzer = TrumpTweetAnalyzer()
        self._event_history: list[NewsEvent] = []
        self._market_keyword_cache: dict[str, list[str]] = {}
        self._seen_content: set[str] = set()  # Dedup

    def build_keyword_map(self, markets: list[Market]) -> dict[str, list[str]]:
        """Extract keywords from market questions for matching against news."""
        stop_words = {
            "will", "the", "be", "is", "on", "in", "at", "to", "by", "a",
            "an", "of", "for", "and", "or", "this", "that", "it", "with",
            "from", "has", "have", "was", "were", "been", "are",
        }

        result: dict[str, list[str]] = {}
        for m in markets:
            words = re.findall(r"[a-zA-Z$]+[a-zA-Z0-9]*", m.question.lower())
            keywords = [w for w in words if w not in stop_words and len(w) > 2]
            # Add special patterns
            if "$" in m.question:
                prices = re.findall(r"\$[\d,]+k?", m.question)
                keywords.extend(p.lower() for p in prices)
            result[m.condition_id] = keywords
            self._market_keyword_cache[m.condition_id] = keywords

        return result

    def detect_events(self, tweets: list[dict[str, Any]]) -> list[NewsEvent]:
        """Detect significant events from a tweet stream."""
        events: list[NewsEvent] = []

        # Group by topic (simple keyword clustering)
        topic_tweets: dict[str, list[dict]] = defaultdict(list)
        for tweet in tweets:
            text = tweet.get("text", "")
            # Skip duplicates
            content_key = text[:100]
            if content_key in self._seen_content:
                continue
            self._seen_content.add(content_key)

            # Extract main topic words
            words = set(re.findall(r"[a-zA-Z]{4,}", text.lower()))
            for w in words:
                topic_tweets[w].append(tweet)

        # Find volume spikes (many tweets about same topic)
        for topic, related in topic_tweets.items():
            if len(related) < 3:
                continue

            # Aggregate sentiment
            from polymarket_bot.clients.twitter import score_text
            sentiments = [score_text(t.get("text", "")) for t in related]
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

            # Check for breaking news
            is_breaking = any(
                _BREAKING_PATTERNS.search(t.get("text", ""))
                for t in related
            )

            # Extract keywords from all related tweets
            all_text = " ".join(t.get("text", "") for t in related)
            keywords = list(set(re.findall(r"[a-zA-Z]{4,}", all_text.lower())))[:20]

            event = NewsEvent(
                source="twitter",
                content=related[0].get("text", ""),
                sentiment=avg_sentiment,
                volume=len(related),
                keywords=keywords,
                breaking=is_breaking,
            )
            events.append(event)

        # Check for Trump-specific events
        for tweet in tweets:
            text = tweet.get("text", "")
            author = tweet.get("author", "").lower()
            if "trump" in author or "realdonaldtrump" in author or "potus" in author:
                impact_type, direction, category = self._trump_analyzer.analyze(text)
                event = NewsEvent(
                    source="trump_tweet",
                    content=text,
                    sentiment=direction,
                    volume=1,
                    keywords=[impact_type, "trump"],
                    breaking=True,
                    relevance_category=category,
                )
                events.append(event)

        self._event_history.extend(events)
        # Keep history bounded
        if len(self._event_history) > 500:
            self._event_history = self._event_history[-250:]

        logger.info("events_detected", count=len(events))
        return events

    def map_event_to_markets(
        self,
        event: NewsEvent,
        markets: list[Market],
    ) -> list[tuple[Market, float]]:
        """Map a detected event to relevant markets with relevance scores."""
        results: list[tuple[Market, float]] = []

        for m in markets:
            keywords = self._market_keyword_cache.get(m.condition_id, [])
            if not keywords:
                keywords = re.findall(r"[a-zA-Z]{4,}", m.question.lower())

            # Compute keyword overlap
            event_kw_set = set(event.keywords)
            market_kw_set = set(keywords)
            overlap = event_kw_set & market_kw_set

            if not overlap:
                continue

            # Relevance = overlap ratio * category alignment
            overlap_score = len(overlap) / max(len(market_kw_set), 1)

            # Boost if event category matches market category
            cat_boost = 1.0
            if event.relevance_category and event.relevance_category == m.category:
                cat_boost = 1.5

            # Boost for breaking news
            breaking_boost = 1.3 if event.breaking else 1.0

            relevance = overlap_score * cat_boost * breaking_boost
            if relevance > 0.1:
                results.append((m, relevance))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:10]  # Top 10 matches

    def estimate_impact(
        self,
        event: NewsEvent,
        market: Market,
    ) -> tuple[float, float]:
        """Estimate direction and magnitude of impact on a market.

        Returns (direction, magnitude) where:
        - direction: positive = bullish for YES, negative = bearish
        - magnitude: 0.0-1.0 estimated price impact
        """
        # Base direction from event sentiment
        direction = event.sentiment

        # Magnitude based on volume, breaking status, freshness
        base_magnitude = 0.03
        if event.breaking:
            base_magnitude *= 2.0
        if event.volume > 10:
            base_magnitude *= 1.5
        elif event.volume > 5:
            base_magnitude *= 1.2

        # Time decay — impact fades over hours
        age_hours = event.age_minutes / 60.0
        decay = max(0.1, 1.0 - age_hours * 0.2)

        # Category-specific multiplier
        cat_mult = _CATEGORY_IMPACT.get(market.category, 0.8)

        magnitude = clamp(base_magnitude * decay * cat_mult, 0.0, 0.15)

        return direction, magnitude

    def generate_signals(
        self,
        events: list[NewsEvent],
        markets: list[Market],
    ) -> list[Signal]:
        """Convert detected events into trading signals."""
        signals: list[Signal] = []

        for event in events:
            if not event.is_fresh:
                continue

            mapped = self.map_event_to_markets(event, markets)

            for market, relevance in mapped:
                direction, magnitude = self.estimate_impact(event, market)

                if abs(magnitude) < 0.02:
                    continue

                yes_token = next((t for t in market.tokens if t.outcome == "Yes"), None)
                no_token = next((t for t in market.tokens if t.outcome == "No"), None)
                if not yes_token or not no_token:
                    continue

                # Determine which side to trade
                if direction > 0:
                    # Bullish event → buy YES
                    fair_value = clamp(yes_token.price + magnitude, 0.05, 0.95)
                    edge = fair_value - yes_token.price
                    token = yes_token
                    market_price = yes_token.price
                else:
                    # Bearish event → buy NO
                    fair_value = clamp(no_token.price + magnitude, 0.05, 0.95)
                    edge = fair_value - no_token.price
                    token = no_token
                    market_price = no_token.price

                if edge < 0.03:
                    continue

                confidence = clamp(
                    0.35 + relevance * 0.2 + (0.1 if event.breaking else 0),
                    0.2, 0.75,
                )

                signals.append(Signal(
                    market_condition_id=market.condition_id,
                    token_id=token.token_id,
                    side=Side.BUY,
                    outcome=token.outcome,
                    estimated_fair_value=fair_value,
                    market_price=market_price,
                    edge=edge,
                    confidence=confidence,
                    strategy="news_reactor",
                    metadata={
                        "event_source": event.source,
                        "event_content": event.content[:100],
                        "event_sentiment": round(event.sentiment, 3),
                        "event_volume": event.volume,
                        "breaking": event.breaking,
                        "relevance": round(relevance, 3),
                        "impact_direction": round(direction, 3),
                        "impact_magnitude": round(magnitude, 4),
                    },
                ))

        logger.info(
            "news_signals_generated",
            events=len(events),
            signals=len(signals),
        )
        return signals
