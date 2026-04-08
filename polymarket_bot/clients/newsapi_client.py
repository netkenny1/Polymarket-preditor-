"""NewsAPI client for topic-driven narrative event ingestion.

Fetches recent news articles across a curated set of geopolitical, macro, and
crypto topics, converts them to NarrativeEvents, and caches results to avoid
hammering the free-tier rate limit. Degrades gracefully to an empty list when
no API key is configured.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp
import structlog

from polymarket_bot.clients.twitter import score_text
from polymarket_bot.data.models import NarrativeCategory, NarrativeEvent

logger = structlog.get_logger()

NEWS_BASE = "https://newsapi.org/v2/everything"

# Ordered list of (query_string, NarrativeCategory) pairs.
# The order also determines concurrency scheduling priority.
TOPIC_QUERIES: list[tuple[str, NarrativeCategory]] = [
    ("Trump tariff executive order trade",                        NarrativeCategory.TRADE_WAR),
    ("Federal Reserve interest rate inflation CPI",               NarrativeCategory.MONETARY_POLICY),
    ("Ukraine Russia war NATO military",                          NarrativeCategory.GEOPOLITICAL),
    ("Israel Hamas Gaza Middle East conflict",                    NarrativeCategory.GEOPOLITICAL),
    ("China Taiwan semiconductor",                                NarrativeCategory.GEOPOLITICAL),
    ("Iran nuclear sanctions",                                    NarrativeCategory.GEOPOLITICAL),
    ("bitcoin ethereum SEC crypto regulation stablecoin",         NarrativeCategory.CRYPTO_REGULATION),
    ("Trump sanctions foreign policy executive",                  NarrativeCategory.TRADE_WAR),
    ("recession GDP unemployment economic",                       NarrativeCategory.FISCAL_POLICY),
    ("stock market crash rally S&P correction",                   NarrativeCategory.MARKET_CRISIS),
]

# Common English stopwords to skip when extracting keywords from article text
_STOPWORDS = frozenset({
    "the", "and", "that", "this", "with", "from", "have", "will", "been",
    "they", "their", "there", "what", "when", "which", "where", "about",
    "would", "could", "should", "after", "before", "more", "some", "into",
    "over", "also", "said", "says", "were", "than", "then", "just", "each",
    "other", "your", "such", "only", "very", "does", "like", "being",
    "while", "through", "those",
})


def _extract_keywords(text: str, max_keywords: int = 10) -> list[str]:
    """Return up to *max_keywords* meaningful words from *text*.

    Splits on whitespace / punctuation, keeps words longer than 4 characters,
    drops stopwords, and deduplicates while preserving first-seen order.
    """
    import re

    tokens = re.findall(r"[A-Za-z]+", text.lower())
    seen: set[str] = set()
    keywords: list[str] = []
    for tok in tokens:
        if len(tok) > 4 and tok not in _STOPWORDS and tok not in seen:
            seen.add(tok)
            keywords.append(tok)
            if len(keywords) >= max_keywords:
                break
    return keywords


def _article_to_event(article: dict[str, Any], category: NarrativeCategory) -> NarrativeEvent | None:
    """Convert a raw NewsAPI article dict into a NarrativeEvent.

    Returns ``None`` when required fields are missing or the article is empty.
    """
    title = article.get("title") or ""
    description = article.get("description") or ""

    if not title:
        return None

    content = f"{title}. {description}".strip()
    sentiment = score_text(content)
    magnitude = max(0.1, min(1.0, abs(sentiment)))

    # Parse publishedAt timestamp; fall back to now
    published_raw = article.get("publishedAt") or ""
    try:
        timestamp = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        timestamp = datetime.now(timezone.utc)

    source_name = (article.get("source") or {}).get("name", "newsapi")
    article_url = article.get("url", "")

    return NarrativeEvent(
        event_id=str(uuid.uuid4()),
        source="newsapi",
        content=content,
        timestamp=timestamp,
        category=category,
        sentiment=sentiment,
        magnitude=magnitude,
        keywords=_extract_keywords(content),
        metadata={
            "source_name": source_name,
            "url": article_url,
            "title": title,
        },
    )


class NewsAPIClient:
    """Client for the NewsAPI /v2/everything endpoint.

    Fetches recent articles for each topic in ``TOPIC_QUERIES``, converts
    them to ``NarrativeEvent`` objects, and caches the combined result for
    ``cache_ttl_minutes`` minutes to stay within free-tier rate limits.

    Usage::

        client = NewsAPIClient(api_key=os.environ["NEWSAPI_KEY"])
        events = await client.fetch_all_topics(lookback_hours=2)
    """

    def __init__(self, api_key: str, cache_ttl_minutes: int = 10) -> None:
        self._api_key = api_key
        self._cache_ttl_seconds = cache_ttl_minutes * 60
        # Simple time-based cache: stores (timestamp, events)
        self._cache: tuple[float, list[NarrativeEvent]] | None = None

    # ------------------------------------------------------------------
    # Low-level fetch
    # ------------------------------------------------------------------

    async def fetch_topic(
        self,
        session: aiohttp.ClientSession,
        query: str,
        category: NarrativeCategory,
        lookback_hours: int = 2,
    ) -> list[NarrativeEvent]:
        """Fetch up to 10 recent articles for *query* and return NarrativeEvents.

        Parameters
        ----------
        session:
            An active ``aiohttp.ClientSession``.
        query:
            Free-text search query forwarded to NewsAPI.
        category:
            The ``NarrativeCategory`` to assign to all resulting events.
        lookback_hours:
            How far back to search (ISO 8601 ``from`` parameter).

        Returns
        -------
        list[NarrativeEvent]
            Parsed events. Returns ``[]`` on any error.
        """
        if not self._api_key:
            return []

        from_dt = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).isoformat()

        params = {
            "q": query,
            "from": from_dt,
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": self._api_key,
            "pageSize": 10,
        }

        try:
            async with session.get(
                NEWS_BASE,
                params=params,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                if resp.status == 401:
                    logger.warning("newsapi_unauthorized", msg="Check your NewsAPI key")
                    return []
                if resp.status == 429:
                    logger.warning("newsapi_rate_limited", query=query)
                    return []
                if resp.status != 200:
                    logger.warning("newsapi_http_error", query=query, status=resp.status)
                    return []

                data = await resp.json(content_type=None)
                articles: list[dict[str, Any]] = data.get("articles", [])

        except aiohttp.ClientError as exc:
            logger.warning("newsapi_request_error", query=query, error=str(exc))
            return []
        except Exception as exc:  # noqa: BLE001
            logger.warning("newsapi_unexpected_error", query=query, error=str(exc))
            return []

        events: list[NarrativeEvent] = []
        for article in articles:
            event = _article_to_event(article, category)
            if event is not None:
                events.append(event)

        logger.info(
            "newsapi_topic_fetched",
            query=query[:50],
            articles=len(articles),
            events=len(events),
        )
        return events

    # ------------------------------------------------------------------
    # High-level aggregation
    # ------------------------------------------------------------------

    async def fetch_all_topics(self, lookback_hours: int = 2) -> list[NarrativeEvent]:
        """Fetch all ``TOPIC_QUERIES`` concurrently and return combined events.

        Results are cached for ``cache_ttl_minutes`` minutes (set at init).
        Calls within the TTL window return the cached list instantly, preventing
        unnecessary API quota consumption.

        Returns an empty list immediately if no API key is configured.
        """
        if not self._api_key:
            logger.debug(
                "newsapi_no_api_key",
                msg="Set NEWSAPI_KEY for live news data; returning empty list",
            )
            return []

        now = time.monotonic()
        if self._cache is not None:
            cached_at, cached_events = self._cache
            if now - cached_at < self._cache_ttl_seconds:
                logger.debug(
                    "newsapi_cache_hit",
                    age_seconds=round(now - cached_at, 1),
                    events=len(cached_events),
                )
                return cached_events

        async with aiohttp.ClientSession() as session:
            tasks = [
                asyncio.create_task(
                    self.fetch_topic(session, query, category, lookback_hours)
                )
                for query, category in TOPIC_QUERIES
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        all_events: list[NarrativeEvent] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                query, _ = TOPIC_QUERIES[i]
                logger.warning(
                    "newsapi_topic_error",
                    query=query[:50],
                    error=str(result),
                )
                continue
            all_events.extend(result)  # type: ignore[arg-type]

        # Sort newest-first
        all_events.sort(key=lambda e: e.timestamp, reverse=True)

        self._cache = (now, all_events)
        logger.info("newsapi_all_topics_fetched", total_events=len(all_events))
        return all_events
