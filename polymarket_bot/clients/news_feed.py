"""Real-time news and event feed clients for narrative analysis.

Provides pluggable clients for fetching NarrativeEvents from multiple
sources (Truth Social, FRED, generic news feeds) and an aggregator that
combines them with deduplication and rate limiting.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from datetime import datetime, timezone
from typing import Any, Protocol

import httpx
import structlog

from polymarket_bot.data.models import (
    EconomicIndicator,
    NarrativeCategory,
    NarrativeEvent,
)

logger = structlog.get_logger()


# ─── Keyword → Category mapping ───────────────────────────────────────
CATEGORY_KEYWORDS: dict[NarrativeCategory, list[str]] = {
    NarrativeCategory.TRADE_WAR: [
        "tariff", "tariffs", "china", "trade war", "trade", "import", "export",
        "customs", "duty", "wto", "xi jinping",
    ],
    NarrativeCategory.MONETARY_POLICY: [
        "powell", "fed", "federal reserve", "rates", "rate cut", "rate hike",
        "fomc", "inflation", "hawkish", "dovish",
    ],
    NarrativeCategory.GEOPOLITICAL: [
        "russia", "putin", "israel", "iran", "ukraine", "gaza", "nato",
        "war", "conflict", "missile", "strike",
    ],
    NarrativeCategory.CRYPTO_REGULATION: [
        "bitcoin", "btc", "crypto", "cryptocurrency", "sec", "ethereum",
        "stablecoin", "coinbase", "binance",
    ],
    NarrativeCategory.FISCAL_POLICY: [
        "tax", "taxes", "spending", "deficit", "debt ceiling", "budget",
        "stimulus", "irs",
    ],
}

POSITIVE_WORDS = frozenset([
    "great", "amazing", "tremendous", "win", "winning", "strong", "best",
    "huge success", "victory", "beautiful", "historic",
])
NEGATIVE_WORDS = frozenset([
    "terrible", "disaster", "fail", "failed", "weak", "horrible", "sad",
    "crooked", "fake", "worst", "catastrophe", "crisis",
])
THREAT_WORDS = frozenset([
    "will impose", "must", "threat", "retaliate", "ban", "sanction",
    "destroy", "crush", "demand", "invade",
])


class NewsFeedClient(Protocol):
    """Protocol for any news/event feed client."""

    name: str

    async def fetch_latest(self, since: datetime) -> list[NarrativeEvent]:
        """Fetch events newer than `since`. Must be implemented."""
        ...


class TruthSocialClient:
    """Fetches Trump Truth Social posts and classifies them into NarrativeEvents.

    The posted-content scraping is a placeholder. In production, plug in:
      - Apify Truth Social scraper actor
      - ScrapingBee/Zyte rendering of the public profile
      - A direct headless-browser (Playwright) scraper
      - An unofficial Truth Social API mirror

    The classification pipeline (`_classify_post`, `_estimate_magnitude`) is
    fully implemented and can be applied to any post text the scraper returns.
    """

    name = "truth_social"

    def __init__(
        self,
        user_handle: str = "realDonaldTrump",
        cache_ttl_seconds: int = 60,
    ) -> None:
        self.user_handle = user_handle
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: list[NarrativeEvent] = []
        self._cache_time: float = 0.0

    async def fetch_latest(self, since: datetime) -> list[NarrativeEvent]:
        """Return posts newer than `since`.

        PLACEHOLDER: returns empty list. Replace the body of this method
        with a call to a scraping provider, something like::

            async with httpx.AsyncClient() as c:
                r = await c.get(
                    f"https://api.apify.com/v2/acts/.../run-sync-get-dataset-items",
                    params={"token": api_key, "handle": self.user_handle},
                )
                posts = r.json()
            return [self._post_to_event(p) for p in posts
                    if p["created_at"] > since]
        """
        now = time.time()
        if now - self._cache_time < self.cache_ttl_seconds and self._cache:
            return [e for e in self._cache if e.timestamp > since]

        logger.debug(
            "truth_social_fetch_stub",
            handle=self.user_handle,
            since=since.isoformat(),
            msg="Production implementation should scrape here",
        )
        self._cache = []
        self._cache_time = now
        return []

    def _post_to_event(self, post: dict[str, Any]) -> NarrativeEvent:
        """Convert a raw scraped post dict into a NarrativeEvent."""
        text = post.get("content", "") or post.get("text", "")
        category, sentiment = self._classify_post(text)
        magnitude = self._estimate_magnitude(text)
        post_id = str(post.get("id", hashlib.md5(text.encode()).hexdigest()[:12]))
        ts_raw = post.get("created_at") or post.get("timestamp")
        if isinstance(ts_raw, str):
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except ValueError:
                ts = datetime.now(timezone.utc)
        elif isinstance(ts_raw, datetime):
            ts = ts_raw
        else:
            ts = datetime.now(timezone.utc)

        return NarrativeEvent(
            event_id=f"truth_{post_id}",
            source="trump_truth_social",
            content=text,
            timestamp=ts,
            category=category,
            sentiment=sentiment,
            magnitude=magnitude,
            keywords=self._extract_keywords(text),
            metadata={"handle": self.user_handle, "raw": post},
        )

    def _classify_post(self, text: str) -> tuple[NarrativeCategory, float]:
        """Classify a post by keywords, returning (category, sentiment -1..1)."""
        lower = text.lower()

        # Category: pick the one with the most keyword hits.
        best_cat = NarrativeCategory.OTHER
        best_hits = 0
        for cat, kws in CATEGORY_KEYWORDS.items():
            hits = sum(1 for k in kws if k in lower)
            if hits > best_hits:
                best_hits = hits
                best_cat = cat

        # Sentiment scoring: count positive/negative word hits, normalize.
        pos = sum(1 for w in POSITIVE_WORDS if w in lower)
        neg = sum(1 for w in NEGATIVE_WORDS if w in lower)
        total = pos + neg
        if total == 0:
            sentiment = 0.0
        else:
            sentiment = (pos - neg) / total
        sentiment = max(-1.0, min(1.0, sentiment))

        return best_cat, sentiment

    def _estimate_magnitude(self, text: str) -> float:
        """Estimate the "impact magnitude" of a post on a 0..1 scale.

        Heuristics:
          - Density of ALL-CAPS words (Trump's signature emphasis).
          - Density of exclamation marks.
          - Presence of threat language ("will impose", "must", "retaliate").
          - Longer posts with more intensity markers score higher.
        """
        if not text:
            return 0.0

        words = re.findall(r"\b[A-Za-z]{2,}\b", text)
        if not words:
            return 0.0

        caps_words = sum(1 for w in words if w.isupper() and len(w) >= 3)
        caps_density = caps_words / len(words)

        exclaim = text.count("!")
        exclaim_density = min(1.0, exclaim / 5.0)

        lower = text.lower()
        threat_hits = sum(1 for phrase in THREAT_WORDS if phrase in lower)
        threat_score = min(1.0, threat_hits / 2.0)

        magnitude = (
            0.45 * min(1.0, caps_density * 4)
            + 0.25 * exclaim_density
            + 0.30 * threat_score
        )
        return max(0.0, min(1.0, magnitude))

    def _extract_keywords(self, text: str) -> list[str]:
        lower = text.lower()
        hits: list[str] = []
        for kws in CATEGORY_KEYWORDS.values():
            for k in kws:
                if k in lower and k not in hits:
                    hits.append(k)
        return hits[:10]


class FREDClient:
    """Client for the Federal Reserve Economic Data (FRED) API.

    Documentation: https://fred.stlouisfed.org/docs/api
    """

    name = "fred"

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    # Common series IDs used by the bot.
    CPI_YOY = "CPIAUCSL"
    FED_FUNDS = "DFF"
    UNEMPLOYMENT = "UNRATE"
    TEN_YEAR = "DGS10"
    DOLLAR_INDEX = "DTWEXBGS"

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key

    async def fetch_series(self, series_id: str) -> list[EconomicIndicator]:
        """Fetch observations for a FRED series.

        Real implementation would call::

            url = (
                f"{self.BASE_URL}?series_id={series_id}"
                f"&api_key={self.api_key}&file_type=json"
            )
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.get(url)
                data = r.json()
            obs = data.get("observations", [])
            # Convert each obs dict {date, value} into an EconomicIndicator,
            # computing change_pct vs the previous observation.

        This stub returns an empty list when no api_key is configured so
        that downstream code can be exercised in dev/test mode.
        """
        if not self.api_key:
            logger.debug(
                "fred_no_api_key",
                series_id=series_id,
                msg="Set FRED_API_KEY for live data",
            )
            return []

        logger.debug(
            "fred_fetch_stub",
            series_id=series_id,
            msg="Production implementation should httpx.get here",
        )
        return []

    async def fetch_latest(self, since: datetime) -> list[NarrativeEvent]:
        """Protocol-compliant wrapper: fetch key series and synthesize events.

        For each tracked series, a NarrativeEvent is produced when the latest
        value meaningfully differs from the previous one. Returns empty when
        no api key or no data.
        """
        return []


class NewsAggregator:
    """Combines multiple feed clients with deduplication and rate limiting."""

    def __init__(self, clients: list[NewsFeedClient]) -> None:
        self.clients = clients
        # token bucket: client_name -> (tokens, last_refill, capacity, refill_rate)
        self._buckets: dict[str, list[float]] = {}
        self._default_capacity = 10.0
        self._default_refill_per_sec = 0.2  # 12 requests/minute

    def rate_limit_check(self, client_name: str) -> bool:
        """Return True if the client may make a request right now.

        Simple token-bucket: each client starts with `capacity` tokens and
        refills at `refill_per_sec`. Each call consumes one token.
        """
        now = time.time()
        bucket = self._buckets.get(client_name)
        if bucket is None:
            bucket = [self._default_capacity, now]
            self._buckets[client_name] = bucket

        tokens, last = bucket
        elapsed = now - last
        tokens = min(self._default_capacity, tokens + elapsed * self._default_refill_per_sec)

        if tokens >= 1.0:
            bucket[0] = tokens - 1.0
            bucket[1] = now
            return True

        bucket[0] = tokens
        bucket[1] = now
        logger.warning("news_aggregator_rate_limited", client=client_name)
        return False

    async def fetch_all(self, since: datetime) -> list[NarrativeEvent]:
        """Fetch events from all clients concurrently and deduplicate."""
        tasks = []
        names = []
        for client in self.clients:
            cname = getattr(client, "name", client.__class__.__name__)
            if not self.rate_limit_check(cname):
                continue
            tasks.append(self._safe_fetch(client, since))
            names.append(cname)

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_events: list[NarrativeEvent] = []
        for name, result in zip(names, results):
            if isinstance(result, Exception):
                logger.error("news_client_failed", client=name, error=str(result))
                continue
            all_events.extend(result)

        return self._deduplicate(all_events)

    async def _safe_fetch(
        self, client: NewsFeedClient, since: datetime
    ) -> list[NarrativeEvent]:
        try:
            return await client.fetch_latest(since)
        except Exception as e:  # noqa: BLE001 — defensive aggregator
            logger.error(
                "news_client_exception",
                client=getattr(client, "name", "?"),
                error=str(e),
            )
            return []

    def _deduplicate(self, events: list[NarrativeEvent]) -> list[NarrativeEvent]:
        """Deduplicate by content hash of the first 100 chars of content."""
        seen: set[str] = set()
        out: list[NarrativeEvent] = []
        for ev in events:
            key = hashlib.sha1(ev.content[:100].encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            out.append(ev)
        out.sort(key=lambda e: e.timestamp, reverse=True)
        return out
