"""Trump / White House monitoring agent.

Watches Trump statements, executive orders, and related news via
NewsAPI and the Twitter v2 API, then asks Claude to characterise
the market impact and produce ``NarrativeEvent`` objects.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp
import structlog

from polymarket_bot.data.models import NarrativeCategory, NarrativeEvent
from polymarket_bot.agents.base_agent import AgentInsight, BaseAgent

logger = structlog.get_logger()

# Map label strings returned by Claude to our enum values
_CATEGORY_MAP: dict[str, NarrativeCategory] = {
    "trade_war": NarrativeCategory.TRADE_WAR,
    "monetary_policy": NarrativeCategory.MONETARY_POLICY,
    "geopolitical": NarrativeCategory.GEOPOLITICAL,
    "crypto_regulation": NarrativeCategory.CRYPTO_REGULATION,
    "fiscal_policy": NarrativeCategory.FISCAL_POLICY,
    "election": NarrativeCategory.ELECTION,
    "market_crisis": NarrativeCategory.MARKET_CRISIS,
    "other": NarrativeCategory.OTHER,
}


class TrumpMonitorAgent(BaseAgent):
    """Monitor Trump statements and White House news for market signals.

    Data sources (all optional — falls back to empty list gracefully):
    * NewsAPI (requires ``newsapi_key`` in config)
    * Twitter v2 search on @realDonaldTrump (requires ``twitter_bearer`` in config)
    """

    name = "trump_monitor"

    def __init__(self, anthropic_client: Any, config: dict) -> None:
        super().__init__(anthropic_client, config)
        self._newsapi_key: str = config.get("newsapi_key", "")
        self._twitter_bearer: str = config.get("twitter_bearer", "")

    # ── Data fetching ────────────────────────────────────────────────

    async def fetch_data(self) -> list[dict]:
        """Fetch Trump-related news and tweets concurrently."""
        tasks: list[asyncio.Task] = []
        results: list[dict] = []

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=20)
        ) as session:
            if self._newsapi_key:
                tasks.append(
                    asyncio.create_task(self._fetch_newsapi(session))
                )
            if self._twitter_bearer:
                tasks.append(
                    asyncio.create_task(self._fetch_trump_tweets(session))
                )

            if not tasks:
                logger.debug(
                    "trump_monitor_no_keys",
                    msg="No API keys configured; returning empty data",
                )
                return []

            for coro in asyncio.as_completed(tasks):
                try:
                    items = await coro
                    results.extend(items)
                except Exception as exc:
                    logger.warning("trump_monitor_fetch_error", error=str(exc))

        return results

    async def _fetch_newsapi(self, session: aiohttp.ClientSession) -> list[dict]:
        """Fetch Trump-related articles from NewsAPI (last 1 hour)."""
        from_dt = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        url = (
            "https://newsapi.org/v2/everything"
            f"?q=Trump+tariff+executive+order+war+crypto"
            f"&language=en&sortBy=publishedAt&pageSize=15"
            f"&apiKey={self._newsapi_key}&from={from_dt}"
        )
        try:
            async with session.get(url) as resp:
                resp.raise_for_status()
                data = await resp.json()
            articles = data.get("articles", [])
            return [
                {
                    "text": (a.get("title", "") + " " + (a.get("description") or "")).strip(),
                    "source": "newsapi",
                    "url": a.get("url", ""),
                    "published_at": a.get("publishedAt", ""),
                }
                for a in articles
                if a.get("title")
            ]
        except aiohttp.ClientResponseError as exc:
            logger.warning("newsapi_fetch_failed", status=exc.status, error=str(exc))
            return []
        except Exception as exc:
            logger.warning("newsapi_fetch_error", error=str(exc))
            return []

    async def _fetch_trump_tweets(self, session: aiohttp.ClientSession) -> list[dict]:
        """Fetch recent @realDonaldTrump tweets from Twitter v2 API."""
        url = (
            "https://api.twitter.com/2/tweets/search/recent"
            "?query=from%3ArealDonaldTrump"
            "&max_results=10"
            "&tweet.fields=created_at,text"
        )
        headers = {"Authorization": f"Bearer {self._twitter_bearer}"}
        try:
            async with session.get(url, headers=headers) as resp:
                resp.raise_for_status()
                data = await resp.json()
            tweets = data.get("data", [])
            return [
                {
                    "text": t.get("text", ""),
                    "source": "trump_tweet",
                    "url": f"https://twitter.com/realDonaldTrump/status/{t.get('id', '')}",
                    "published_at": t.get("created_at", ""),
                }
                for t in tweets
                if t.get("text")
            ]
        except aiohttp.ClientResponseError as exc:
            logger.warning("twitter_fetch_failed", status=exc.status, error=str(exc))
            return []
        except Exception as exc:
            logger.warning("twitter_fetch_error", error=str(exc))
            return []

    # ── Prompt builder ───────────────────────────────────────────────

    def build_prompt(self, data: list[dict]) -> str:
        """Build Claude prompt asking for structured market-impact analysis."""
        content_lines = []
        for item in data[:20]:  # cap to avoid huge prompts
            text = item.get("text", "")
            source = item.get("source", "news")
            published = item.get("published_at", "")
            if text:
                content_lines.append(f"[{source}] {published}: {text}")

        if not content_lines:
            return ""

        content = "\n".join(content_lines)

        return f"""Analyze these recent Trump statements/news and their market impact. Return ONLY valid JSON:
{{
  "dominant_action": "tariff|sanction|threat|deal|crypto_policy|other",
  "categories": ["trade_war", "monetary_policy", "geopolitical", "crypto_regulation", "fiscal_policy", "election"],
  "sentiment": -1.0,
  "magnitude": 0.8,
  "affected_assets": {{
    "gold": 0.3,
    "oil": -0.2,
    "sp500": -0.5,
    "crypto": 0.1,
    "usd": 0.4
  }},
  "key_countries": ["China", "Canada"],
  "key_themes": ["tariffs", "retaliation"],
  "urgency": "breaking|developing|background",
  "narrative_summary": "Two sentence Jiang-style summary of the situation and its historical parallel."
}}

Statements/news to analyze:
{content}"""

    # ── Analysis override ────────────────────────────────────────────

    async def analyze(self) -> AgentInsight:
        """Fetch data, call Claude, build NarrativeEvents per category found."""
        try:
            data = await self.fetch_data()
        except Exception as exc:
            logger.warning("trump_monitor_analyze_fetch_failed", error=str(exc))
            return self._empty_insight()

        if not data:
            return self._empty_insight()

        prompt = self.build_prompt(data)
        if not prompt:
            return self._empty_insight()

        raw_text, tokens = await self._call_claude(prompt)
        if not raw_text:
            return self._empty_insight()

        parsed = self._parse_json_from_text(raw_text)
        cost = tokens * self._cost_per_token

        # Build one NarrativeEvent per identified category
        events: list[NarrativeEvent] = []
        sentiment = float(parsed.get("sentiment", 0.0))
        magnitude = float(parsed.get("magnitude", 0.5))
        summary = str(parsed.get("narrative_summary", "Trump statement analysis"))
        key_themes: list[str] = parsed.get("key_themes", [])
        keywords = self._extract_keywords(summary) + [
            str(t) for t in key_themes[:5]
        ]

        for cat_label in parsed.get("categories", []):
            cat = _CATEGORY_MAP.get(str(cat_label).lower(), NarrativeCategory.OTHER)
            events.append(
                NarrativeEvent(
                    event_id=str(uuid.uuid4())[:12],
                    source="trump_statement",
                    content=summary,
                    timestamp=datetime.now(timezone.utc),
                    category=cat,
                    sentiment=max(-1.0, min(1.0, sentiment)),
                    magnitude=max(0.0, min(1.0, magnitude)),
                    keywords=keywords,
                    metadata={
                        "dominant_action": parsed.get("dominant_action", "other"),
                        "urgency": parsed.get("urgency", "background"),
                        "key_countries": parsed.get("key_countries", []),
                    },
                )
            )

        # Fall back to a single OTHER event if Claude returned no categories
        if not events and parsed:
            events.append(
                NarrativeEvent(
                    event_id=str(uuid.uuid4())[:12],
                    source="trump_statement",
                    content=summary,
                    timestamp=datetime.now(timezone.utc),
                    category=NarrativeCategory.OTHER,
                    sentiment=max(-1.0, min(1.0, sentiment)),
                    magnitude=max(0.0, min(1.0, magnitude)),
                    keywords=keywords,
                )
            )

        cross_asset: dict[str, float] = parsed.get("affected_assets", {})
        confidence = magnitude

        logger.info(
            "trump_monitor_analyzed",
            events=len(events),
            dominant_action=parsed.get("dominant_action"),
            urgency=parsed.get("urgency"),
            tokens=tokens,
            cost_usd=round(cost, 6),
        )

        return AgentInsight(
            agent_name=self.name,
            events=events,
            cross_asset_impact=cross_asset,
            raw_analysis=raw_text,
            confidence=confidence,
            tokens_used=tokens,
            cost_usd=cost,
        )
