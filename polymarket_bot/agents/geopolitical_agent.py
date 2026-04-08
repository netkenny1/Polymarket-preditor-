"""Geopolitical monitoring agent.

Watches active conflict zones (Ukraine/Russia, Israel/Hamas, China/Taiwan,
Iran) via NewsAPI and asks Claude to assess escalation levels and their
expected impact on safe-haven assets, oil, and Polymarket election markets.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp
import structlog

from polymarket_bot.data.models import NarrativeCategory, NarrativeEvent
from polymarket_bot.agents.base_agent import AgentInsight, BaseAgent

logger = structlog.get_logger()

# NewsAPI queries for major conflict theatres
_GEO_QUERIES: list[tuple[str, str]] = [
    ("ukraine_russia", "Ukraine Russia war military ceasefire NATO"),
    ("israel_hamas", "Israel Hamas Gaza war ceasefire Hezbollah"),
    ("china_taiwan", "China Taiwan military strait invasion"),
    ("iran_nuclear", "Iran nuclear sanctions IAEA enrichment"),
]

# Escalation threshold above which we create a NarrativeEvent
_ESCALATION_THRESHOLD: float = 0.4


class GeopoliticalAgent(BaseAgent):
    """Monitor geopolitical hotspots for market-moving escalation signals.

    For each active conflict where Claude rates escalation > 0.4 we emit
    a GEOPOLITICAL NarrativeEvent weighted by that escalation score.
    """

    name = "geopolitical"

    def __init__(self, anthropic_client: Any, config: dict) -> None:
        super().__init__(anthropic_client, config)
        self._newsapi_key: str = config.get("newsapi_key", "")

    # ── Data fetching ────────────────────────────────────────────────

    async def fetch_data(self) -> list[dict]:
        """Fetch news across all monitored conflict zones concurrently."""
        if not self._newsapi_key:
            logger.debug("geopolitical_agent_no_newsapi_key")
            return []

        from_dt = (datetime.now(timezone.utc) - timedelta(hours=6)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        items: list[dict] = []

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=20)
        ) as session:
            import asyncio  # noqa: PLC0415

            tasks = [
                asyncio.create_task(
                    self._fetch_one_query(session, region, query, from_dt)
                )
                for region, query in _GEO_QUERIES
            ]
            for coro in asyncio.as_completed(tasks):
                try:
                    batch = await coro
                    items.extend(batch)
                except Exception as exc:
                    logger.warning("geopolitical_fetch_error", error=str(exc))

        return items

    async def _fetch_one_query(
        self,
        session: aiohttp.ClientSession,
        region: str,
        query: str,
        from_dt: str,
    ) -> list[dict]:
        """Fetch NewsAPI articles for a single geopolitical query."""
        url = (
            "https://newsapi.org/v2/everything"
            f"?q={query.replace(' ', '+')}"
            f"&language=en&sortBy=publishedAt&pageSize=10"
            f"&apiKey={self._newsapi_key}&from={from_dt}"
        )
        try:
            async with session.get(url) as resp:
                resp.raise_for_status()
                data = await resp.json()
            articles = data.get("articles", [])
            return [
                {
                    "text": (
                        a.get("title", "") + " " + (a.get("description") or "")
                    ).strip(),
                    "source": f"newsapi_{region}",
                    "region": region,
                    "url": a.get("url", ""),
                    "published_at": a.get("publishedAt", ""),
                }
                for a in articles
                if a.get("title")
            ]
        except aiohttp.ClientResponseError as exc:
            logger.warning(
                "geopolitical_newsapi_failed",
                region=region,
                status=exc.status,
                error=str(exc),
            )
            return []
        except Exception as exc:
            logger.warning("geopolitical_newsapi_error", region=region, error=str(exc))
            return []

    # ── Prompt builder ───────────────────────────────────────────────

    def build_prompt(self, data: list[dict]) -> str:
        """Build prompt requesting per-region escalation assessment."""
        content_lines = []
        for item in data[:30]:
            text = item.get("text", "")
            region = item.get("region", "unknown")
            published = item.get("published_at", "")
            if text:
                content_lines.append(f"[{region}] {published}: {text}")

        if not content_lines:
            return ""

        content = "\n".join(content_lines)

        return f"""Analyze these geopolitical news items and assess conflict escalation.
Return ONLY valid JSON:
{{
  "conflict_regions": [
    {{
      "region": "ukraine_russia",
      "escalation_level": 0.7,
      "trend": "escalating|stable|de-escalating",
      "key_development": "Brief description of the most significant recent development."
    }}
  ],
  "oil_impact": 0.3,
  "gold_impact": 0.4,
  "election_impacts": {{
    "us_election": 0.1,
    "eu_elections": 0.05
  }},
  "sentiment": -0.5,
  "magnitude": 0.7,
  "affected_assets": {{
    "oil": 0.3,
    "gold": 0.4,
    "defense_stocks": 0.5,
    "sp500": -0.2,
    "usd": 0.1
  }},
  "narrative_summary": "Two-sentence summary of the dominant geopolitical risk and its historical parallel."
}}

Geopolitical news to analyze:
{content}"""

    # ── Analysis override ────────────────────────────────────────────

    async def analyze(self) -> AgentInsight:
        """Fetch news, call Claude, produce one GEOPOLITICAL event per
        active conflict region with escalation above the threshold."""
        try:
            data = await self.fetch_data()
        except Exception as exc:
            logger.warning("geopolitical_agent_fetch_failed", error=str(exc))
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

        sentiment = float(parsed.get("sentiment", -0.3))
        magnitude = float(parsed.get("magnitude", 0.5))
        summary = str(
            parsed.get("narrative_summary", "Geopolitical tension analysis")
        )
        base_keywords = self._extract_keywords(summary)

        events: list[NarrativeEvent] = []

        for region_data in parsed.get("conflict_regions", []):
            if not isinstance(region_data, dict):
                continue
            escalation = float(region_data.get("escalation_level", 0.0))
            if escalation < _ESCALATION_THRESHOLD:
                continue

            region_name = str(region_data.get("region", "unknown"))
            key_dev = str(
                region_data.get("key_development", f"Escalation in {region_name}")
            )
            region_keywords = self._extract_keywords(key_dev) + base_keywords
            # Scale sentiment by escalation
            event_sentiment = max(-1.0, min(1.0, sentiment * (1.0 + escalation * 0.3)))

            events.append(
                NarrativeEvent(
                    event_id=str(uuid.uuid4())[:12],
                    source=self.name,
                    content=key_dev,
                    timestamp=datetime.now(timezone.utc),
                    category=NarrativeCategory.GEOPOLITICAL,
                    sentiment=event_sentiment,
                    magnitude=max(0.0, min(1.0, escalation)),
                    keywords=region_keywords[:10],
                    metadata={
                        "region": region_name,
                        "escalation_level": escalation,
                        "trend": region_data.get("trend", "stable"),
                        "oil_impact": parsed.get("oil_impact", 0.0),
                        "gold_impact": parsed.get("gold_impact", 0.0),
                    },
                )
            )

        # Ensure at least one event if Claude identified significant risk
        if not events and magnitude > 0.5:
            events.append(
                NarrativeEvent(
                    event_id=str(uuid.uuid4())[:12],
                    source=self.name,
                    content=summary,
                    timestamp=datetime.now(timezone.utc),
                    category=NarrativeCategory.GEOPOLITICAL,
                    sentiment=max(-1.0, min(1.0, sentiment)),
                    magnitude=max(0.0, min(1.0, magnitude)),
                    keywords=base_keywords,
                )
            )

        cross_asset: dict[str, float] = parsed.get("affected_assets", {})
        confidence = magnitude

        logger.info(
            "geopolitical_agent_analyzed",
            active_regions=len(events),
            oil_impact=parsed.get("oil_impact"),
            gold_impact=parsed.get("gold_impact"),
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
