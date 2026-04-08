"""Economics / monetary-policy monitoring agent.

Fetches Federal Reserve, CPI, and interest-rate news from NewsAPI (and
optionally FRED) then asks Claude to characterise the macro regime and
produce ``NarrativeEvent`` objects for MONETARY_POLICY and, when the
recession risk is elevated, MARKET_CRISIS.
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

# NewsAPI query targeting Fed / CPI / macro news
_NEWSAPI_QUERY = (
    "Federal+Reserve+interest+rate+inflation+CPI+Powell"
    "+FOMC+hawkish+dovish+recession+yield"
)


class EconomicsAgent(BaseAgent):
    """Monitor macroeconomic developments and monetary policy signals.

    Data sources:
    * NewsAPI (requires ``newsapi_key`` in config)
    * FRED via optional ``fred_client`` passed in config (duck-typed, expected
      to have a ``get_series(series_id)`` method returning a list of dicts)
    """

    name = "economics"

    def __init__(self, anthropic_client: Any, config: dict) -> None:
        super().__init__(anthropic_client, config)
        self._newsapi_key: str = config.get("newsapi_key", "")
        self._fred_client: Any = config.get("fred_client")  # optional

    # ── Data fetching ────────────────────────────────────────────────

    async def fetch_data(self) -> list[dict]:
        """Fetch macro news from NewsAPI and FRED data points (if available)."""
        items: list[dict] = []

        if self._newsapi_key:
            items.extend(await self._fetch_newsapi())
        else:
            logger.debug("economics_agent_no_newsapi_key")

        if self._fred_client is not None:
            items.extend(await self._fetch_fred())

        return items

    async def _fetch_newsapi(self) -> list[dict]:
        """Pull Fed / inflation news from NewsAPI (last 6 hours)."""
        from_dt = (datetime.now(timezone.utc) - timedelta(hours=6)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={_NEWSAPI_QUERY}"
            f"&language=en&sortBy=publishedAt&pageSize=15"
            f"&apiKey={self._newsapi_key}&from={from_dt}"
        )
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=20)
            ) as session:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

            articles = data.get("articles", [])
            return [
                {
                    "text": (
                        a.get("title", "") + " " + (a.get("description") or "")
                    ).strip(),
                    "source": "newsapi_economics",
                    "url": a.get("url", ""),
                    "published_at": a.get("publishedAt", ""),
                }
                for a in articles
                if a.get("title")
            ]
        except aiohttp.ClientResponseError as exc:
            logger.warning("economics_newsapi_failed", status=exc.status, error=str(exc))
            return []
        except Exception as exc:
            logger.warning("economics_newsapi_error", error=str(exc))
            return []

    async def _fetch_fred(self) -> list[dict]:
        """Pull a few key FRED series (DGS10, UNRATE, CPIAUCSL) if available."""
        series_ids = ["DGS10", "UNRATE", "CPIAUCSL"]
        items: list[dict] = []
        for series_id in series_ids:
            try:
                # Run synchronous FRED client in executor
                import asyncio  # noqa: PLC0415
                observations = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda sid=series_id: self._fred_client.get_series(sid),
                )
                if observations:
                    latest = observations[-1]
                    items.append(
                        {
                            "text": f"FRED {series_id}: {latest}",
                            "source": "fred",
                            "series_id": series_id,
                            "value": latest,
                        }
                    )
            except Exception as exc:
                logger.debug("fred_fetch_skipped", series=series_id, error=str(exc))
        return items

    # ── Prompt builder ───────────────────────────────────────────────

    def build_prompt(self, data: list[dict]) -> str:
        """Build prompt requesting a structured macro / Fed stance analysis."""
        content_lines = []
        for item in data[:20]:
            text = item.get("text", "")
            source = item.get("source", "news")
            published = item.get("published_at", "")
            if text:
                content_lines.append(f"[{source}] {published}: {text}")

        if not content_lines:
            return ""

        content = "\n".join(content_lines)

        return f"""Analyze these macroeconomic news items and their market implications.
Return ONLY valid JSON:
{{
  "fed_stance": "hawkish|dovish|neutral",
  "recession_risk": 0.3,
  "inflation_trend": "rising|falling|stable",
  "yield_curve_signal": "inverted|flat|normal|steepening",
  "dominant_narrative": "monetary_policy|recession|stagflation",
  "sentiment": -0.2,
  "magnitude": 0.6,
  "affected_assets": {{
    "bonds": 0.4,
    "gold": 0.2,
    "sp500": -0.3,
    "usd": 0.1,
    "crypto": -0.1
  }},
  "key_themes": ["rate_hike", "inflation_persistence"],
  "narrative_summary": "Two-sentence summary of the current macro regime and outlook."
}}

Macro news to analyze:
{content}"""

    # ── Analysis override ────────────────────────────────────────────

    async def analyze(self) -> AgentInsight:
        """Fetch data, call Claude, produce MONETARY_POLICY (and optionally
        MARKET_CRISIS) NarrativeEvents."""
        try:
            data = await self.fetch_data()
        except Exception as exc:
            logger.warning("economics_agent_fetch_failed", error=str(exc))
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

        sentiment = float(parsed.get("sentiment", 0.0))
        magnitude = float(parsed.get("magnitude", 0.5))
        recession_risk = float(parsed.get("recession_risk", 0.0))
        summary = str(
            parsed.get("narrative_summary", "Macroeconomic analysis")
        )
        key_themes: list[str] = parsed.get("key_themes", [])
        keywords = self._extract_keywords(summary) + [
            str(t) for t in key_themes[:5]
        ]

        events: list[NarrativeEvent] = []

        # Primary monetary policy event
        events.append(
            NarrativeEvent(
                event_id=str(uuid.uuid4())[:12],
                source=self.name,
                content=summary,
                timestamp=datetime.now(timezone.utc),
                category=NarrativeCategory.MONETARY_POLICY,
                sentiment=max(-1.0, min(1.0, sentiment)),
                magnitude=max(0.0, min(1.0, magnitude)),
                keywords=keywords,
                metadata={
                    "fed_stance": parsed.get("fed_stance", "neutral"),
                    "inflation_trend": parsed.get("inflation_trend", "stable"),
                    "yield_curve_signal": parsed.get("yield_curve_signal", "normal"),
                    "recession_risk": recession_risk,
                },
            )
        )

        # Secondary MARKET_CRISIS event when recession risk is elevated
        if recession_risk > 0.6:
            crisis_summary = (
                f"Recession risk elevated at {recession_risk:.0%}. "
                + summary
            )
            events.append(
                NarrativeEvent(
                    event_id=str(uuid.uuid4())[:12],
                    source=self.name,
                    content=crisis_summary,
                    timestamp=datetime.now(timezone.utc),
                    category=NarrativeCategory.MARKET_CRISIS,
                    sentiment=max(-1.0, min(1.0, sentiment - 0.2)),
                    magnitude=max(0.0, min(1.0, recession_risk)),
                    keywords=keywords + ["recession", "crisis"],
                    metadata={"recession_risk": recession_risk},
                )
            )

        cross_asset: dict[str, float] = parsed.get("affected_assets", {})
        confidence = magnitude

        logger.info(
            "economics_agent_analyzed",
            fed_stance=parsed.get("fed_stance"),
            recession_risk=recession_risk,
            events=len(events),
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
