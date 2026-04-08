"""Crypto regulation and sentiment agent."""
from __future__ import annotations
import asyncio, uuid, aiohttp, structlog
from datetime import datetime, timezone, timedelta
from polymarket_bot.data.models import NarrativeEvent, NarrativeCategory
from polymarket_bot.agents.base_agent import BaseAgent, AgentInsight

logger = structlog.get_logger()


class CryptoAgent(BaseAgent):
    name = "crypto"

    def __init__(self, anthropic_client, config: dict) -> None:
        super().__init__(anthropic_client, config)
        self._news_api_key: str = config.get("news_api_key", "")

    async def fetch_data(self) -> list[dict]:
        results: list[dict] = []
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
                # Fear/Greed index (always free)
                try:
                    async with s.get("https://api.alternative.me/fng/?limit=1") as r:
                        if r.status == 200:
                            d = await r.json(content_type=None)
                            fg = d.get("data", [{}])[0]
                            results.append({
                                "text": f"Crypto Fear & Greed Index: {fg.get('value','?')} ({fg.get('value_classification','?')})",
                                "source": "fear_greed",
                                "value": int(fg.get("value", 50)),
                            })
                except Exception:
                    pass
                # NewsAPI crypto regulation headlines
                if self._news_api_key:
                    since = (datetime.now(timezone.utc) - timedelta(hours=4)).strftime("%Y-%m-%dT%H:%M:%SZ")
                    params = dict(q="bitcoin ethereum SEC crypto regulation stablecoin ETF", language="en",
                                  sortBy="publishedAt", pageSize=10, apiKey=self._news_api_key,
                                  from_param=since)
                    try:
                        async with s.get("https://newsapi.org/v2/everything", params=params) as r:
                            if r.status == 200:
                                arts = (await r.json()).get("articles", [])
                                for a in arts:
                                    results.append({"text": f"{a['title']}. {a.get('description','')}", "source": "newsapi"})
                    except Exception:
                        pass
        except Exception as exc:
            logger.warning("crypto_agent_fetch_failed", error=str(exc))
        return results

    def build_prompt(self, data: list[dict]) -> str:
        content = "\n".join(d["text"] for d in data[:12])
        return f"""Analyze these crypto news/sentiment signals. Return ONLY valid JSON:
{{
  "regulatory_sentiment": -1.0,
  "btc_trend": "bullish|bearish|neutral",
  "fear_greed_value": 50,
  "key_regulatory_events": ["SEC ETF ruling"],
  "sentiment": -0.3,
  "magnitude": 0.6,
  "narrative_summary": "One Jiang-style sentence about current crypto situation and historical parallel."
}}
Signals:
{content}"""

    async def analyze(self) -> AgentInsight:
        data = await self.fetch_data()
        if not data:
            return self._empty_insight()

        # Rule-based from fear/greed without Claude if no key
        fg_item = next((d for d in data if d.get("source") == "fear_greed"), None)
        fg_value = fg_item.get("value", 50) if fg_item else 50

        raw_text, tokens = await self._call_claude(self.build_prompt(data))
        parsed = self._parse_json_from_text(raw_text) if raw_text else {}

        sentiment = float(parsed.get("sentiment", -0.3 if fg_value < 30 else 0.2 if fg_value > 70 else 0.0))
        magnitude = float(parsed.get("magnitude", 0.6 if fg_value < 25 or fg_value > 75 else 0.3))
        summary = parsed.get("narrative_summary", f"Crypto Fear/Greed at {fg_value}")

        events: list[NarrativeEvent] = []
        if magnitude > 0.25:
            events.append(NarrativeEvent(
                event_id=str(uuid.uuid4())[:12], source="crypto_agent",
                content=summary, timestamp=datetime.now(timezone.utc),
                category=NarrativeCategory.CRYPTO_REGULATION,
                sentiment=max(-1.0, min(1.0, sentiment)),
                magnitude=max(0.0, min(1.0, magnitude)),
                keywords=self._extract_keywords(summary) + ["crypto", "bitcoin", "regulation"],
                metadata={"fear_greed": fg_value},
            ))

        return AgentInsight(
            agent_name=self.name, events=events,
            cross_asset_impact={"crypto": sentiment},
            raw_analysis=raw_text, confidence=magnitude,
            tokens_used=tokens, cost_usd=tokens * self._cost_per_token,
        )
