"""Market-wide sentiment agent — VIX, Fear/Greed, S&P signals. No Claude needed."""
from __future__ import annotations
import asyncio, uuid, aiohttp, structlog
from datetime import datetime, timezone
from polymarket_bot.data.models import NarrativeEvent, NarrativeCategory
from polymarket_bot.agents.base_agent import BaseAgent, AgentInsight

logger = structlog.get_logger()


class MarketSentimentAgent(BaseAgent):
    name = "market_sentiment"

    def __init__(self, anthropic_client, config: dict) -> None:
        super().__init__(anthropic_client, config)
        self._av_key: str = config.get("alpha_vantage_api_key", "")

    async def fetch_data(self) -> list[dict]:
        results: list[dict] = []
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
            # Always fetch Fear/Greed (free)
            try:
                async with s.get("https://api.alternative.me/fng/?limit=1") as r:
                    if r.status == 200:
                        d = await r.json(content_type=None)
                        fg = d.get("data", [{}])[0]
                        results.append({"type": "fear_greed", "value": int(fg.get("value", 50)),
                                        "classification": fg.get("value_classification", "Neutral")})
            except Exception:
                pass
            # Alpha Vantage SPY daily
            if self._av_key:
                try:
                    params = dict(function="TIME_SERIES_DAILY_ADJUSTED", symbol="SPY",
                                  outputsize="compact", apikey=self._av_key)
                    async with s.get("https://www.alphavantage.co/query", params=params) as r:
                        if r.status == 200:
                            ts = (await r.json()).get("Time Series (Daily)", {})
                            dates = sorted(ts.keys(), reverse=True)[:2]
                            if len(dates) == 2:
                                today_close = float(ts[dates[0]]["4. close"])
                                prev_close = float(ts[dates[1]]["4. close"])
                                chg = (today_close - prev_close) / prev_close * 100
                                results.append({"type": "spy_change", "value": round(chg, 2)})
                except Exception:
                    pass
        return results

    def build_prompt(self, data: list[dict]) -> str:
        return ""  # Pure rule-based — no Claude needed

    async def analyze(self) -> AgentInsight:
        """Rule-based signal generation — no Claude API call."""
        try:
            data = await self.fetch_data()
        except Exception:
            return self._empty_insight()

        events: list[NarrativeEvent] = []
        cross_asset: dict[str, float] = {}

        fg = next((d for d in data if d.get("type") == "fear_greed"), None)
        spy = next((d for d in data if d.get("type") == "spy_change"), None)

        if fg:
            val = fg["value"]
            if val < 25:
                events.append(NarrativeEvent(
                    event_id=str(uuid.uuid4())[:12], source="market_sentiment_agent",
                    content=f"Extreme Fear: Crypto/Market Fear & Greed at {val} ({fg['classification']}). Historic precedent: markets near capitulation.",
                    timestamp=datetime.now(timezone.utc), category=NarrativeCategory.MARKET_CRISIS,
                    sentiment=-0.7, magnitude=0.85,
                    keywords=["fear", "panic", "selloff", "crisis", "capitulation"],
                    metadata={"fear_greed": val},
                ))
                cross_asset["sp500"] = -0.5
            elif val < 40:
                events.append(NarrativeEvent(
                    event_id=str(uuid.uuid4())[:12], source="market_sentiment_agent",
                    content=f"Fear regime: Market Fear & Greed at {val}. Risk-off sentiment elevated.",
                    timestamp=datetime.now(timezone.utc), category=NarrativeCategory.MARKET_CRISIS,
                    sentiment=-0.35, magnitude=0.5,
                    keywords=["fear", "risk", "selloff"],
                    metadata={"fear_greed": val},
                ))
                cross_asset["sp500"] = -0.25
            elif val > 75:
                events.append(NarrativeEvent(
                    event_id=str(uuid.uuid4())[:12], source="market_sentiment_agent",
                    content=f"Extreme Greed: Fear & Greed at {val}. Historically precedes corrections.",
                    timestamp=datetime.now(timezone.utc), category=NarrativeCategory.MONETARY_POLICY,
                    sentiment=0.25, magnitude=0.35,
                    keywords=["greed", "complacency", "overbought"],
                    metadata={"fear_greed": val},
                ))
                cross_asset["sp500"] = 0.15

        if spy:
            chg = spy["value"]
            if chg < -2.0:
                events.append(NarrativeEvent(
                    event_id=str(uuid.uuid4())[:12], source="market_sentiment_agent",
                    content=f"S&P 500 dropped {chg:.1f}% today — market-wide risk-off signal.",
                    timestamp=datetime.now(timezone.utc), category=NarrativeCategory.MARKET_CRISIS,
                    sentiment=-0.6, magnitude=min(abs(chg) / 5.0, 1.0),
                    keywords=["crash", "selloff", "equity", "risk"],
                    metadata={"spy_change": chg},
                ))
                cross_asset["sp500"] = chg / 10.0

        return AgentInsight(
            agent_name=self.name, events=events,
            cross_asset_impact=cross_asset,
            raw_analysis=str(data), confidence=0.7 if events else 0.0,
            tokens_used=0, cost_usd=0.0,
        )
