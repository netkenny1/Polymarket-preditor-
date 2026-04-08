"""Predictive History Agent — implements Jiang Xueqin's psychohistory methodology.

This agent applies the "Predictive History" YouTube channel's framework:
1. Identify the closest historical analog to current events
2. Extract Jiang's 5 structural variables (elite cohesion, fiscal capacity, etc.)
3. Model key actors via game theory (who wants what, what are they likely to do)
4. Identify current phase of the historical cycle (escalation/peak/resolution/aftermath)
5. Project what happens next based on the analog

Runs on a 2-hour cycle using claude-opus-4-6 for deep structural analysis.
Output feeds directly into NarrativeEngine and enriches HistoricalPatternMatcher.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any

import aiohttp
import structlog

from polymarket_bot.data.models import (
    NarrativeCategory,
    NarrativeEvent,
    HistoricalPattern,
    HistoricalPhase,
)
from polymarket_bot.agents.base_agent import BaseAgent, AgentInsight

logger = structlog.get_logger()

# ── Built-in historical analog library ───────────────────────────────────────
# Each entry: (id, name, category, description, keywords, outcome_direction, phases_hint)
HISTORICAL_ANALOGS: list[dict[str, Any]] = [
    {
        "id": "smoot_hawley_1930",
        "name": "Smoot-Hawley Tariff Escalation (1930)",
        "category": "trade_war",
        "description": (
            "US imposed sweeping tariffs; trading partners retaliated; global trade collapsed 66%; "
            "Great Depression deepened. Pattern: protectionist spiral → recession → political backlash."
        ),
        "keywords": ["tariff", "trade war", "retaliatory", "protectionism", "import", "export", "customs"],
        "outcome_direction": -1.0,
        "phases": ["Announcement & Initial Retaliation", "Escalation & Market Collapse", "Diplomatic Breakdown", "Long Recession"],
    },
    {
        "id": "sicilian_expedition_415bce",
        "name": "Athenian Sicilian Expedition (415-413 BCE)",
        "category": "geopolitical",
        "description": (
            "Athens overextended militarily; underestimated local resistance; supply lines failed; "
            "catastrophic defeat. Pattern: imperial overreach → logistical failure → strategic reversal."
        ),
        "keywords": ["military", "overreach", "invasion", "war", "troops", "occupation", "supply"],
        "outcome_direction": -0.9,
        "phases": ["Strategic Overconfidence", "Initial Engagement", "Logistical Breakdown", "Catastrophic Defeat"],
    },
    {
        "id": "stagflation_1970s",
        "name": "1970s Stagflation Cycle",
        "category": "monetary_policy",
        "description": (
            "Oil shock + loose monetary policy → stagflation. Fed eventually forced into painful rate hikes. "
            "Pattern: supply-side inflation + demand stimulus → stagflation trap."
        ),
        "keywords": ["stagflation", "inflation", "fed", "oil", "supply shock", "recession", "interest rate"],
        "outcome_direction": -0.7,
        "phases": ["Oil Shock Trigger", "Stagflation Recognition", "Policy Paralysis", "Volcker Shock Resolution"],
    },
    {
        "id": "financial_crisis_2008",
        "name": "2008 Global Financial Crisis",
        "category": "market_crisis",
        "description": (
            "Credit bubble burst; Lehman collapse; contagion spread globally. "
            "Pattern: leverage buildup → trigger event → cascade failure → policy intervention."
        ),
        "keywords": ["credit", "bank", "crisis", "contagion", "lehman", "mortgage", "default", "liquidity"],
        "outcome_direction": -1.0,
        "phases": ["Credit Bubble Recognition", "Trigger Event", "Cascade Failure", "Government Intervention", "Recovery"],
    },
    {
        "id": "cuban_missile_crisis_1962",
        "name": "Cuban Missile Crisis (1962)",
        "category": "geopolitical",
        "description": (
            "Brinkmanship between superpowers; back-channel diplomacy resolved crisis. "
            "Pattern: escalation to threshold → credible threat exchange → negotiated de-escalation."
        ),
        "keywords": ["nuclear", "brinkmanship", "nato", "russia", "china", "missile", "escalation", "diplomacy"],
        "outcome_direction": 0.3,  # resolution positive
        "phases": ["Discovery & Confrontation", "Brinkmanship Peak", "Back-channel Negotiation", "Resolution & De-escalation"],
    },
    {
        "id": "trade_war_2018",
        "name": "US-China Trade War (2018-2019)",
        "category": "trade_war",
        "description": (
            "US imposed tariffs on China; tit-for-tat retaliation; markets volatile; "
            "eventually Phase 1 deal negotiated. Pattern: tariff escalation → market pain → partial deal."
        ),
        "keywords": ["china", "tariff", "trade", "phase one", "decoupling", "supply chain", "fentanyl"],
        "outcome_direction": -0.4,
        "phases": ["Initial Tariffs", "Retaliation Spiral", "Market Volatility Peak", "Negotiation", "Phase 1 Deal"],
    },
    {
        "id": "dotcom_bubble_2000",
        "name": "Dot-com Bubble (1999-2001)",
        "category": "market_crisis",
        "description": (
            "Tech speculation detached from fundamentals; eventual crash -78%. "
            "Pattern: narrative-driven euphoria → fundamentals ignored → sudden repricing."
        ),
        "keywords": ["tech", "bubble", "nasdaq", "speculation", "ai", "crypto", "overvalued"],
        "outcome_direction": -0.9,
        "phases": ["Euphoria & Speculation", "Peak Valuations", "First Cracks", "Cascade Decline", "Repricing"],
    },
    {
        "id": "nixon_china_1972",
        "name": "Nixon Opening to China (1972)",
        "category": "geopolitical",
        "description": (
            "Unexpected diplomatic breakthrough; shocked allies; restructured global order. "
            "Pattern: outsider thinking breaks deadlock → new strategic alignment possible."
        ),
        "keywords": ["china", "diplomatic", "deal", "alliance", "breakthrough", "nato", "pivot"],
        "outcome_direction": 0.6,
        "phases": ["Secret Negotiations", "Shocking Announcement", "Implementation", "Strategic Realignment"],
    },
    {
        "id": "asian_financial_crisis_1997",
        "name": "Asian Financial Crisis (1997-1998)",
        "category": "market_crisis",
        "description": (
            "Currency pegs broke; contagion spread from Thailand to Korea, Indonesia. "
            "Pattern: fixed exchange rate stress → speculative attack → IMF intervention → painful adjustment."
        ),
        "keywords": ["currency", "emerging market", "imf", "dollar", "peg", "devaluation", "contagion"],
        "outcome_direction": -0.8,
        "phases": ["Currency Peg Stress", "Speculative Attack", "Contagion Spread", "IMF Intervention", "Recovery"],
    },
    {
        "id": "covid_supply_chain_2020",
        "name": "COVID Supply Chain Disruption (2020-2021)",
        "category": "trade_war",
        "description": (
            "Global supply chains broken; reshoring debate accelerated; commodity spikes. "
            "Pattern: black swan shock → deglobalization narrative → commodity supercycle."
        ),
        "keywords": ["supply chain", "reshoring", "pandemic", "commodity", "shortage", "inflation", "disruption"],
        "outcome_direction": -0.5,
        "phases": ["Shock & Shutdown", "Supply Disruption", "Inflation Surge", "Normalization"],
    },
]

_ANALOG_IDS = {a["id"]: a for a in HISTORICAL_ANALOGS}
_ANALOG_LIST_TEXT = "\n".join(
    f"- {a['id']}: {a['name']} — {a['description'][:120]}"
    for a in HISTORICAL_ANALOGS
)


class PredictiveHistoryAgent(BaseAgent):
    """Applies Jiang Xueqin's Predictive History methodology via Claude opus-4."""

    name = "predictive_history"
    model = "claude-opus-4-6"  # Deep structural analysis requires best model

    def __init__(self, anthropic_client, config: dict) -> None:
        super().__init__(anthropic_client, config)
        self._news_api_key: str = config.get("news_api_key", "")
        self._last_analog: str | None = None  # Track continuity across cycles
        self._last_phase: str | None = None

    async def fetch_data(self) -> list[dict]:
        """Fetch current geopolitical/economic headlines for context."""
        results: list[dict] = []
        if not self._news_api_key:
            return results
        queries = [
            "Trump tariff trade war sanction China",
            "Ukraine Russia war NATO military",
            "Federal Reserve inflation recession",
            "Israel Iran Middle East conflict",
            "stock market crash economic crisis",
        ]
        since = (datetime.now(timezone.utc) - timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as s:
            tasks = []
            for q in queries:
                params = dict(q=q, language="en", sortBy="publishedAt",
                              pageSize=5, apiKey=self._news_api_key, from_param=since)
                tasks.append(self._fetch_one(s, params))
            gathered = await asyncio.gather(*tasks, return_exceptions=True)
            for batch in gathered:
                if isinstance(batch, list):
                    results.extend(batch)
        return results[:30]

    async def _fetch_one(self, session: aiohttp.ClientSession, params: dict) -> list[dict]:
        try:
            async with session.get("https://newsapi.org/v2/everything", params=params) as r:
                if r.status == 200:
                    arts = (await r.json()).get("articles", [])
                    return [{"text": f"{a['title']}. {a.get('description', '')}", "source": "newsapi"} for a in arts]
        except Exception:
            pass
        return []

    def build_prompt(self, data: list[dict]) -> str:
        headlines = "\n".join(f"- {d['text'][:200]}" for d in data[:25] if d.get("text"))
        prev_context = ""
        if self._last_analog:
            prev_context = f"\nPrevious analysis identified analog: {self._last_analog}, phase: {self._last_phase}. Update if needed."

        return f"""You are Jiang Xueqin applying the Predictive History methodology to current events.

AVAILABLE HISTORICAL ANALOGS:
{_ANALOG_LIST_TEXT}

CURRENT HEADLINES ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}):
{headlines}
{prev_context}

Apply the full Predictive History methodology:
1. HISTORICAL ANALOG: Which analog fits best? (use the id)
2. STRUCTURAL VARIABLES (0-1 scale): Extract from current events
3. ACTOR ANALYSIS: Key players and their incentives (game theory)
4. PHASE: Where are we in the historical cycle?
5. PROJECTION: What does history say comes next?

Return ONLY valid JSON:
{{
  "historical_analog_id": "trade_war_2018",
  "historical_analog_name": "US-China Trade War (2018-2019)",
  "analog_similarity": 0.82,
  "structural_variables": {{
    "elite_cohesion": 0.3,
    "fiscal_capacity": 0.6,
    "public_opinion_alignment": 0.4,
    "actor_incentive_alignment": 0.2,
    "supply_chain_integrity": 0.5
  }},
  "actors": [
    {{"name": "Trump Administration", "incentive": "Domestic political wins, reduce trade deficit", "likely_move": "Escalate tariffs further before negotiating"}},
    {{"name": "China", "incentive": "Protect export economy, avoid humiliation", "likely_move": "Selective retaliation, wait for leverage"}},
    {{"name": "Federal Reserve", "incentive": "Price stability amid supply shock", "likely_move": "Hold rates, monitor inflation"}}
  ],
  "current_phase": "escalation",
  "next_phase": "market_volatility_peak",
  "projected_outcome": "Further tariff escalation before forced negotiation. Markets will price in recession risk.",
  "timeline": "3-6 months to resolution",
  "market_impact": {{
    "geopolitical_risk": -0.7,
    "commodity_direction": 0.3,
    "equity_direction": -0.5,
    "crypto_direction": -0.2,
    "usd_direction": 0.3,
    "polymarket_categories": ["trade_war", "monetary_policy", "election"]
  }},
  "confidence": 0.75,
  "narrative_summary": "Current US tariff escalation mirrors the Smoot-Hawley pattern: retaliatory spirals historically deepened recessions rather than solving trade deficits. We are in Phase 2 (Retaliation Spiral) — history suggests markets have not yet priced the full downside."
}}"""

    async def analyze(self) -> AgentInsight:
        """Full Predictive History analysis cycle."""
        data = await self.fetch_data()
        if not data and not self._last_analog:
            return self._empty_insight()

        # Use whatever data we have, even stale
        raw_text, tokens = await self._call_claude(
            self.build_prompt(data if data else [{"text": "No new headlines — continue prior analysis."}]),
            use_deep_model=True,  # Always use opus for this agent
        )
        if not raw_text:
            return self._empty_insight()

        parsed = self._parse_json_from_text(raw_text)
        if not parsed:
            return self._empty_insight()

        cost = tokens * 0.000_015  # opus is more expensive

        # Update continuity tracking
        self._last_analog = parsed.get("historical_analog_id", self._last_analog)
        self._last_phase = parsed.get("current_phase", self._last_phase)

        events = self._build_narrative_events(parsed)
        cross_asset = parsed.get("market_impact", {})

        logger.info(
            "predictive_history_analyzed",
            analog=self._last_analog,
            phase=self._last_phase,
            similarity=parsed.get("analog_similarity", 0),
            events=len(events),
            confidence=parsed.get("confidence", 0),
        )

        return AgentInsight(
            agent_name=self.name,
            events=events,
            cross_asset_impact=cross_asset,
            raw_analysis=raw_text,
            confidence=float(parsed.get("confidence", 0.5)),
            tokens_used=tokens,
            cost_usd=cost,
        )

    def _build_narrative_events(self, parsed: dict) -> list[NarrativeEvent]:
        """Convert Jiang analysis into NarrativeEvents for the NarrativeEngine."""
        events: list[NarrativeEvent] = []
        summary = parsed.get("narrative_summary", "")
        if not summary:
            return events

        market_impact = parsed.get("market_impact", {})
        equity_dir = float(market_impact.get("equity_direction", 0.0))
        confidence = float(parsed.get("confidence", 0.5))
        analog_id = parsed.get("historical_analog_id", "unknown")
        analog = _ANALOG_IDS.get(analog_id, {})

        # Map the analog's category to NarrativeCategory
        cat_map = {
            "trade_war": NarrativeCategory.TRADE_WAR,
            "geopolitical": NarrativeCategory.GEOPOLITICAL,
            "monetary_policy": NarrativeCategory.MONETARY_POLICY,
            "market_crisis": NarrativeCategory.MARKET_CRISIS,
        }
        cat_str = analog.get("category", "other")
        category = cat_map.get(cat_str, NarrativeCategory.OTHER)

        # Primary event from the narrative summary
        keywords = self._extract_keywords(summary)
        keywords += analog.get("keywords", [])[:5]
        events.append(NarrativeEvent(
            event_id=str(uuid.uuid4())[:12],
            source="predictive_history",
            content=summary,
            timestamp=datetime.now(timezone.utc),
            category=category,
            sentiment=max(-1.0, min(1.0, equity_dir)),
            magnitude=max(0.1, min(1.0, confidence)),
            keywords=keywords[:15],
            metadata={
                "historical_analog": analog_id,
                "analog_name": parsed.get("historical_analog_name", ""),
                "analog_similarity": parsed.get("analog_similarity", 0),
                "current_phase": parsed.get("current_phase", "unknown"),
                "next_phase": parsed.get("next_phase", "unknown"),
                "projected_outcome": parsed.get("projected_outcome", ""),
                "structural_variables": parsed.get("structural_variables", {}),
                "actors": parsed.get("actors", []),
                "timeline": parsed.get("timeline", ""),
            },
        ))

        # If actor analysis reveals specific concerns, add supplementary events
        actors = parsed.get("actors", [])
        if len(actors) >= 2 and float(parsed.get("structural_variables", {}).get("actor_incentive_alignment", 0.5)) < 0.35:
            # High actor conflict → elevated geopolitical risk event
            actor_names = ", ".join(a.get("name", "") for a in actors[:3])
            events.append(NarrativeEvent(
                event_id=str(uuid.uuid4())[:12],
                source="predictive_history_actors",
                content=f"Game-theoretic analysis: {actor_names} have deeply misaligned incentives — conflict escalation likely.",
                timestamp=datetime.now(timezone.utc),
                category=NarrativeCategory.GEOPOLITICAL,
                sentiment=-0.5,
                magnitude=0.65,
                keywords=self._extract_keywords(f"conflict escalation {actor_names}"),
                metadata={"actors": actors, "phase": parsed.get("current_phase")},
            ))

        return events

    def get_last_analog_info(self) -> dict[str, Any]:
        """Return current analog and phase for use by orchestrator."""
        return {
            "analog_id": self._last_analog,
            "phase": self._last_phase,
            "analog_data": _ANALOG_IDS.get(self._last_analog or "", {}),
        }
