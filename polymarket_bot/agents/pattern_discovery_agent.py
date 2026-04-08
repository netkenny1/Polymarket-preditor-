"""Pattern Discovery Agent — finds new historical patterns from prediction accuracy data.

Runs on a slow 6-hour cycle using claude-opus-4-6 to analyze narrative history
and discover new patterns that aren't in the built-in library.
"""
from __future__ import annotations
import asyncio, uuid, structlog
from datetime import datetime, timezone
from typing import Any
from polymarket_bot.data.models import NarrativeCategory, NarrativeEvent, HistoricalPattern, HistoricalPhase
from polymarket_bot.agents.base_agent import BaseAgent, AgentInsight

logger = structlog.get_logger()


class PatternDiscoveryAgent(BaseAgent):
    name = "pattern_discovery"
    model = "claude-opus-4-6"

    def __init__(self, anthropic_client, config: dict, db: Any = None) -> None:
        super().__init__(anthropic_client, config)
        self._db = db  # SQLiteStore for narrative history queries

    async def fetch_data(self) -> list[dict]:
        if not self._db:
            return []
        try:
            history = self._db.get_narrative_history(days_back=30)
            return [dict(h) for h in history[:100]]
        except Exception as exc:
            logger.warning("pattern_discovery_fetch_failed", error=str(exc))
            return []

    def build_prompt(self, data: list[dict]) -> str:
        if not data:
            summary = "No narrative history available yet."
        else:
            # Summarize narrative history
            by_category: dict[str, list] = {}
            for d in data:
                c = d.get("category", "other")
                by_category.setdefault(c, []).append(d)
            summary_parts = []
            for cat, entries in by_category.items():
                avg_strength = sum(e.get("strength", 0) for e in entries) / len(entries)
                avg_dir = sum(e.get("predicted_direction", 0) for e in entries) / len(entries)
                summary_parts.append(f"- {cat}: {len(entries)} narratives, avg strength={avg_strength:.2f}, avg direction={avg_dir:.2f}")
            summary = "\n".join(summary_parts)

        return f"""You are a market historian analyzing Polymarket prediction narrative patterns.

NARRATIVE HISTORY SUMMARY (last 30 days):
{summary}

Identify any NEW recurring patterns not covered by these standard categories:
- trade_war, monetary_policy, geopolitical, crypto_regulation, fiscal_policy, election, market_crisis

For each new pattern you discover, return a JSON array of pattern objects.
Return ONLY valid JSON:
[
  {{
    "pattern_id": "trump_social_media_shock_2025",
    "name": "Trump Social Media Market Shock",
    "category": "trade_war",
    "description": "Sudden Trump posts about tariffs/sanctions cause immediate market reactions followed by reversal within 48-72 hours as policy details emerge.",
    "trigger_keywords": ["trump", "tariff", "truth social", "executive order", "sanction"],
    "outcome_direction": -0.3,
    "outcome_magnitude": 0.6,
    "timeline_days": 3,
    "phases": [
      {{"phase_name": "Announcement Shock", "duration_days": 1, "direction": -0.8, "keywords": ["breaking", "tariff", "trump"]}},
      {{"phase_name": "Details Emerge", "duration_days": 1, "direction": 0.2, "keywords": ["clarification", "exemption", "detail"]}},
      {{"phase_name": "Market Repricing", "duration_days": 1, "direction": -0.3, "keywords": ["impact", "assess", "trade"]}}
    ],
    "confidence": 0.65,
    "weight_adjustments": []
  }}
]

If no new patterns are discovered, return an empty array: []"""

    async def run_discovery(self) -> list[HistoricalPattern]:
        """Run pattern discovery and return new HistoricalPattern objects."""
        data = await self.fetch_data()
        raw_text, tokens = await self._call_claude(self.build_prompt(data), use_deep_model=True)
        if not raw_text:
            return []
        parsed = self._parse_json_from_text(raw_text)
        if not isinstance(parsed, list):
            # Try to extract list from dict
            if isinstance(parsed, dict):
                parsed = parsed.get("patterns", parsed.get("new_patterns", []))
            else:
                return []
        patterns: list[HistoricalPattern] = []
        cat_map = {
            "trade_war": NarrativeCategory.TRADE_WAR,
            "monetary_policy": NarrativeCategory.MONETARY_POLICY,
            "geopolitical": NarrativeCategory.GEOPOLITICAL,
            "crypto_regulation": NarrativeCategory.CRYPTO_REGULATION,
            "fiscal_policy": NarrativeCategory.FISCAL_POLICY,
            "election": NarrativeCategory.ELECTION,
            "market_crisis": NarrativeCategory.MARKET_CRISIS,
        }
        for p in parsed:
            if not isinstance(p, dict) or not p.get("pattern_id"):
                continue
            cat = cat_map.get(p.get("category", "other"), NarrativeCategory.OTHER)
            phases = [
                HistoricalPhase(
                    phase_name=ph.get("phase_name", f"Phase {i+1}"),
                    description=ph.get("phase_name", ""),
                    duration_days=int(ph.get("duration_days", 1)),
                    keywords=ph.get("keywords", []),
                    sequence_index=i,
                )
                for i, ph in enumerate(p.get("phases", []))
            ]
            hp = HistoricalPattern(
                pattern_id=p["pattern_id"],
                name=p.get("name", p["pattern_id"]),
                category=cat,
                description=p.get("description", ""),
                trigger_keywords=p.get("trigger_keywords", []),
                timeline_days=int(p.get("timeline_days", 7)),
                outcome_direction=float(p.get("outcome_direction", 0.0)),
                outcome_magnitude=float(p.get("outcome_magnitude", 0.5)),
                phases=phases,
                source_period="auto_discovered",
            )
            patterns.append(hp)
            logger.info("pattern_discovered", pattern_id=hp.pattern_id, name=hp.name)
        # Save to DB
        if self._db:
            for hp in patterns:
                try:
                    self._db.save_pattern_weight(hp.pattern_id, weight=0.3, auto_discovered=True)
                except Exception:
                    pass
        cost = tokens * 0.000_015
        if self._db:
            try:
                self._db.save_agent_insight(self.name, raw_text[:5000], 0, tokens, cost)
            except Exception:
                pass
        return patterns

    async def fetch_data(self) -> list[dict]:
        if not self._db:
            return []
        try:
            return self._db.get_narrative_history(days_back=30)[:100]
        except Exception:
            return []

    async def analyze(self) -> AgentInsight:
        patterns = await self.run_discovery()
        events: list[NarrativeEvent] = []
        if patterns:
            events.append(NarrativeEvent(
                event_id=str(uuid.uuid4())[:12], source="pattern_discovery",
                content=f"Discovered {len(patterns)} new narrative pattern(s): {', '.join(p.name for p in patterns)}",
                timestamp=datetime.now(timezone.utc),
                category=NarrativeCategory.OTHER,
                sentiment=0.1, magnitude=0.3,
                keywords=["pattern", "discovery", "analysis"],
            ))
        return AgentInsight(agent_name=self.name, events=events, raw_analysis=str(patterns),
                            confidence=0.6 if patterns else 0.0, tokens_used=0, cost_usd=0.0)
