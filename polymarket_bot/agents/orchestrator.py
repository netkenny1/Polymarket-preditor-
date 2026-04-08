"""Council Orchestrator — manages all AI agents and coordinates analysis cycles.

Runs agents concurrently every 15 minutes (fast agents) and on longer cycles for
deep analysis (Predictive History: 2h, Pattern Discovery: 6h).

Outputs: list[NarrativeEvent] ready for NarrativeEngine.ingest_tweet_events()
"""
from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Any

import structlog

from polymarket_bot.agents.base_agent import AgentInsight, BaseAgent
from polymarket_bot.data.models import CrossAssetImpact, NarrativeEvent

logger = structlog.get_logger()


class CouncilOrchestrator:
    """Manages the AI agent council. Runs concurrent analysis cycles and
    feeds results into the NarrativeEngine via ingest_tweet_events().
    """

    def __init__(
        self,
        fast_agents: list[BaseAgent],
        deep_agents: list[BaseAgent],
        narrative_engine: Any,
        db: Any = None,
        council_cycle_minutes: int = 15,
        predictive_history_hours: int = 2,
        pattern_discovery_hours: int = 6,
        max_daily_cost_usd: float = 5.0,
    ) -> None:
        self._fast_agents = fast_agents  # Run every council_cycle_minutes
        self._deep_agents = deep_agents  # Run on longer cycles
        self._engine = narrative_engine
        self._db = db
        self._cycle_minutes = council_cycle_minutes
        self._ph_hours = predictive_history_hours
        self._pd_hours = pattern_discovery_hours
        self._max_daily_cost = max_daily_cost_usd

        self._last_fast_run: datetime | None = None
        self._last_deep_run: datetime | None = None
        self._last_discovery_run: datetime | None = None
        self._insight_history: deque[AgentInsight] = deque(maxlen=500)
        self._last_cross_asset: CrossAssetImpact | None = None
        self._running = False

    # ── Main loop entry-points ───────────────────────────────────────

    async def start_background(self) -> None:
        """Start the council as a background async task."""
        self._running = True
        await asyncio.gather(
            self._fast_loop(),
            self._deep_loop(),
            self._discovery_loop(),
            return_exceptions=True,
        )

    async def run_cycle(self) -> list[NarrativeEvent]:
        """Run one fast-agent cycle synchronously (for use in the main trading loop)."""
        return await self._run_fast_agents()

    async def run_deep_cycle(self) -> list[NarrativeEvent]:
        """Run one deep-analysis cycle (Predictive History agent)."""
        return await self._run_deep_agents()

    # ── Private loops ────────────────────────────────────────────────

    async def _fast_loop(self) -> None:
        while self._running:
            try:
                if self._cost_ok():
                    await self._run_fast_agents()
                else:
                    logger.warning("council_cost_limit_reached_skipping_cycle")
            except Exception as exc:
                logger.error("council_fast_loop_error", error=str(exc))
            await asyncio.sleep(self._cycle_minutes * 60)

    async def _deep_loop(self) -> None:
        await asyncio.sleep(60)  # Stagger start
        while self._running:
            try:
                if self._cost_ok():
                    await self._run_deep_agents()
            except Exception as exc:
                logger.error("council_deep_loop_error", error=str(exc))
            await asyncio.sleep(self._ph_hours * 3600)

    async def _discovery_loop(self) -> None:
        await asyncio.sleep(120)  # Stagger start
        while self._running:
            try:
                if self._cost_ok():
                    await self._run_pattern_discovery()
            except Exception as exc:
                logger.error("council_discovery_loop_error", error=str(exc))
            await asyncio.sleep(self._pd_hours * 3600)

    # ── Agent execution ──────────────────────────────────────────────

    async def _run_fast_agents(self) -> list[NarrativeEvent]:
        if not self._fast_agents:
            return []
        tasks = [agent.analyze() for agent in self._fast_agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_events: list[NarrativeEvent] = []
        total_cost = 0.0
        for agent, result in zip(self._fast_agents, results):
            if isinstance(result, Exception):
                logger.error("agent_cycle_failed", agent=agent.name, error=str(result))
                continue
            insight: AgentInsight = result
            all_events.extend(insight.events)
            total_cost += insight.cost_usd
            self._insight_history.append(insight)
            # Persist to DB
            if self._db and insight.tokens_used > 0:
                try:
                    self._db.save_agent_insight(
                        insight.agent_name, insight.raw_analysis[:3000],
                        len(insight.events), insight.tokens_used, insight.cost_usd,
                    )
                except Exception:
                    pass

        if all_events:
            self._engine.ingest_tweet_events(all_events)
            # Build aggregate cross-asset impact
            self._last_cross_asset = self._aggregate_cross_asset(results)

        self._last_fast_run = datetime.now(timezone.utc)
        logger.info(
            "council_fast_cycle_completed",
            events=len(all_events),
            agents=len(self._fast_agents),
            cost_usd=round(total_cost, 4),
        )
        return all_events

    async def _run_deep_agents(self) -> list[NarrativeEvent]:
        if not self._deep_agents:
            return []
        all_events: list[NarrativeEvent] = []
        for agent in self._deep_agents:
            try:
                insight = await agent.analyze()
                all_events.extend(insight.events)
                self._insight_history.append(insight)
                if self._db and insight.tokens_used > 0:
                    try:
                        self._db.save_agent_insight(
                            insight.agent_name, insight.raw_analysis[:5000],
                            len(insight.events), insight.tokens_used, insight.cost_usd,
                        )
                    except Exception:
                        pass
            except Exception as exc:
                logger.error("deep_agent_failed", agent=agent.name, error=str(exc))

        if all_events:
            self._engine.ingest_tweet_events(all_events)

        self._last_deep_run = datetime.now(timezone.utc)
        logger.info("council_deep_cycle_completed", events=len(all_events))
        return all_events

    async def _run_pattern_discovery(self) -> None:
        from polymarket_bot.agents.pattern_discovery_agent import PatternDiscoveryAgent
        discovery_agents = [a for a in self._fast_agents + self._deep_agents
                            if isinstance(a, PatternDiscoveryAgent)]
        for da in discovery_agents:
            try:
                patterns = await da.run_discovery()
                if patterns:
                    logger.info("patterns_discovered", count=len(patterns),
                                names=[p.name for p in patterns])
                    # Inject discovered patterns into narrative engine if possible
                    if hasattr(self._engine, "_active_narratives"):
                        pass  # Patterns go into HistoricalPatternMatcher separately
            except Exception as exc:
                logger.error("pattern_discovery_failed", error=str(exc))
        self._last_discovery_run = datetime.now(timezone.utc)

    # ── Cross-asset aggregation ───────────────────────────────────────

    def _aggregate_cross_asset(self, results: list) -> CrossAssetImpact | None:
        """Aggregate cross-asset signals from all agent insights."""
        impacts: dict[str, list[float]] = {
            "sp500": [], "gold": [], "oil": [], "crypto": [], "usd": []
        }
        source_events = []
        for r in results:
            if isinstance(r, AgentInsight) and r.cross_asset_impact:
                for key in impacts:
                    val = r.cross_asset_impact.get(key, r.cross_asset_impact.get(f"{key}_direction"))
                    if val is not None:
                        impacts[key].append(float(val))
                if r.events:
                    source_events.append(r.agent_name)

        def avg(lst: list[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        if not any(impacts.values()):
            return None

        return CrossAssetImpact(
            source_event=f"Council cycle: {', '.join(source_events[:4])}",
            sp500_direction=avg(impacts["sp500"]),
            gold_direction=avg(impacts["gold"]),
            oil_direction=avg(impacts["oil"]),
            crypto_direction=avg(impacts["crypto"]),
            usd_direction=avg(impacts["usd"]),
            confidence=0.6,
        )

    def get_cross_asset_summary(self) -> CrossAssetImpact | None:
        """Return the most recent cross-asset impact for use by NarrativeStrategy."""
        return self._last_cross_asset

    def get_daily_cost(self) -> float:
        """Return estimated Claude spend today (from DB)."""
        if self._db:
            try:
                return self._db.get_total_daily_cost()
            except Exception:
                pass
        return 0.0

    def _cost_ok(self) -> bool:
        return self.get_daily_cost() < self._max_daily_cost

    def stop(self) -> None:
        self._running = False
