"""Narrative analysis strategy — the integration point.

Orchestrates the full narrative pipeline:
1. Ingest events from context (tweets, economic data, market moves)
2. Build/update narratives via NarrativeEngine
3. Match narratives to historical patterns (predictive history)
4. Run parallel scenarios via ScenarioCouncil
5. Convert council votes into Signal objects
"""

from __future__ import annotations

from typing import Any

import structlog

from polymarket_bot.data.models import (
    Market,
    OrderBook,
    Side,
    Signal,
)
from polymarket_bot.narrative.council import ScenarioCouncil
from polymarket_bot.narrative.engine import NarrativeEngine
from polymarket_bot.narrative.patterns import HistoricalPatternMatcher
from polymarket_bot.strategies.base import BaseStrategy
from polymarket_bot.utils.helpers import clamp

logger = structlog.get_logger()


class NarrativeStrategy(BaseStrategy):
    """Trading strategy based on narrative analysis and historical pattern matching.

    Excels at medium-term predictions (hours to days) where a coherent
    narrative drives market movement. Uses Professor Jiang's "predictive
    history" approach: match current events to historical parallels,
    run competing scenarios, and let them vote on predictions.
    """

    name = "narrative_analysis"

    def __init__(
        self,
        economic_client: Any = None,
        news_reactor: Any = None,
        max_active_narratives: int = 10,
        min_events: int = 3,
        decay_hours: float = 48.0,
        council_size: int = 5,
        council_min_agreement: float = 0.6,
        pattern_threshold: float = 0.4,
        max_signal_confidence: float = 0.75,
        eval_interval: int = 5,
        min_edge: float = 0.03,
    ) -> None:
        self._economic_client = economic_client
        self._news_reactor = news_reactor
        self._min_edge = min_edge
        self._max_confidence = max_signal_confidence
        self._council_min_agreement = council_min_agreement

        self._engine = NarrativeEngine(
            max_active=max_active_narratives,
            min_events=min_events,
            decay_hours=decay_hours,
        )
        self._pattern_matcher = HistoricalPatternMatcher(
            similarity_threshold=pattern_threshold,
        )
        self._council = ScenarioCouncil(
            council_size=council_size,
            min_agreement=council_min_agreement,
            evaluation_interval=eval_interval,
        )

        self._step_count = 0
        self._pattern_cache: dict[str, list] = {}

    def generate_signals(
        self,
        markets: list[Market],
        order_books: dict[str, OrderBook],
        context: dict[str, Any],
    ) -> list[Signal]:
        """Generate trading signals from narrative analysis.

        Data flow:
        1. Extract events from context
        2. Feed into NarrativeEngine
        3. For each narrative, find historical patterns
        4. Run scenario council
        5. Convert votes to Signals
        """
        self._step_count += 1
        signals: list[Signal] = []
        tradeable = self.filter_tradeable_markets(markets)

        if not tradeable:
            return signals

        # 1. Ingest events from context
        self._ingest_from_context(context, tradeable)

        # 2. Update narratives
        narratives = self._engine.update_narratives()
        if not narratives:
            return signals

        # 3. Map narratives to markets
        market_narratives = self._engine.map_narratives_to_markets(
            narratives, tradeable
        )

        # 4. For each narrative, run pattern matching + scenarios
        for narrative in narratives:
            nid = narrative.narrative_id

            # Find historical parallels (cache to avoid recomputing)
            if nid not in self._pattern_cache or self._step_count % 10 == 0:
                self._pattern_cache[nid] = (
                    self._pattern_matcher.find_matching_patterns(narrative)
                )
            pattern_matches = self._pattern_cache[nid]

            # Link best pattern to narrative
            if pattern_matches:
                best_pattern, sim, phase = pattern_matches[0]
                narrative.historical_pattern_id = best_pattern.pattern_id

                # Get prediction from historical pattern
                pred_dir, pred_mag = self._pattern_matcher.predict_next_phase(
                    best_pattern, phase
                )
                # Blend pattern prediction with narrative direction
                narrative.predicted_direction = (
                    narrative.predicted_direction * 0.6 + pred_dir * 0.4
                )

            # Generate/update scenarios
            scenarios = self._council.generate_scenarios(
                narrative, pattern_matches
            )

            # Evaluate scenario accuracy
            self._council.evaluate_scenarios(narrative, tradeable)

            # Periodically prune and refresh
            if self._step_count % 20 == 0:
                self._council.prune_and_refresh(narrative)

        # 5. Generate signals for each market
        for market in tradeable:
            relevant = market_narratives.get(market.condition_id, [])
            if not relevant:
                continue

            signal = self._generate_market_signal(market, relevant)
            if signal is not None:
                signals.append(signal)

        logger.info(
            "narrative_signals",
            narratives=len(narratives),
            signals=len(signals),
        )

        return signals

    def _ingest_from_context(
        self, context: dict[str, Any], markets: list[Market]
    ) -> None:
        """Extract and ingest events from the strategy context dict."""
        # News events from NewsReactor
        news_events = context.get("news_events", [])
        if news_events:
            self._engine.ingest_tweet_events(news_events)

        # Economic indicators
        economic_data = context.get("economic_indicators", [])
        if not economic_data and self._economic_client is not None:
            try:
                economic_data = self._economic_client.get_all_indicators()
            except Exception:
                pass
        if economic_data:
            self._engine.ingest_economic_data(economic_data)

        # Market moves
        self._engine.ingest_market_moves(markets, context)

    def _generate_market_signal(
        self,
        market: Market,
        relevant_narratives: list[tuple[Any, float]],
    ) -> Signal | None:
        """Convert narrative analysis into a signal for one market.

        Aggregates across all relevant narratives, weighted by relevance.
        Uses each narrative's scenario council vote.
        """
        total_weight = 0.0
        weighted_direction = 0.0
        weighted_magnitude = 0.0
        max_confidence = 0.0
        metadata_narratives = []

        for narrative, relevance in relevant_narratives:
            # Get council vote for this narrative
            direction, magnitude, agreement = self._council.vote(
                narrative.narrative_id
            )

            # Fall back to raw narrative if no council vote
            if agreement == 0:
                direction = narrative.predicted_direction
                magnitude = 0.05
                agreement = 0.5

            w = relevance * agreement
            total_weight += w
            weighted_direction += direction * w
            weighted_magnitude += magnitude * w
            max_confidence = max(max_confidence, narrative.confidence * agreement)

            metadata_narratives.append({
                "narrative": narrative.title,
                "category": narrative.category.value,
                "pattern": narrative.historical_pattern_id,
                "direction": round(direction, 3),
                "agreement": round(agreement, 3),
                "relevance": round(relevance, 3),
                "events": narrative.event_count,
            })

        if total_weight < 0.05:
            return None

        final_direction = weighted_direction / total_weight
        final_magnitude = weighted_magnitude / total_weight

        # Convert to fair value and edge
        yes_price = market.yes_price

        if final_direction > 0:
            # Narrative predicts YES more likely
            fair_value = clamp(yes_price + final_magnitude, 0.05, 0.95)
            edge = fair_value - yes_price
            token = next(
                (t for t in market.tokens if t.outcome == "Yes"), None
            )
            market_price = yes_price
        else:
            # Narrative predicts NO more likely
            no_price = 1.0 - yes_price
            fair_value = clamp(no_price + final_magnitude, 0.05, 0.95)
            edge = fair_value - no_price
            token = next(
                (t for t in market.tokens if t.outcome == "No"), None
            )
            market_price = no_price

        if token is None or edge < self._min_edge:
            return None

        confidence = clamp(
            max_confidence * (0.5 + 0.5 * min(1.0, total_weight)),
            0.15,
            self._max_confidence,
        )

        return Signal(
            market_condition_id=market.condition_id,
            token_id=token.token_id,
            side=Side.BUY,
            outcome=token.outcome,
            estimated_fair_value=fair_value,
            market_price=market_price,
            edge=edge,
            confidence=confidence,
            strategy=self.name,
            metadata={
                "narratives": metadata_narratives,
                "final_direction": round(final_direction, 3),
                "final_magnitude": round(final_magnitude, 4),
                "num_narratives": len(relevant_narratives),
                "step": self._step_count,
            },
        )
