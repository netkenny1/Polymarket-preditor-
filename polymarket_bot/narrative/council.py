"""Scenario council — parallel narrative simulations that vote on predictions.

Multiple "agents" (scenarios) each interpret current events differently.
They compete: scenarios whose predictions match incoming data gain weight;
those that fail lose weight. The council's weighted vote produces trading signals.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import structlog

from polymarket_bot.data.models import (
    HistoricalPattern,
    Market,
    Narrative,
    NarrativeCategory,
    SimulationScenario,
)
from polymarket_bot.utils.helpers import clamp

logger = structlog.get_logger()

# Scenario templates: each narrative category has different scenario archetypes
_SCENARIO_TEMPLATES: dict[NarrativeCategory, list[dict[str, Any]]] = {
    NarrativeCategory.TRADE_WAR: [
        {"name": "Full Escalation", "direction_mult": -1.5, "magnitude_mult": 1.3},
        {"name": "Negotiated Resolution", "direction_mult": 0.8, "magnitude_mult": 0.8},
        {"name": "Selective Tariffs", "direction_mult": -0.5, "magnitude_mult": 0.6},
        {"name": "Status Quo Bluff", "direction_mult": 0.1, "magnitude_mult": 0.3},
        {"name": "Retaliation Spiral", "direction_mult": -2.0, "magnitude_mult": 1.5},
    ],
    NarrativeCategory.MONETARY_POLICY: [
        {"name": "Aggressive Tightening", "direction_mult": -1.2, "magnitude_mult": 1.2},
        {"name": "Dovish Pivot", "direction_mult": 1.0, "magnitude_mult": 1.0},
        {"name": "Gradual Adjustment", "direction_mult": -0.3, "magnitude_mult": 0.5},
        {"name": "Emergency Cut", "direction_mult": 1.5, "magnitude_mult": 1.5},
        {"name": "Hold Steady", "direction_mult": 0.0, "magnitude_mult": 0.2},
    ],
    NarrativeCategory.GEOPOLITICAL: [
        {"name": "Military Escalation", "direction_mult": -1.5, "magnitude_mult": 1.4},
        {"name": "Diplomatic Resolution", "direction_mult": 0.8, "magnitude_mult": 0.7},
        {"name": "Frozen Conflict", "direction_mult": -0.2, "magnitude_mult": 0.3},
        {"name": "Proxy Conflict", "direction_mult": -0.8, "magnitude_mult": 0.9},
        {"name": "De-escalation", "direction_mult": 0.5, "magnitude_mult": 0.5},
    ],
    NarrativeCategory.CRYPTO_REGULATION: [
        {"name": "Regulatory Crackdown", "direction_mult": -1.3, "magnitude_mult": 1.2},
        {"name": "Clear Framework", "direction_mult": 0.8, "magnitude_mult": 0.8},
        {"name": "Mixed Signals", "direction_mult": -0.2, "magnitude_mult": 0.4},
        {"name": "Pro-Crypto Pivot", "direction_mult": 1.5, "magnitude_mult": 1.3},
        {"name": "Gradual Adoption", "direction_mult": 0.5, "magnitude_mult": 0.5},
    ],
    NarrativeCategory.ELECTION: [
        {"name": "Frontrunner Wins", "direction_mult": 0.5, "magnitude_mult": 0.6},
        {"name": "Upset Victory", "direction_mult": -1.0, "magnitude_mult": 1.4},
        {"name": "Contested Result", "direction_mult": -0.8, "magnitude_mult": 1.2},
        {"name": "Landslide", "direction_mult": 1.0, "magnitude_mult": 1.0},
        {"name": "Status Quo", "direction_mult": 0.1, "magnitude_mult": 0.3},
    ],
    NarrativeCategory.MARKET_CRISIS: [
        {"name": "Systemic Contagion", "direction_mult": -2.0, "magnitude_mult": 1.5},
        {"name": "Contained Event", "direction_mult": -0.3, "magnitude_mult": 0.4},
        {"name": "V-Shaped Recovery", "direction_mult": 0.8, "magnitude_mult": 1.0},
        {"name": "Prolonged Bear", "direction_mult": -1.0, "magnitude_mult": 1.2},
        {"name": "Policy Rescue", "direction_mult": 0.5, "magnitude_mult": 0.8},
    ],
    NarrativeCategory.FISCAL_POLICY: [
        {"name": "Stimulus Boost", "direction_mult": 0.8, "magnitude_mult": 0.9},
        {"name": "Austerity Shock", "direction_mult": -0.8, "magnitude_mult": 0.8},
        {"name": "Government Shutdown", "direction_mult": -0.5, "magnitude_mult": 0.7},
        {"name": "Bipartisan Deal", "direction_mult": 0.4, "magnitude_mult": 0.5},
        {"name": "Debt Crisis", "direction_mult": -1.5, "magnitude_mult": 1.3},
    ],
    NarrativeCategory.OTHER: [
        {"name": "Bullish Continuation", "direction_mult": 0.5, "magnitude_mult": 0.5},
        {"name": "Bearish Reversal", "direction_mult": -0.5, "magnitude_mult": 0.5},
        {"name": "Sideways Drift", "direction_mult": 0.0, "magnitude_mult": 0.2},
        {"name": "Volatility Spike", "direction_mult": 0.0, "magnitude_mult": 1.0},
        {"name": "Trend Break", "direction_mult": -0.3, "magnitude_mult": 0.7},
    ],
}


class ScenarioCouncil:
    """Run multiple narrative scenarios in parallel and vote on predictions.

    Each scenario represents a different "agent" with a different
    perspective. Scenarios that predict well gain weight through
    Bayesian updating; those that fail lose weight.
    """

    def __init__(
        self,
        council_size: int = 5,
        min_agreement: float = 0.6,
        evaluation_interval: int = 5,
    ) -> None:
        self._council_size = council_size
        self._min_agreement = min_agreement
        self._eval_interval = evaluation_interval
        self._scenarios_by_narrative: dict[str, list[SimulationScenario]] = {}
        self._step_count = 0

    def generate_scenarios(
        self,
        narrative: Narrative,
        pattern_matches: list[tuple[HistoricalPattern, float, int]] | None = None,
    ) -> list[SimulationScenario]:
        """Generate competing scenarios for a narrative.

        Uses category-specific templates and historical pattern matches
        to create diverse scenarios with different assumptions.
        """
        nid = narrative.narrative_id

        # Check if we already have scenarios for this narrative
        if nid in self._scenarios_by_narrative:
            return self._scenarios_by_narrative[nid]

        templates = _SCENARIO_TEMPLATES.get(
            narrative.category,
            _SCENARIO_TEMPLATES[NarrativeCategory.OTHER],
        )

        # Use up to council_size templates
        selected = templates[: self._council_size]
        scenarios: list[SimulationScenario] = []

        for template in selected:
            base_direction = narrative.predicted_direction
            direction = clamp(
                base_direction * template["direction_mult"], -1.0, 1.0
            )
            magnitude = clamp(
                abs(base_direction) * 0.1 * template["magnitude_mult"], 0.0, 0.2
            )

            # If we have a historical pattern match, bias toward it
            initial_weight = 1.0
            if pattern_matches:
                best_pattern, similarity, phase_idx = pattern_matches[0]
                # Historical repeat scenario gets extra weight
                if "escalation" in template["name"].lower() or "continuation" in template["name"].lower():
                    initial_weight += similarity * 0.5

            scenario = SimulationScenario(
                scenario_id=str(uuid.uuid4())[:12],
                name=template["name"],
                narrative_id=nid,
                assumptions={
                    "direction_mult": template["direction_mult"],
                    "magnitude_mult": template["magnitude_mult"],
                    "category": narrative.category.value,
                },
                predicted_direction=direction,
                predicted_magnitude=magnitude,
                weight=initial_weight,
            )
            scenarios.append(scenario)

        # Normalize weights
        total_w = sum(s.weight for s in scenarios)
        if total_w > 0:
            for s in scenarios:
                s.weight /= total_w

        self._scenarios_by_narrative[nid] = scenarios

        logger.info(
            "scenarios_generated",
            narrative_id=nid,
            count=len(scenarios),
        )

        return scenarios

    def evaluate_scenarios(
        self,
        narrative: Narrative,
        markets: list[Market],
    ) -> None:
        """Evaluate scenarios against actual market data using Bayesian updating.

        Scenarios that predicted well gain weight; those that predicted
        poorly lose weight.
        """
        self._step_count += 1

        nid = narrative.narrative_id
        scenarios = self._scenarios_by_narrative.get(nid, [])
        if not scenarios:
            return

        # Only evaluate periodically
        if self._step_count % self._eval_interval != 0:
            return

        # Compute how well each scenario predicted actual events
        recent_events = narrative.events[-5:]  # Last 5 events
        if not recent_events:
            return

        avg_recent_sentiment = sum(e.sentiment for e in recent_events) / len(
            recent_events
        )

        for scenario in scenarios:
            # How well did this scenario's direction match reality?
            predicted = scenario.predicted_direction
            actual = avg_recent_sentiment

            # Accuracy: 1.0 if perfect match, 0.0 if opposite
            if predicted == 0 and actual == 0:
                accuracy = 0.5
            else:
                # Cosine-like similarity for direction
                direction_match = 1.0 - abs(predicted - actual) / 2.0
                accuracy = clamp(direction_match, 0.0, 1.0)

            scenario.accuracy_history.append(accuracy)
            # Keep bounded
            if len(scenario.accuracy_history) > 20:
                scenario.accuracy_history = scenario.accuracy_history[-20:]

            # Bayesian weight update: multiply by likelihood
            likelihood = 0.3 + 0.7 * accuracy  # Floor at 0.3 to avoid zeroing out
            scenario.weight *= likelihood

        # Renormalize weights
        total_w = sum(s.weight for s in scenarios)
        if total_w > 0:
            for s in scenarios:
                s.weight /= total_w

        logger.info(
            "scenarios_evaluated",
            narrative_id=nid,
            weights=[round(s.weight, 3) for s in scenarios],
        )

    def vote(
        self, narrative_id: str
    ) -> tuple[float, float, float]:
        """Have scenarios vote on predicted direction.

        Returns (weighted_direction, weighted_magnitude, agreement_ratio).

        agreement_ratio: fraction of total weight that agrees with the
        majority direction. If < min_agreement, the council is split
        and confidence should be low.
        """
        scenarios = self._scenarios_by_narrative.get(narrative_id, [])
        if not scenarios:
            return 0.0, 0.0, 0.0

        # Weighted direction and magnitude
        total_weight = sum(s.weight for s in scenarios)
        if total_weight == 0:
            return 0.0, 0.0, 0.0

        weighted_dir = sum(
            s.predicted_direction * s.weight for s in scenarios
        ) / total_weight
        weighted_mag = sum(
            s.predicted_magnitude * s.weight for s in scenarios
        ) / total_weight

        # Compute agreement: what fraction of weight agrees with majority?
        bullish_weight = sum(
            s.weight for s in scenarios if s.predicted_direction > 0
        )
        bearish_weight = sum(
            s.weight for s in scenarios if s.predicted_direction < 0
        )
        neutral_weight = sum(
            s.weight for s in scenarios if s.predicted_direction == 0
        )

        majority_weight = max(bullish_weight, bearish_weight, neutral_weight)
        agreement = majority_weight / total_weight if total_weight > 0 else 0.0

        return (
            clamp(weighted_dir, -1.0, 1.0),
            clamp(weighted_mag, 0.0, 0.2),
            clamp(agreement, 0.0, 1.0),
        )

    def prune_and_refresh(
        self, narrative: Narrative
    ) -> list[SimulationScenario]:
        """Remove worst-performing scenarios and generate replacements.

        Keeps at least 2 scenarios. Removes those with weight < 10%
        of the best scenario.
        """
        nid = narrative.narrative_id
        scenarios = self._scenarios_by_narrative.get(nid, [])
        if len(scenarios) <= 2:
            return scenarios

        max_weight = max(s.weight for s in scenarios)
        threshold = max_weight * 0.1

        # Keep scenarios above threshold (minimum 2)
        surviving = [s for s in scenarios if s.weight >= threshold]
        if len(surviving) < 2:
            surviving = sorted(scenarios, key=lambda s: s.weight, reverse=True)[:2]

        # Generate replacements if needed
        while len(surviving) < self._council_size:
            templates = _SCENARIO_TEMPLATES.get(
                narrative.category,
                _SCENARIO_TEMPLATES[NarrativeCategory.OTHER],
            )
            # Pick a template not already represented
            existing_names = {s.name for s in surviving}
            new_template = None
            for t in templates:
                if t["name"] not in existing_names:
                    new_template = t
                    break

            if new_template is None:
                break

            new_scenario = SimulationScenario(
                scenario_id=str(uuid.uuid4())[:12],
                name=new_template["name"],
                narrative_id=nid,
                assumptions={
                    "direction_mult": new_template["direction_mult"],
                    "magnitude_mult": new_template["magnitude_mult"],
                    "category": narrative.category.value,
                },
                predicted_direction=clamp(
                    narrative.predicted_direction * new_template["direction_mult"],
                    -1.0,
                    1.0,
                ),
                predicted_magnitude=clamp(
                    abs(narrative.predicted_direction)
                    * 0.1
                    * new_template["magnitude_mult"],
                    0.0,
                    0.2,
                ),
                weight=1.0 / self._council_size,  # Start with equal share
            )
            surviving.append(new_scenario)

        # Renormalize
        total_w = sum(s.weight for s in surviving)
        if total_w > 0:
            for s in surviving:
                s.weight /= total_w

        self._scenarios_by_narrative[nid] = surviving
        return surviving

    def get_scenarios(self, narrative_id: str) -> list[SimulationScenario]:
        """Get active scenarios for a narrative."""
        return self._scenarios_by_narrative.get(narrative_id, [])

    def clear_narrative(self, narrative_id: str) -> None:
        """Remove all scenarios for a narrative."""
        self._scenarios_by_narrative.pop(narrative_id, None)
