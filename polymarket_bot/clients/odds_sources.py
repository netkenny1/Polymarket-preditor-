"""External odds and data sources for cross-market comparison.

Fetches odds from external prediction markets, sportsbooks, and polling
aggregators to find arbitrage and statistical edges vs Polymarket prices.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger()


@dataclass
class ExternalOdds:
    """Odds from an external source."""

    source: str
    event_name: str
    outcome: str
    implied_probability: float
    raw_odds: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PollData:
    """Polling data for political markets."""

    pollster: str
    candidate: str
    pct: float
    sample_size: int = 0
    date: Optional[datetime] = None
    grade: str = ""  # Pollster rating


@dataclass
class EloRating:
    """ELO rating for sports teams."""

    team: str
    rating: float
    sport: str = ""


class OddsAggregator:
    """Aggregates odds from multiple external sources.

    Used by the arbitrage and statistical strategies to find mispricings
    between Polymarket and other prediction/betting platforms.
    """

    def __init__(self) -> None:
        self._client = httpx.Client(timeout=30.0)

    def close(self) -> None:
        self._client.close()

    # ── Polling Data (Politics) ──────────────────────────────────

    def get_polling_average(self, candidates: list[str], polls: list[PollData]) -> dict[str, float]:
        """Compute weighted polling average for candidates.

        Uses a recency-weighted average where newer polls get more weight.
        Pollster grade is used as a quality weight.
        """
        if not polls:
            return {c: 1.0 / len(candidates) for c in candidates}

        grade_weights = {"A+": 3.0, "A": 2.5, "A-": 2.2, "B+": 2.0, "B": 1.5, "B-": 1.2, "C+": 1.0, "C": 0.8, "D": 0.5}

        candidate_scores: dict[str, list[tuple[float, float]]] = {c: [] for c in candidates}

        now = datetime.utcnow()
        for poll in polls:
            # Recency weight: half-life of 14 days
            age_days = (now - poll.date).days if poll.date else 30
            recency_weight = 0.5 ** (age_days / 14)

            # Quality weight
            quality_weight = grade_weights.get(poll.grade, 1.0)

            # Sample size weight (log scale)
            import math
            size_weight = math.log(max(poll.sample_size, 100)) / math.log(1000)

            total_weight = recency_weight * quality_weight * size_weight

            if poll.candidate in candidate_scores:
                candidate_scores[poll.candidate].append((poll.pct, total_weight))

        result = {}
        for candidate, scores in candidate_scores.items():
            if scores:
                weighted_sum = sum(pct * w for pct, w in scores)
                weight_sum = sum(w for _, w in scores)
                result[candidate] = weighted_sum / weight_sum if weight_sum > 0 else 0.5
            else:
                result[candidate] = 0.0

        # Normalize to probabilities
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}

        return result

    # ── ELO Ratings (Sports) ─────────────────────────────────────

    def elo_win_probability(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected win probability from ELO ratings.

        Uses the standard ELO expected score formula:
        E(A) = 1 / (1 + 10^((Rb - Ra) / 400))
        """
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def get_elo_fair_value(self, team_a: EloRating, team_b: EloRating, home_advantage: float = 50.0) -> dict[str, float]:
        """Get fair value probabilities from ELO ratings.

        Includes home court/field advantage adjustment.
        """
        adjusted_a = team_a.rating + home_advantage
        prob_a = self.elo_win_probability(adjusted_a, team_b.rating)
        return {
            team_a.team: prob_a,
            team_b.team: 1.0 - prob_a,
        }

    # ── Complement Arbitrage ─────────────────────────────────────

    def find_complement_arb(
        self,
        outcome_prices: dict[str, float],
        tolerance: float = 0.03,
    ) -> Optional[dict[str, Any]]:
        """Detect complement arbitrage in multi-outcome markets.

        If the sum of all outcome prices is significantly != 1.0,
        there is an arbitrage opportunity.

        Returns:
            Dict with arb details if found, None otherwise.
        """
        if not outcome_prices:
            return None

        total = sum(outcome_prices.values())

        if total < 1.0 - tolerance:
            # Prices sum to less than 1: buy all outcomes
            profit_pct = 1.0 - total
            return {
                "type": "underpriced_complement",
                "action": "buy_all",
                "total_cost": total,
                "guaranteed_payout": 1.0,
                "profit_pct": profit_pct,
                "outcomes": outcome_prices,
            }
        elif total > 1.0 + tolerance:
            # Prices sum to more than 1: sell all outcomes
            profit_pct = total - 1.0
            return {
                "type": "overpriced_complement",
                "action": "sell_all",
                "total_revenue": total,
                "max_loss": 1.0,
                "profit_pct": profit_pct,
                "outcomes": outcome_prices,
            }

        return None

    # ── Cross-Source Comparison ───────────────────────────────────

    def find_cross_market_edge(
        self,
        polymarket_price: float,
        external_odds: list[ExternalOdds],
        min_edge: float = 0.03,
    ) -> Optional[dict[str, Any]]:
        """Compare Polymarket price to external sources to find edges.

        If the average external implied probability differs significantly
        from the Polymarket price, there may be an edge.
        """
        if not external_odds:
            return None

        external_probs = [o.implied_probability for o in external_odds]
        avg_external = sum(external_probs) / len(external_probs)

        edge = avg_external - polymarket_price

        if abs(edge) >= min_edge:
            return {
                "polymarket_price": polymarket_price,
                "external_avg": avg_external,
                "edge": edge,
                "direction": "BUY" if edge > 0 else "SELL",
                "sources": [
                    {"source": o.source, "prob": o.implied_probability}
                    for o in external_odds
                ],
                "confidence": min(1.0, len(external_odds) / 5),
            }

        return None
