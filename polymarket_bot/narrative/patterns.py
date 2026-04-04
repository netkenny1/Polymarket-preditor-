"""Historical pattern matching — Professor Jiang's predictive history approach.

Matches current narratives to historical precedents and uses how those
episodes played out to predict current market outcomes.
"""

from __future__ import annotations

from typing import Any

import structlog

from polymarket_bot.data.models import (
    HistoricalPattern,
    HistoricalPhase,
    Market,
    Narrative,
    NarrativeCategory,
    Side,
    Signal,
)
from polymarket_bot.narrative.pattern_library import get_builtin_patterns
from polymarket_bot.utils.helpers import clamp

logger = structlog.get_logger()


class HistoricalPatternMatcher:
    """Match current narratives to historical precedents for prediction."""

    def __init__(self, similarity_threshold: float = 0.4) -> None:
        self._threshold = similarity_threshold
        self._patterns: list[HistoricalPattern] = get_builtin_patterns()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_matching_patterns(
        self, narrative: Narrative
    ) -> list[tuple[HistoricalPattern, float, int]]:
        """Find historical patterns that match a narrative.

        Returns list of (pattern, similarity_score, current_phase_index) tuples,
        sorted by similarity descending.

        Similarity scoring:
        1. Category match (same category = 0.4 base score)
        2. Keyword overlap (Jaccard similarity, up to 0.35)
        3. Direction alignment (0.15 if narrative direction matches pattern)
        4. Event density bonus (0.1 if narrative has many events)
        """
        matches: list[tuple[HistoricalPattern, float, int]] = []
        for pattern in self._patterns:
            sim = self._compute_similarity(narrative, pattern)
            if sim >= self._threshold:
                phase_idx = self._identify_current_phase(narrative, pattern)
                matches.append((pattern, sim, phase_idx))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:5]  # Top 5 matches

    def predict_next_phase(
        self, pattern: HistoricalPattern, current_phase_idx: int
    ) -> tuple[float, float]:
        """Predict direction and magnitude from the NEXT phase of a historical pattern.

        If we're in phase N, look at phase N+1 to predict what comes next.
        If we're in the last phase, use the pattern's overall outcome.

        Returns (predicted_direction: -1 to 1, predicted_magnitude: 0 to 1)
        """
        if not pattern.phases:
            return pattern.outcome_direction, pattern.outcome_magnitude

        next_idx = current_phase_idx + 1
        if next_idx < len(pattern.phases):
            next_phase = pattern.phases[next_idx]
            # Average market impact across categories
            impacts = list(next_phase.market_impact.values())
            if impacts:
                direction = sum(impacts) / len(impacts)
            else:
                direction = pattern.outcome_direction
            magnitude = min(abs(direction) * 0.5, 0.15)
            return clamp(direction, -1.0, 1.0), clamp(magnitude, 0.0, 0.2)
        else:
            # Last phase — use overall outcome with decay
            return pattern.outcome_direction * 0.5, pattern.outcome_magnitude * 0.3

    @property
    def pattern_count(self) -> int:
        """Number of loaded historical patterns."""
        return len(self._patterns)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_similarity(
        self, narrative: Narrative, pattern: HistoricalPattern
    ) -> float:
        """Compute similarity between narrative and pattern (0 to 1)."""
        score = 0.0

        # 1. Category match (0.4)
        if narrative.category == pattern.category:
            score += 0.4

        # 2. Keyword overlap — Jaccard similarity (up to 0.35)
        narrative_kw: set[str] = set()
        for event in narrative.events:
            narrative_kw.update(w.lower() for w in event.keywords)
        # Also extract from title/thesis
        narrative_kw.update(
            w.lower() for w in narrative.title.split() if len(w) > 3
        )

        pattern_kw = set(w.lower() for w in pattern.trigger_keywords)

        if narrative_kw and pattern_kw:
            intersection = narrative_kw & pattern_kw
            union = narrative_kw | pattern_kw
            jaccard = len(intersection) / len(union) if union else 0
            score += jaccard * 0.35

        # 3. Direction alignment (0.15)
        if narrative.predicted_direction != 0 and pattern.outcome_direction != 0:
            same_sign = (narrative.predicted_direction > 0) == (
                pattern.outcome_direction > 0
            )
            score += 0.15 if same_sign else 0.0

        # 4. Event density bonus (0.1)
        if narrative.event_count >= 5:
            score += 0.1
        elif narrative.event_count >= 3:
            score += 0.05

        return min(score, 1.0)

    def _identify_current_phase(
        self, narrative: Narrative, pattern: HistoricalPattern
    ) -> int:
        """Determine which phase of the historical pattern we're currently in.

        Matches narrative event keywords against each phase's keywords.
        Returns the index of the best-matching phase.
        """
        if not pattern.phases:
            return 0

        narrative_kw: set[str] = set()
        for event in narrative.events:
            narrative_kw.update(w.lower() for w in event.keywords)

        best_idx = 0
        best_score = -1.0

        for i, phase in enumerate(pattern.phases):
            phase_kw = set(w.lower() for w in phase.keywords)
            if not phase_kw:
                continue
            overlap = len(narrative_kw & phase_kw)
            # Weight by recency — later phases get slight bonus if ambiguous
            score = overlap + i * 0.1
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx
