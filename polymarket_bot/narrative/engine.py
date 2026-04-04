"""Narrative engine — builds and maintains coherent narratives from event streams.

Ingests events from tweets, economic data, and market moves. Clusters related
events into narratives, decays old ones, and maps narratives to affected markets.
"""

from __future__ import annotations

import re
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import structlog

from polymarket_bot.data.models import (
    Market,
    MarketCategory,
    Narrative,
    NarrativeCategory,
    NarrativeEvent,
)

logger = structlog.get_logger()

# Keyword banks for theme classification
_CATEGORY_KEYWORDS: dict[NarrativeCategory, list[str]] = {
    NarrativeCategory.TRADE_WAR: [
        "tariff", "trade war", "import", "export", "duties", "customs",
        "sanctions", "trade deficit", "trade surplus", "protectionism",
        "retaliatory", "dumping", "quota", "embargo",
    ],
    NarrativeCategory.MONETARY_POLICY: [
        "fed", "federal reserve", "interest rate", "rate cut", "rate hike",
        "inflation", "cpi", "ppi", "fomc", "powell", "monetary",
        "quantitative", "tightening", "easing", "dovish", "hawkish",
    ],
    NarrativeCategory.GEOPOLITICAL: [
        "war", "military", "troops", "conflict", "invasion", "nato",
        "un", "security council", "missile", "nuclear", "diplomacy",
        "ceasefire", "peace talks", "tensions",
    ],
    NarrativeCategory.CRYPTO_REGULATION: [
        "crypto", "bitcoin", "sec", "regulation", "ban", "cbdc",
        "stablecoin", "defi", "exchange", "mining", "etf",
        "digital asset", "blockchain",
    ],
    NarrativeCategory.FISCAL_POLICY: [
        "spending", "budget", "deficit", "debt ceiling", "stimulus",
        "tax", "infrastructure", "bailout", "appropriations",
        "government shutdown", "fiscal",
    ],
    NarrativeCategory.ELECTION: [
        "election", "vote", "poll", "candidate", "campaign",
        "primary", "debate", "ballot", "swing state", "electoral",
        "midterm", "inauguration",
    ],
    NarrativeCategory.MARKET_CRISIS: [
        "crash", "crisis", "panic", "bank failure", "contagion",
        "liquidity", "margin call", "selloff", "black swan",
        "circuit breaker", "flash crash", "collapse",
    ],
}

# Map NarrativeCategory to MarketCategory for relevance matching
_NARRATIVE_TO_MARKET_CATEGORY: dict[NarrativeCategory, list[MarketCategory]] = {
    NarrativeCategory.TRADE_WAR: [MarketCategory.POLITICS, MarketCategory.OTHER],
    NarrativeCategory.MONETARY_POLICY: [MarketCategory.OTHER, MarketCategory.CRYPTO],
    NarrativeCategory.GEOPOLITICAL: [MarketCategory.POLITICS],
    NarrativeCategory.CRYPTO_REGULATION: [MarketCategory.CRYPTO],
    NarrativeCategory.FISCAL_POLICY: [MarketCategory.POLITICS, MarketCategory.OTHER],
    NarrativeCategory.ELECTION: [MarketCategory.POLITICS],
    NarrativeCategory.MARKET_CRISIS: [MarketCategory.CRYPTO, MarketCategory.OTHER],
    NarrativeCategory.OTHER: [],
}


class NarrativeEngine:
    """Builds and maintains narratives from heterogeneous event streams.

    Data flow:
    1. Ingest events from tweets, economic data, market moves
    2. Classify events by category (trade_war, monetary_policy, etc.)
    3. Cluster related events into narratives
    4. Score narrative strength based on event count, recency, consistency
    5. Map narratives to affected Polymarket markets
    """

    def __init__(
        self,
        max_active: int = 10,
        min_events: int = 3,
        decay_hours: float = 48.0,
    ) -> None:
        self._max_active = max_active
        self._min_events = min_events
        self._decay_hours = decay_hours
        self._active_narratives: dict[str, Narrative] = {}
        self._event_buffer: list[NarrativeEvent] = []
        self._market_keyword_cache: dict[str, list[str]] = {}

    def ingest_tweet_events(self, events: list[Any]) -> None:
        """Convert NewsReactor events into NarrativeEvents and buffer them."""
        for event in events:
            # Accept both NewsEvent objects and dicts
            if hasattr(event, "content"):
                content = event.content
                sentiment = getattr(event, "sentiment", 0.0)
                keywords = getattr(event, "keywords", [])
                source = getattr(event, "source", "twitter")
                ts = getattr(event, "timestamp", datetime.now(timezone.utc))
            elif isinstance(event, dict):
                content = event.get("text", event.get("content", ""))
                sentiment = event.get("sentiment", 0.0)
                keywords = event.get("keywords", [])
                source = event.get("source", "twitter")
                ts = event.get("timestamp", datetime.now(timezone.utc))
            else:
                continue

            if not content:
                continue

            category = self._classify_category(content, keywords)
            magnitude = min(abs(sentiment) * 0.8 + 0.1, 1.0)

            # Extract keywords if not provided
            if not keywords:
                keywords = self._extract_keywords(content)

            ne = NarrativeEvent(
                event_id=str(uuid.uuid4())[:12],
                source=source,
                content=content,
                timestamp=ts,
                category=category,
                sentiment=sentiment,
                magnitude=magnitude,
                keywords=keywords,
            )
            self._event_buffer.append(ne)

    def ingest_economic_data(self, indicators: list[dict]) -> None:
        """Convert economic indicators into NarrativeEvents."""
        for ind in indicators:
            name = ind.get("name", "")
            value = ind.get("value", 0)
            change_pct = ind.get("change_pct", 0)

            # Only create events for significant changes (>2%)
            if abs(change_pct) < 2.0:
                continue

            direction = "improving" if change_pct > 0 else "worsening"
            content = f"{name} {direction}: {value} ({change_pct:+.1f}%)"

            category = self._classify_economic_indicator(name)
            sentiment = 0.3 if change_pct > 0 else -0.3
            # Trade balance worsening is bearish
            if "trade_balance" in name and change_pct < 0:
                sentiment = -0.4
            # Tariff increases are trade war signals
            if "tariff" in name and change_pct > 0:
                sentiment = -0.3
                category = NarrativeCategory.TRADE_WAR

            ne = NarrativeEvent(
                event_id=str(uuid.uuid4())[:12],
                source="economic_data",
                content=content,
                timestamp=datetime.now(timezone.utc),
                category=category,
                sentiment=sentiment,
                magnitude=min(abs(change_pct) / 10.0, 1.0),
                keywords=self._extract_keywords(content) + [name],
                metadata=ind,
            )
            self._event_buffer.append(ne)

    def ingest_market_moves(
        self, markets: list[Market], context: dict[str, Any]
    ) -> None:
        """Detect significant market moves and create events from them."""
        for market in markets:
            key = f"price_history_{market.condition_id}"
            history = context.get(key, [])
            if len(history) < 5:
                continue

            # Detect large recent move
            recent_move = history[-1] - history[-5]
            if abs(recent_move) < 0.08:
                continue

            direction = "surging" if recent_move > 0 else "dropping"
            content = f"Market {direction}: {market.question[:80]} ({recent_move:+.1%})"
            category = self._market_category_to_narrative(market.category)

            ne = NarrativeEvent(
                event_id=str(uuid.uuid4())[:12],
                source="market_move",
                content=content,
                timestamp=datetime.now(timezone.utc),
                category=category,
                sentiment=0.5 if recent_move > 0 else -0.5,
                magnitude=min(abs(recent_move) * 3, 1.0),
                keywords=self._extract_keywords(market.question),
            )
            self._event_buffer.append(ne)

    def update_narratives(self) -> list[Narrative]:
        """Process buffered events, update existing or create new narratives.

        Algorithm:
        1. For each buffered event, find matching active narrative
        2. If match found, append event and update narrative
        3. If no match, group unmatched events by category
        4. If enough events in a category, create new narrative
        5. Decay old narratives
        6. Prune narratives below threshold

        Returns list of active narratives sorted by strength.
        """
        if not self._event_buffer:
            return self.get_active_narratives()

        unmatched: list[NarrativeEvent] = []

        # 1-2. Try to match events to existing narratives
        for event in self._event_buffer:
            matched = False
            for narrative in self._active_narratives.values():
                if self._event_fits_narrative(event, narrative):
                    narrative.events.append(event)
                    self._update_narrative_state(narrative)
                    matched = True
                    break
            if not matched:
                unmatched.append(event)

        # 3-4. Group unmatched events by category, create new narratives
        by_category: dict[NarrativeCategory, list[NarrativeEvent]] = defaultdict(list)
        for event in unmatched:
            by_category[event.category].append(event)

        for category, events in by_category.items():
            if len(events) >= self._min_events:
                narrative = self._create_narrative(events, category)
                self._active_narratives[narrative.narrative_id] = narrative

        # Clear buffer
        self._event_buffer.clear()

        # 5. Decay old narratives
        self._decay_narratives()

        # 6. Prune weak narratives
        self._prune_narratives()

        # Enforce max active limit
        if len(self._active_narratives) > self._max_active:
            sorted_narratives = sorted(
                self._active_narratives.values(),
                key=lambda n: n.strength,
                reverse=True,
            )
            keep_ids = {n.narrative_id for n in sorted_narratives[: self._max_active]}
            self._active_narratives = {
                nid: n
                for nid, n in self._active_narratives.items()
                if nid in keep_ids
            }

        logger.info(
            "narratives_updated",
            active=len(self._active_narratives),
            buffered_events=0,
        )

        return self.get_active_narratives()

    def map_narratives_to_markets(
        self,
        narratives: list[Narrative],
        markets: list[Market],
    ) -> dict[str, list[tuple[Narrative, float]]]:
        """Map narratives to markets with relevance scores.

        Returns {market_condition_id: [(narrative, relevance_score), ...]}
        """
        result: dict[str, list[tuple[Narrative, float]]] = defaultdict(list)

        for narrative in narratives:
            for market in markets:
                relevance = self._compute_market_relevance(narrative, market)
                if relevance > 0.15:
                    result[market.condition_id].append((narrative, relevance))

        # Sort each market's narratives by relevance
        for mid in result:
            result[mid].sort(key=lambda x: x[1], reverse=True)
            result[mid] = result[mid][:5]  # Top 5 per market

        return dict(result)

    def get_active_narratives(self) -> list[Narrative]:
        """Return active narratives sorted by strength."""
        active = [
            n for n in self._active_narratives.values()
            if n.is_active and n.strength > 0.1
        ]
        active.sort(key=lambda n: n.strength, reverse=True)
        return active

    # ── Private helpers ─────────────────────────────────────────

    def _classify_category(
        self, content: str, keywords: list[str]
    ) -> NarrativeCategory:
        """Classify text into a narrative category using keyword matching."""
        text = (content + " " + " ".join(keywords)).lower()
        best_cat = NarrativeCategory.OTHER
        best_score = 0

        for category, kw_list in _CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in kw_list if kw in text)
            if score > best_score:
                best_score = score
                best_cat = category

        return best_cat

    def _classify_economic_indicator(self, name: str) -> NarrativeCategory:
        """Classify an economic indicator into a narrative category."""
        name_lower = name.lower()
        if any(w in name_lower for w in ["trade", "import", "export", "tariff"]):
            return NarrativeCategory.TRADE_WAR
        if any(w in name_lower for w in ["fed", "rate", "cpi", "inflation"]):
            return NarrativeCategory.MONETARY_POLICY
        if any(w in name_lower for w in ["spending", "budget", "deficit", "debt"]):
            return NarrativeCategory.FISCAL_POLICY
        return NarrativeCategory.OTHER

    def _market_category_to_narrative(
        self, market_cat: MarketCategory
    ) -> NarrativeCategory:
        """Map MarketCategory to default NarrativeCategory."""
        mapping = {
            MarketCategory.POLITICS: NarrativeCategory.ELECTION,
            MarketCategory.CRYPTO: NarrativeCategory.CRYPTO_REGULATION,
            MarketCategory.SPORTS: NarrativeCategory.OTHER,
            MarketCategory.POP_CULTURE: NarrativeCategory.OTHER,
            MarketCategory.SCIENCE: NarrativeCategory.OTHER,
            MarketCategory.OTHER: NarrativeCategory.OTHER,
        }
        return mapping.get(market_cat, NarrativeCategory.OTHER)

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract meaningful keywords from text."""
        stop_words = {
            "will", "the", "be", "is", "on", "in", "at", "to", "by", "a",
            "an", "of", "for", "and", "or", "this", "that", "it", "with",
            "from", "has", "have", "was", "were", "been", "are", "what",
            "when", "where", "how", "does", "did", "not", "but", "can",
        }
        words = re.findall(r"[a-zA-Z]{3,}", text.lower())
        return [w for w in words if w not in stop_words][:15]

    def _event_fits_narrative(
        self, event: NarrativeEvent, narrative: Narrative
    ) -> bool:
        """Check if an event belongs to an existing narrative."""
        # Must be same category
        if event.category != narrative.category:
            return False

        # Must be within decay window
        if narrative.age_hours > self._decay_hours:
            return False

        # Check keyword overlap
        event_kw = set(w.lower() for w in event.keywords)
        narrative_kw = set()
        for ne in narrative.events[-10:]:  # Recent events only
            narrative_kw.update(w.lower() for w in ne.keywords)

        if not event_kw or not narrative_kw:
            return False

        overlap = len(event_kw & narrative_kw)
        return overlap >= 2

    def _create_narrative(
        self, events: list[NarrativeEvent], category: NarrativeCategory
    ) -> Narrative:
        """Create a new narrative from a cluster of related events."""
        # Compute average sentiment as direction
        sentiments = [e.sentiment for e in events]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

        # Build title from category
        titles = {
            NarrativeCategory.TRADE_WAR: "Trade War Developments",
            NarrativeCategory.MONETARY_POLICY: "Monetary Policy Shift",
            NarrativeCategory.GEOPOLITICAL: "Geopolitical Tensions",
            NarrativeCategory.CRYPTO_REGULATION: "Crypto Regulatory Changes",
            NarrativeCategory.FISCAL_POLICY: "Fiscal Policy Moves",
            NarrativeCategory.ELECTION: "Election Dynamics",
            NarrativeCategory.MARKET_CRISIS: "Market Crisis Unfolding",
            NarrativeCategory.OTHER: "Developing Situation",
        }

        # Build thesis from events
        sources = set(e.source for e in events)
        direction_word = "bullish" if avg_sentiment > 0 else "bearish"
        thesis = (
            f"Multiple {', '.join(sources)} signals indicate {direction_word} "
            f"pressure in {category.value.replace('_', ' ')} space"
        )

        narrative = Narrative(
            narrative_id=str(uuid.uuid4())[:12],
            title=titles.get(category, "Developing Situation"),
            category=category,
            thesis=thesis,
            events=list(events),
            predicted_direction=avg_sentiment,
            confidence=min(0.3 + len(events) * 0.05, 0.7),
            strength=min(0.3 + len(events) * 0.1, 0.9),
        )

        logger.info(
            "narrative_created",
            narrative_id=narrative.narrative_id,
            category=category.value,
            events=len(events),
        )

        return narrative

    def _update_narrative_state(self, narrative: Narrative) -> None:
        """Update narrative direction, confidence, strength after new event."""
        sentiments = [e.sentiment for e in narrative.events[-20:]]
        if sentiments:
            narrative.predicted_direction = sum(sentiments) / len(sentiments)

        narrative.strength = min(0.3 + len(narrative.events) * 0.08, 0.95)
        narrative.confidence = min(0.3 + len(narrative.events) * 0.04, 0.8)
        narrative.updated_at = datetime.now(timezone.utc)

    def _decay_narratives(self) -> None:
        """Apply time decay to narrative strength."""
        for narrative in self._active_narratives.values():
            age = narrative.age_hours
            if age > self._decay_hours:
                narrative.is_active = False
                narrative.strength = 0.0
            else:
                # Linear decay after half the decay window
                half = self._decay_hours / 2
                if age > half:
                    decay_factor = 1.0 - (age - half) / half
                    narrative.strength *= max(decay_factor, 0.1)

    def _prune_narratives(self) -> None:
        """Remove inactive or very weak narratives."""
        to_remove = [
            nid
            for nid, n in self._active_narratives.items()
            if not n.is_active or n.strength < 0.05
        ]
        for nid in to_remove:
            del self._active_narratives[nid]

    def _compute_market_relevance(
        self, narrative: Narrative, market: Market
    ) -> float:
        """Compute how relevant a narrative is to a specific market."""
        score = 0.0

        # Category alignment
        relevant_cats = _NARRATIVE_TO_MARKET_CATEGORY.get(narrative.category, [])
        if market.category in relevant_cats:
            score += 0.4

        # Keyword overlap between narrative events and market question
        narrative_kw = set()
        for event in narrative.events:
            narrative_kw.update(w.lower() for w in event.keywords)

        market_kw = set(self._extract_keywords(market.question))
        if narrative_kw and market_kw:
            overlap = len(narrative_kw & market_kw)
            score += min(overlap * 0.1, 0.4)

        # Narrative strength boost
        score *= narrative.strength

        return min(score, 1.0)
