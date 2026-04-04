"""Comprehensive tests for the narrative analysis system.

Tests cover:
- Data models (NarrativeEvent, Narrative, HistoricalPattern, SimulationScenario)
- NarrativeEngine (event ingestion, narrative clustering, decay, market mapping)
- HistoricalPatternMatcher (pattern matching, phase estimation, prediction)
- ScenarioCouncil (scenario generation, evaluation, voting, pruning)
- NarrativeStrategy (full pipeline generate_signals)
- EconomicDataClient (mock client)
- Integration (signal aggregator weight, full pipeline)
"""

from __future__ import annotations

from datetime import datetime, timezone

from polymarket_bot.clients.economic_data import MockEconomicDataClient
from polymarket_bot.data.models import (
    EconomicIndicator,
    HistoricalPattern,
    HistoricalPhase,
    Market,
    MarketCategory,
    Narrative,
    NarrativeCategory,
    NarrativeEvent,
    OrderBook,
    OrderBookLevel,
    SimulationScenario,
    Token,
)
from polymarket_bot.narrative.council import ScenarioCouncil
from polymarket_bot.narrative.engine import NarrativeEngine
from polymarket_bot.narrative.pattern_library import get_builtin_patterns
from polymarket_bot.narrative.patterns import HistoricalPatternMatcher
from polymarket_bot.narrative.strategy import NarrativeStrategy
from polymarket_bot.strategies.signals import STRATEGY_WEIGHTS


# ── Test Helpers ────────────────────────────────────────────────


def _make_market(
    condition_id: str = "mkt_1",
    question: str = "Will tariffs increase?",
    category: MarketCategory = MarketCategory.POLITICS,
    yes_price: float = 0.55,
) -> Market:
    return Market(
        condition_id=condition_id,
        question=question,
        slug="test-market",
        tokens=[
            Token(token_id=f"{condition_id}_yes", outcome="Yes", price=yes_price),
            Token(
                token_id=f"{condition_id}_no",
                outcome="No",
                price=round(1 - yes_price, 4),
            ),
        ],
        category=category,
        liquidity=5000.0,
        volume_24h=10000.0,
        active=True,
    )


def _make_narrative_event(
    content: str = "New tariff announced on Chinese goods",
    category: NarrativeCategory = NarrativeCategory.TRADE_WAR,
    sentiment: float = -0.4,
    keywords: list[str] | None = None,
    source: str = "twitter",
) -> NarrativeEvent:
    return NarrativeEvent(
        event_id="evt_test",
        source=source,
        content=content,
        timestamp=datetime.now(timezone.utc),
        category=category,
        sentiment=sentiment,
        magnitude=0.5,
        keywords=keywords or ["tariff", "china", "trade"],
    )


def _make_narrative(
    category: NarrativeCategory = NarrativeCategory.TRADE_WAR,
    num_events: int = 5,
    direction: float = -0.3,
) -> Narrative:
    events = []
    for i in range(num_events):
        events.append(
            _make_narrative_event(
                content=f"Trade war event {i}",
                category=category,
                sentiment=direction,
                keywords=["tariff", "china", "trade", "duties"],
            )
        )
    return Narrative(
        narrative_id="narr_test",
        title="Trade War Developments",
        category=category,
        thesis="Tariff escalation is increasing",
        events=events,
        predicted_direction=direction,
        confidence=0.6,
        strength=0.7,
    )


def _make_order_book(token_id: str, mid: float = 0.55) -> OrderBook:
    spread = 0.02
    return OrderBook(
        token_id=token_id,
        bids=[OrderBookLevel(price=mid - spread / 2, size=100)],
        asks=[OrderBookLevel(price=mid + spread / 2, size=100)],
    )


# ── Data Model Tests ────────────────────────────────────────────


class TestDataModels:
    def test_narrative_event_creation(self):
        event = _make_narrative_event()
        assert event.event_id == "evt_test"
        assert event.category == NarrativeCategory.TRADE_WAR
        assert event.sentiment == -0.4
        assert "tariff" in event.keywords

    def test_narrative_creation(self):
        narrative = _make_narrative()
        assert narrative.narrative_id == "narr_test"
        assert narrative.category == NarrativeCategory.TRADE_WAR
        assert narrative.event_count == 5
        assert narrative.is_active is True
        assert narrative.age_hours >= 0

    def test_narrative_age_hours(self):
        narrative = _make_narrative()
        assert narrative.age_hours < 0.01  # Just created

    def test_economic_indicator(self):
        ind = EconomicIndicator(
            name="trade_balance",
            value=-70.0,
            previous_value=-65.0,
            change_pct=-7.7,
            unit="billion_usd",
        )
        assert ind.name == "trade_balance"
        assert ind.change_pct == -7.7

    def test_historical_phase(self):
        phase = HistoricalPhase(
            phase_name="Escalation",
            description="Tariffs increase",
            duration_days=30,
            market_impact={"politics": -0.1, "crypto": -0.05},
            keywords=["tariff", "escalation"],
            sequence_index=0,
        )
        assert phase.phase_name == "Escalation"
        assert phase.market_impact["politics"] == -0.1

    def test_historical_pattern(self):
        pattern = HistoricalPattern(
            pattern_id="test_pattern",
            name="Test Pattern",
            category=NarrativeCategory.TRADE_WAR,
            description="Test",
            trigger_keywords=["tariff", "trade"],
            phases=[
                HistoricalPhase("P1", "Phase 1", 30, {"politics": -0.1}, ["tariff"], 0),
                HistoricalPhase("P2", "Phase 2", 60, {"politics": 0.1}, ["deal"], 1),
            ],
        )
        assert len(pattern.phases) == 2
        assert pattern.trigger_keywords == ["tariff", "trade"]

    def test_simulation_scenario(self):
        scenario = SimulationScenario(
            scenario_id="sc_1",
            name="Full Escalation",
            narrative_id="narr_1",
            predicted_direction=-0.5,
            predicted_magnitude=0.1,
            weight=0.3,
        )
        assert scenario.avg_accuracy == 0.5  # No history
        scenario.accuracy_history = [0.8, 0.6, 0.7]
        assert abs(scenario.avg_accuracy - 0.7) < 0.01


# ── Pattern Library Tests ───────────────────────────────────────


class TestPatternLibrary:
    def test_loads_all_patterns(self):
        patterns = get_builtin_patterns()
        assert len(patterns) == 10

    def test_all_patterns_have_required_fields(self):
        for p in get_builtin_patterns():
            assert p.pattern_id
            assert p.name
            assert isinstance(p.category, NarrativeCategory)
            assert p.description
            assert len(p.trigger_keywords) > 0

    def test_all_patterns_have_phases(self):
        for p in get_builtin_patterns():
            assert len(p.phases) >= 3, f"{p.name} has fewer than 3 phases"

    def test_phase_sequence_indices(self):
        for p in get_builtin_patterns():
            for i, phase in enumerate(p.phases):
                assert phase.sequence_index == i, (
                    f"{p.name} phase {phase.phase_name} has wrong index"
                )

    def test_trade_war_pattern_exists(self):
        patterns = get_builtin_patterns()
        trade_war = [p for p in patterns if "trade war" in p.name.lower()]
        assert len(trade_war) == 1
        tw = trade_war[0]
        assert tw.category == NarrativeCategory.TRADE_WAR
        assert "tariff" in tw.trigger_keywords


# ── Historical Pattern Matcher Tests ────────────────────────────


class TestHistoricalPatternMatcher:
    def setup_method(self):
        self.matcher = HistoricalPatternMatcher(similarity_threshold=0.4)

    def test_pattern_count(self):
        assert self.matcher.pattern_count == 10

    def test_trade_war_narrative_matches_trade_war_pattern(self):
        narrative = _make_narrative(
            category=NarrativeCategory.TRADE_WAR,
            num_events=5,
            direction=-0.3,
        )
        matches = self.matcher.find_matching_patterns(narrative)
        assert len(matches) > 0
        best_pattern, score, phase_idx = matches[0]
        assert best_pattern.category == NarrativeCategory.TRADE_WAR
        assert score >= 0.4

    def test_no_match_for_unrelated_narrative(self):
        narrative = Narrative(
            narrative_id="narr_unrelated",
            title="Celebrity Gossip",
            category=NarrativeCategory.OTHER,
            thesis="Some celebrity did something",
            events=[
                NarrativeEvent(
                    event_id="e1",
                    source="twitter",
                    content="celebrity gossip news",
                    timestamp=datetime.now(timezone.utc),
                    category=NarrativeCategory.OTHER,
                    sentiment=0.1,
                    magnitude=0.2,
                    keywords=["celebrity", "gossip", "entertainment"],
                )
            ],
            predicted_direction=0.1,
            confidence=0.3,
            strength=0.3,
        )
        matches = self.matcher.find_matching_patterns(narrative)
        # Should have no matches or very low scores
        high_matches = [(p, s, i) for p, s, i in matches if s >= 0.5]
        assert len(high_matches) == 0

    def test_predict_next_phase(self):
        patterns = get_builtin_patterns()
        trade_war = next(p for p in patterns if "trade war" in p.name.lower())

        # If in phase 0 (escalation), predict phase 1 (retaliation)
        direction, magnitude = self.matcher.predict_next_phase(trade_war, 0)
        assert isinstance(direction, float)
        assert isinstance(magnitude, float)
        assert -1.0 <= direction <= 1.0
        assert 0.0 <= magnitude <= 0.2

    def test_predict_last_phase_uses_outcome(self):
        patterns = get_builtin_patterns()
        trade_war = next(p for p in patterns if "trade war" in p.name.lower())
        last_idx = len(trade_war.phases) - 1

        direction, magnitude = self.matcher.predict_next_phase(
            trade_war, last_idx
        )
        # Should use outcome with decay
        assert isinstance(direction, float)

    def test_similarity_scoring_category_weight(self):
        narrative = _make_narrative(category=NarrativeCategory.TRADE_WAR)
        matches = self.matcher.find_matching_patterns(narrative)

        # Best match should be trade war category
        if matches:
            assert matches[0][0].category == NarrativeCategory.TRADE_WAR


# ── Narrative Engine Tests ──────────────────────────────────────


class TestNarrativeEngine:
    def setup_method(self):
        self.engine = NarrativeEngine(
            max_active=10,
            min_events=3,
            decay_hours=48.0,
        )

    def test_ingest_tweet_events(self):
        events = [
            {"text": "New tariffs on China announced!", "sentiment": -0.5,
             "keywords": ["tariff", "china"], "source": "twitter"},
            {"text": "Trade war escalating", "sentiment": -0.6,
             "keywords": ["trade", "war", "escalating"], "source": "twitter"},
            {"text": "More tariff threats", "sentiment": -0.4,
             "keywords": ["tariff", "threats"], "source": "twitter"},
        ]
        self.engine.ingest_tweet_events(events)
        assert len(self.engine._event_buffer) == 3

    def test_ingest_economic_data(self):
        indicators = [
            {"name": "trade_balance", "value": -75.0,
             "previous_value": -65.0, "change_pct": -15.4, "unit": "billion_usd"},
        ]
        self.engine.ingest_economic_data(indicators)
        assert len(self.engine._event_buffer) == 1
        assert self.engine._event_buffer[0].category == NarrativeCategory.TRADE_WAR

    def test_small_economic_changes_filtered(self):
        indicators = [
            {"name": "cpi_yoy", "value": 3.21,
             "previous_value": 3.20, "change_pct": 0.3, "unit": "percent"},
        ]
        self.engine.ingest_economic_data(indicators)
        assert len(self.engine._event_buffer) == 0  # Too small

    def test_narrative_creation_from_events(self):
        # Need min_events (3) events in same category
        events = [
            {"text": f"Tariff news {i}", "sentiment": -0.4,
             "keywords": ["tariff", "china", "trade"], "source": "twitter"}
            for i in range(4)
        ]
        self.engine.ingest_tweet_events(events)
        narratives = self.engine.update_narratives()
        assert len(narratives) >= 1
        assert narratives[0].category == NarrativeCategory.TRADE_WAR

    def test_narrative_update_with_new_events(self):
        # Create initial narrative
        events = [
            {"text": f"Tariff event {i}", "sentiment": -0.3,
             "keywords": ["tariff", "china", "trade", "duties"], "source": "twitter"}
            for i in range(4)
        ]
        self.engine.ingest_tweet_events(events)
        narratives = self.engine.update_narratives()
        initial_count = narratives[0].event_count if narratives else 0

        # Add more events that should match
        more_events = [
            {"text": "More tariff escalation", "sentiment": -0.5,
             "keywords": ["tariff", "china", "escalation", "trade"], "source": "twitter"},
        ]
        self.engine.ingest_tweet_events(more_events)
        narratives = self.engine.update_narratives()

        if narratives:
            assert narratives[0].event_count >= initial_count

    def test_max_active_narratives(self):
        engine = NarrativeEngine(max_active=2, min_events=1)

        categories = [
            NarrativeCategory.TRADE_WAR,
            NarrativeCategory.MONETARY_POLICY,
            NarrativeCategory.GEOPOLITICAL,
        ]
        for cat in categories:
            events = [
                _make_narrative_event(
                    content=f"{cat.value} event",
                    category=cat,
                    keywords=[cat.value, "test", "event"],
                )
            ]
            engine._event_buffer.extend(events)

        engine.update_narratives()
        assert len(engine._active_narratives) <= 2

    def test_narrative_to_market_mapping(self):
        narratives = [_make_narrative()]
        markets = [
            _make_market("m1", "Will US tariffs on China increase?", MarketCategory.POLITICS),
            _make_market("m2", "Will Bitcoin reach $100k?", MarketCategory.CRYPTO),
        ]
        mapping = self.engine.map_narratives_to_markets(narratives, markets)
        # Tariff narrative should be more relevant to tariff market
        if "m1" in mapping:
            assert len(mapping["m1"]) > 0

    def test_get_active_narratives_sorted(self):
        n1 = _make_narrative()
        n1.narrative_id = "n1"
        n1.strength = 0.5
        n2 = _make_narrative()
        n2.narrative_id = "n2"
        n2.strength = 0.8
        self.engine._active_narratives = {"n1": n1, "n2": n2}

        active = self.engine.get_active_narratives()
        assert active[0].narrative_id == "n2"  # Stronger first


# ── Scenario Council Tests ──────────────────────────────────────


class TestScenarioCouncil:
    def setup_method(self):
        self.council = ScenarioCouncil(
            council_size=5,
            min_agreement=0.6,
            evaluation_interval=1,
        )

    def test_generate_scenarios(self):
        narrative = _make_narrative()
        scenarios = self.council.generate_scenarios(narrative)
        assert len(scenarios) == 5
        assert all(s.narrative_id == narrative.narrative_id for s in scenarios)

    def test_scenarios_have_different_names(self):
        narrative = _make_narrative()
        scenarios = self.council.generate_scenarios(narrative)
        names = [s.name for s in scenarios]
        assert len(set(names)) == len(names)  # All unique

    def test_scenarios_weights_sum_to_one(self):
        narrative = _make_narrative()
        scenarios = self.council.generate_scenarios(narrative)
        total = sum(s.weight for s in scenarios)
        assert abs(total - 1.0) < 0.01

    def test_vote_returns_direction_magnitude_agreement(self):
        narrative = _make_narrative()
        self.council.generate_scenarios(narrative)
        direction, magnitude, agreement = self.council.vote(
            narrative.narrative_id
        )
        assert -1.0 <= direction <= 1.0
        assert 0.0 <= magnitude <= 0.2
        assert 0.0 <= agreement <= 1.0

    def test_vote_empty_narrative_returns_zeros(self):
        direction, magnitude, agreement = self.council.vote("nonexistent")
        assert direction == 0.0
        assert magnitude == 0.0
        assert agreement == 0.0

    def test_evaluate_updates_weights(self):
        narrative = _make_narrative()
        self.council.generate_scenarios(narrative)

        initial_weights = [
            s.weight for s in self.council.get_scenarios(narrative.narrative_id)
        ]

        # Add events and evaluate
        markets = [_make_market()]
        self.council.evaluate_scenarios(narrative, markets)

        new_weights = [
            s.weight for s in self.council.get_scenarios(narrative.narrative_id)
        ]
        # Weights should still sum to ~1
        assert abs(sum(new_weights) - 1.0) < 0.01

    def test_prune_and_refresh(self):
        narrative = _make_narrative()
        scenarios = self.council.generate_scenarios(narrative)

        # Artificially make one scenario very weak
        scenarios[0].weight = 0.001
        # Renormalize others
        total = sum(s.weight for s in scenarios[1:])
        for s in scenarios[1:]:
            s.weight /= total * (1 / 0.999)

        refreshed = self.council.prune_and_refresh(narrative)
        assert len(refreshed) == 5  # Should be back to council_size
        # Weights should sum to ~1
        assert abs(sum(s.weight for s in refreshed) - 1.0) < 0.01

    def test_minimum_scenarios_preserved(self):
        narrative = _make_narrative()
        scenarios = self.council.generate_scenarios(narrative)

        # Make all but one very weak
        for i, s in enumerate(scenarios):
            s.weight = 0.001 if i > 0 else 1.0

        refreshed = self.council.prune_and_refresh(narrative)
        assert len(refreshed) >= 2


# ── Economic Data Client Tests ──────────────────────────────────


class TestEconomicDataClient:
    def test_mock_client_returns_indicators(self):
        client = MockEconomicDataClient(seed=42)
        indicators = client.get_all_indicators()
        assert len(indicators) > 0
        for ind in indicators:
            assert "name" in ind
            assert "value" in ind
            assert "change_pct" in ind

    def test_mock_client_deterministic(self):
        c1 = MockEconomicDataClient(seed=42)
        c2 = MockEconomicDataClient(seed=42)
        c1.step()
        c2.step()
        i1 = c1.get_all_indicators()
        i2 = c2.get_all_indicators()
        for a, b in zip(i1, i2):
            assert a["value"] == b["value"]

    def test_mock_client_different_seeds(self):
        c1 = MockEconomicDataClient(seed=42)
        c2 = MockEconomicDataClient(seed=99)
        c1.step()
        c2.step()
        i1 = c1.get_all_indicators()
        i2 = c2.get_all_indicators()
        # At least some values should differ
        diffs = [a["value"] != b["value"] for a, b in zip(i1, i2)]
        assert any(diffs)

    def test_mock_client_values_change_over_time(self):
        client = MockEconomicDataClient(seed=42)
        initial = client.get_all_indicators()
        for _ in range(10):
            client.step()
        after = client.get_all_indicators()
        # Values should have changed
        diffs = [
            abs(a["value"] - b["value"]) > 0.001
            for a, b in zip(initial, after)
        ]
        assert any(diffs)

    def test_trade_data(self):
        client = MockEconomicDataClient(seed=42)
        trade = client.get_trade_data()
        assert "trade_balance_bn" in trade
        assert "imports_bn" in trade
        assert "exports_bn" in trade
        assert "tariff_rate_avg" in trade


# ── Narrative Strategy Tests ────────────────────────────────────


class TestNarrativeStrategy:
    def setup_method(self):
        self.econ_client = MockEconomicDataClient(seed=42)
        self.strategy = NarrativeStrategy(
            economic_client=self.econ_client,
            min_edge=0.03,
            min_events=2,  # Lower for testing
        )

    def test_implements_base_strategy(self):
        from polymarket_bot.strategies.base import BaseStrategy
        assert isinstance(self.strategy, BaseStrategy)

    def test_strategy_name(self):
        assert self.strategy.name == "narrative_analysis"

    def test_no_signals_without_events(self):
        markets = [_make_market()]
        order_books = {
            "mkt_1_yes": _make_order_book("mkt_1_yes"),
            "mkt_1_no": _make_order_book("mkt_1_no", 0.45),
        }
        signals = self.strategy.generate_signals(markets, order_books, {})
        assert isinstance(signals, list)

    def test_generates_signals_with_narrative_events(self):
        markets = [
            _make_market("m1", "Will US tariffs on China increase?", MarketCategory.POLITICS, 0.50),
        ]
        order_books = {
            "m1_yes": _make_order_book("m1_yes", 0.50),
            "m1_no": _make_order_book("m1_no", 0.50),
        }

        # Feed enough events to create a narrative
        context = {
            "news_events": [
                {"text": f"Tariff escalation event {i}", "sentiment": -0.5,
                 "keywords": ["tariff", "china", "trade", "escalation", "increase"],
                 "source": "trump_tweet"}
                for i in range(5)
            ],
        }

        signals = self.strategy.generate_signals(markets, order_books, context)
        assert isinstance(signals, list)
        for sig in signals:
            assert sig.strategy == "narrative_analysis"
            assert "narratives" in sig.metadata
            assert sig.edge >= 0

    def test_signal_confidence_capped(self):
        markets = [
            _make_market("m1", "Will tariffs escalate?", MarketCategory.POLITICS, 0.40),
        ]
        order_books = {}
        context = {
            "news_events": [
                {"text": f"Major tariff war event {i}", "sentiment": -0.8,
                 "keywords": ["tariff", "china", "trade", "war", "escalate"],
                 "source": "twitter"}
                for i in range(10)
            ],
        }

        signals = self.strategy.generate_signals(markets, order_books, context)
        for sig in signals:
            assert sig.confidence <= 0.75  # Max confidence cap


# ── Integration Tests ───────────────────────────────────────────


class TestIntegration:
    def test_strategy_weight_in_aggregator(self):
        assert "narrative_analysis" in STRATEGY_WEIGHTS
        assert STRATEGY_WEIGHTS["narrative_analysis"] == 1.15

    def test_full_pipeline_with_mock_data(self):
        """End-to-end: economic data + news events -> narratives -> signals."""
        econ = MockEconomicDataClient(seed=42)
        strategy = NarrativeStrategy(
            economic_client=econ,
            min_edge=0.03,
            min_events=2,
        )

        markets = [
            _make_market("m1", "Will US impose new tariffs on China?", MarketCategory.POLITICS, 0.50),
            _make_market("m2", "Will Bitcoin reach $100k?", MarketCategory.CRYPTO, 0.30),
        ]
        order_books = {}

        # Step economic data to generate changes
        for _ in range(5):
            econ.step()

        context = {
            "news_events": [
                {"text": "Trump announces massive new tariffs on China",
                 "sentiment": -0.7, "keywords": ["tariff", "china", "trump", "trade"],
                 "source": "trump_tweet"},
                {"text": "China threatens retaliation on US tariffs",
                 "sentiment": -0.6, "keywords": ["china", "retaliation", "tariff", "trade"],
                 "source": "twitter"},
                {"text": "Trade war fears escalate globally",
                 "sentiment": -0.5, "keywords": ["trade", "war", "tariff", "global"],
                 "source": "twitter"},
            ],
            "economic_indicators": econ.get_all_indicators(),
        }

        # First call builds narratives
        signals1 = strategy.generate_signals(markets, order_books, context)
        assert isinstance(signals1, list)

        # Second call should have established narratives
        context2 = {
            "news_events": [
                {"text": "More tariff threats from administration",
                 "sentiment": -0.4, "keywords": ["tariff", "threats", "trade", "china"],
                 "source": "twitter"},
            ],
        }
        signals2 = strategy.generate_signals(markets, order_books, context2)
        assert isinstance(signals2, list)

    def test_narrative_strategy_with_market_moves(self):
        """Test that market moves are detected and feed into narratives."""
        strategy = NarrativeStrategy(
            economic_client=MockEconomicDataClient(seed=42),
            min_edge=0.03,
            min_events=2,
        )

        markets = [
            _make_market("m1", "Will tariffs increase?", MarketCategory.POLITICS, 0.60),
        ]
        order_books = {}

        # Provide price history showing a big move
        context = {
            "price_history_m1": [0.50, 0.51, 0.52, 0.53, 0.60],
            "news_events": [
                {"text": "Tariff announcement drives market", "sentiment": -0.5,
                 "keywords": ["tariff", "market", "trade", "increase"],
                 "source": "twitter"},
                {"text": "More tariff news", "sentiment": -0.4,
                 "keywords": ["tariff", "news", "trade"],
                 "source": "twitter"},
            ],
        }

        signals = strategy.generate_signals(markets, order_books, context)
        assert isinstance(signals, list)
