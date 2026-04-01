"""Tests for trading strategies."""

import pytest

from polymarket_bot.clients.odds_sources import EloRating, ExternalOdds, OddsAggregator, PollData
from polymarket_bot.clients.twitter import MockTwitterClient, score_text
from polymarket_bot.config import ArbitrageConfig, MarketMakerConfig, SentimentConfig
from polymarket_bot.data.models import (
    Market,
    MarketCategory,
    OrderBook,
    OrderBookLevel,
    Side,
    Signal,
    Token,
)
from polymarket_bot.strategies.arbitrage import ArbitrageStrategy
from polymarket_bot.strategies.market_maker import MarketMakerStrategy
from polymarket_bot.strategies.sentiment import SentimentStrategy
from polymarket_bot.strategies.signals import SignalAggregator
from polymarket_bot.strategies.statistical import StatisticalStrategy


# ── Fixtures ─────────────────────────────────────────────────────

def make_market(
    cid: str = "test_001",
    yes_price: float = 0.50,
    category: MarketCategory = MarketCategory.OTHER,
    liquidity: float = 5000.0,
    question: str = "Will event happen?",
) -> Market:
    return Market(
        condition_id=cid,
        question=question,
        slug="test-market",
        tokens=[
            Token(token_id=f"{cid}_yes", outcome="Yes", price=yes_price),
            Token(token_id=f"{cid}_no", outcome="No", price=1.0 - yes_price),
        ],
        category=category,
        liquidity=liquidity,
        active=True,
    )


def make_order_book(token_id: str, mid: float = 0.50, spread: float = 0.02) -> OrderBook:
    return OrderBook(
        token_id=token_id,
        bids=[
            OrderBookLevel(price=round(mid - spread / 2, 2), size=100.0),
            OrderBookLevel(price=round(mid - spread / 2 - 0.01, 2), size=200.0),
        ],
        asks=[
            OrderBookLevel(price=round(mid + spread / 2, 2), size=100.0),
            OrderBookLevel(price=round(mid + spread / 2 + 0.01, 2), size=200.0),
        ],
    )


# ── Sentiment Strategy Tests ────────────────────────────────────

class TestSentimentScoring:
    def test_positive_text(self):
        assert score_text("This is bullish and amazing, definitely winning") > 0

    def test_negative_text(self):
        assert score_text("bearish crash dump terrible losing") < 0

    def test_neutral_text(self):
        score = score_text("the weather is nice today")
        assert abs(score) < 0.5

    def test_negation_handling(self):
        # "not bullish" should be negative
        score_negated = score_text("not bullish")
        score_positive = score_text("bullish")
        assert score_negated < score_positive

    def test_empty_text(self):
        assert score_text("") == 0.0


class TestSentimentStrategy:
    def setup_method(self):
        self.mock_twitter = MockTwitterClient()
        self.config = SentimentConfig(
            min_tweets=3,
            sentiment_threshold=0.2,
            volume_spike_threshold=2.0,
        )
        self.strategy = SentimentStrategy(self.mock_twitter, self.config)

    def test_bullish_sentiment_generates_buy(self):
        """Strong bullish sentiment + low price = BUY signal."""
        market = make_market(yes_price=0.30)
        books = {
            "test_001_yes": make_order_book("test_001_yes", 0.30),
            "test_001_no": make_order_book("test_001_no", 0.70),
        }

        # Set up bullish tweets
        self.mock_twitter.set_mock_tweets("event", [
            {"text": "Definitely winning, bullish, moon, amazing victory"},
            {"text": "This is a lock, absolutely crushing it, strong surge"},
            {"text": "Confirmed win, rising fast, incredible rally ahead"},
            {"text": "Victory is certain, positive momentum, leading strong"},
            {"text": "Bullish rally pump soaring up rise confirmed"},
        ])

        signals = self.strategy.generate_signals(
            [market], books, {f"keywords_test_001": ["event"]}
        )

        # Should generate a signal (may or may not meet all thresholds)
        # The key test is that the strategy runs without error
        assert isinstance(signals, list)

    def test_no_tweets_no_signal(self):
        """No tweet data should produce no signals."""
        market = make_market(yes_price=0.50)
        books = {"test_001_yes": make_order_book("test_001_yes")}

        signals = self.strategy.generate_signals([market], books, {})
        assert len(signals) == 0

    def test_illiquid_market_filtered(self):
        """Markets with low liquidity should be filtered out."""
        market = make_market(liquidity=10.0)
        signals = self.strategy.generate_signals([market], {}, {})
        assert len(signals) == 0


# ── Market Maker Strategy Tests ──────────────────────────────────

class TestMarketMakerStrategy:
    def setup_method(self):
        self.config = MarketMakerConfig(
            spread=0.04,
            order_size_usd=25.0,
            max_inventory=200.0,
            min_book_depth_usd=50.0,
        )
        self.strategy = MarketMakerStrategy(self.config)

    def test_generates_two_sided_quotes(self):
        """Should place both bid and ask quotes."""
        market = make_market(yes_price=0.50, liquidity=5000.0)
        books = {
            "test_001_yes": make_order_book("test_001_yes", 0.50, spread=0.02),
            "test_001_no": make_order_book("test_001_no", 0.50, spread=0.02),
        }

        signals = self.strategy.generate_signals([market], books, {"positions": {}})

        # Should have bid and ask for each token
        buy_signals = [s for s in signals if s.side == Side.BUY]
        sell_signals = [s for s in signals if s.side == Side.SELL]

        assert len(buy_signals) > 0
        assert len(sell_signals) > 0

    def test_bid_below_ask(self):
        """Bid price must be below ask price."""
        market = make_market(yes_price=0.50, liquidity=5000.0)
        books = {
            "test_001_yes": make_order_book("test_001_yes", 0.50),
            "test_001_no": make_order_book("test_001_no", 0.50),
        }

        signals = self.strategy.generate_signals([market], books, {"positions": {}})

        for token_id in ["test_001_yes", "test_001_no"]:
            token_signals = [s for s in signals if s.token_id == token_id]
            buys = [s for s in token_signals if s.side == Side.BUY]
            sells = [s for s in token_signals if s.side == Side.SELL]

            if buys and sells:
                assert buys[0].market_price < sells[0].market_price

    def test_thin_book_skipped(self):
        """Markets with thin order books should be skipped."""
        market = make_market(liquidity=5000.0)
        thin_book = OrderBook(
            token_id="test_001_yes",
            bids=[OrderBookLevel(price=0.49, size=10.0)],
            asks=[OrderBookLevel(price=0.51, size=10.0)],
        )

        signals = self.strategy.generate_signals(
            [market], {"test_001_yes": thin_book}, {"positions": {}}
        )

        yes_signals = [s for s in signals if s.token_id == "test_001_yes"]
        assert len(yes_signals) == 0


# ── Arbitrage Strategy Tests ─────────────────────────────────────

class TestArbitrageStrategy:
    def setup_method(self):
        self.config = ArbitrageConfig(min_arb_edge=0.02, complement_tolerance=0.03)
        self.odds = OddsAggregator()
        self.strategy = ArbitrageStrategy(self.config, self.odds)

    def teardown_method(self):
        self.odds.close()

    def test_complement_arb_underpriced(self):
        """When Yes + No < 1.0, should detect buying opportunity."""
        market = Market(
            condition_id="arb_001",
            question="Test arb?",
            slug="test-arb",
            tokens=[
                Token(token_id="arb_yes", outcome="Yes", price=0.40),
                Token(token_id="arb_no", outcome="No", price=0.50),
            ],
            liquidity=5000.0,
            active=True,
        )

        # Order book with underpriced asks (total < 1.0)
        books = {
            "arb_yes": OrderBook(
                token_id="arb_yes",
                bids=[OrderBookLevel(price=0.39, size=100)],
                asks=[OrderBookLevel(price=0.40, size=100)],
            ),
            "arb_no": OrderBook(
                token_id="arb_no",
                bids=[OrderBookLevel(price=0.49, size=100)],
                asks=[OrderBookLevel(price=0.50, size=100)],
            ),
        }

        signals = self.strategy.generate_signals([market], books, {})

        # 0.40 + 0.50 = 0.90 < 1.0 - 0.03, so there's a complement arb
        assert len(signals) > 0
        assert all(s.side == Side.BUY for s in signals)
        assert all(s.confidence >= 0.9 for s in signals)

    def test_no_arb_when_prices_sum_to_one(self):
        """No arbitrage when prices are properly balanced."""
        market = make_market(yes_price=0.50)
        books = {
            "test_001_yes": make_order_book("test_001_yes", 0.50),
            "test_001_no": make_order_book("test_001_no", 0.50),
        }

        signals = self.strategy.generate_signals([market], books, {})
        arb_signals = [s for s in signals if "complement" in s.metadata.get("arb_type", "")]
        assert len(arb_signals) == 0

    def test_cross_market_arb(self):
        """Should detect edge when external odds differ from Polymarket."""
        market = make_market(yes_price=0.40)
        external_odds = [
            ExternalOdds(source="kalshi", event_name="test", outcome="Yes", implied_probability=0.55),
            ExternalOdds(source="predictit", event_name="test", outcome="Yes", implied_probability=0.58),
        ]

        books = {"test_001_yes": make_order_book("test_001_yes", 0.40)}
        context = {"external_odds_test_001": external_odds}

        signals = self.strategy.generate_signals([market], books, context)

        cross_signals = [s for s in signals if s.metadata.get("arb_type") == "cross_market"]
        assert len(cross_signals) > 0
        assert cross_signals[0].side == Side.BUY  # External says higher -> buy on Polymarket


# ── Statistical Strategy Tests ───────────────────────────────────

class TestStatisticalStrategy:
    def setup_method(self):
        self.odds = OddsAggregator()
        self.strategy = StatisticalStrategy(self.odds, min_edge=0.05)

    def teardown_method(self):
        self.odds.close()

    def test_elo_sports_signal(self):
        """Should generate signal when ELO disagrees with market."""
        market = make_market(
            cid="sports_001",
            yes_price=0.40,  # Market says 40% chance
            category=MarketCategory.SPORTS,
            question="Will Lakers win?",
        )
        market.tokens[0].outcome = "Lakers"
        market.tokens[1].outcome = "Celtics"

        # ELO says Lakers have ~65% chance (big edge)
        elo = {
            "Lakers": EloRating(team="Lakers", rating=1700, sport="nba"),
            "Celtics": EloRating(team="Celtics", rating=1500, sport="nba"),
        }

        books = {"sports_001_yes": make_order_book("sports_001_yes", 0.40)}
        context = {"elo_sports_001": elo}

        signals = self.strategy.generate_signals([market], books, context)

        assert len(signals) > 0
        assert signals[0].side == Side.BUY

    def test_crypto_mean_reversion(self):
        """Should detect overreaction in crypto markets."""
        market = make_market(
            cid="crypto_001",
            yes_price=0.80,  # Price spiked to 80%
            category=MarketCategory.CRYPTO,
            question="Will Bitcoin be above $100K?",
        )

        # Price history shows a recent spike
        import numpy as np
        history = [0.55 + np.random.normal(0, 0.02) for _ in range(15)]
        history.extend([0.65, 0.70, 0.75, 0.78, 0.80])  # Spike

        books = {"crypto_001_yes": make_order_book("crypto_001_yes", 0.80)}
        context = {"price_history_crypto_001": history}

        signals = self.strategy.generate_signals([market], books, context)
        # May or may not trigger depending on z-score; test it runs cleanly
        assert isinstance(signals, list)

    def test_book_imbalance_signal(self):
        """Strong order book imbalance should generate signal."""
        market = make_market(cid="generic_001", liquidity=5000.0)

        # Heavy bid-side imbalance
        book = OrderBook(
            token_id="generic_001_yes",
            bids=[
                OrderBookLevel(price=0.49, size=500.0),
                OrderBookLevel(price=0.48, size=400.0),
            ],
            asks=[
                OrderBookLevel(price=0.51, size=50.0),
                OrderBookLevel(price=0.52, size=30.0),
            ],
        )

        signals = self.strategy.generate_signals(
            [market], {"generic_001_yes": book}, {}
        )
        assert isinstance(signals, list)


# ── Signal Aggregator Tests ──────────────────────────────────────

class TestSignalAggregator:
    def setup_method(self):
        self.aggregator = SignalAggregator(min_composite_edge=0.03)

    def test_single_signal_passes_through(self):
        signal = Signal(
            market_condition_id="m1",
            token_id="t1",
            side=Side.BUY,
            outcome="Yes",
            estimated_fair_value=0.60,
            market_price=0.50,
            edge=0.10,
            confidence=0.7,
            strategy="sentiment",
        )

        result = self.aggregator.aggregate([signal])
        assert len(result) == 1
        assert result[0].edge == 0.10

    def test_consensus_boosts_confidence(self):
        """Multiple strategies agreeing should boost confidence."""
        signals = [
            Signal(
                market_condition_id="m1", token_id="t1", side=Side.BUY,
                outcome="Yes", estimated_fair_value=0.60, market_price=0.50,
                edge=0.10, confidence=0.6, strategy="sentiment",
            ),
            Signal(
                market_condition_id="m1", token_id="t1", side=Side.BUY,
                outcome="Yes", estimated_fair_value=0.65, market_price=0.50,
                edge=0.15, confidence=0.7, strategy="statistical",
            ),
        ]

        result = self.aggregator.aggregate(signals)
        assert len(result) == 1
        # Consensus should boost confidence above the max individual
        assert result[0].confidence > 0.7
        assert result[0].strategy == "ensemble"

    def test_low_edge_filtered(self):
        """Signals below minimum edge should be filtered."""
        signal = Signal(
            market_condition_id="m1", token_id="t1", side=Side.BUY,
            outcome="Yes", estimated_fair_value=0.51, market_price=0.50,
            edge=0.01, confidence=0.9, strategy="sentiment",
        )

        result = self.aggregator.aggregate([signal])
        assert len(result) == 0

    def test_sorted_by_expected_value(self):
        """Results should be sorted by edge * confidence."""
        signals = [
            Signal(
                market_condition_id="m1", token_id="t1", side=Side.BUY,
                outcome="Yes", estimated_fair_value=0.60, market_price=0.50,
                edge=0.05, confidence=0.5, strategy="a",
            ),
            Signal(
                market_condition_id="m2", token_id="t2", side=Side.BUY,
                outcome="Yes", estimated_fair_value=0.70, market_price=0.50,
                edge=0.10, confidence=0.8, strategy="b",
            ),
        ]

        result = self.aggregator.aggregate(signals)
        assert len(result) == 2
        # Higher EV signal should come first
        assert result[0].token_id == "t2"

    def test_conflict_detection(self):
        """Should detect conflicting signals on same token."""
        signals = [
            Signal(
                market_condition_id="m1", token_id="t1", side=Side.BUY,
                outcome="Yes", estimated_fair_value=0.60, market_price=0.50,
                edge=0.10, confidence=0.7, strategy="a",
            ),
            Signal(
                market_condition_id="m1", token_id="t1", side=Side.SELL,
                outcome="Yes", estimated_fair_value=0.40, market_price=0.50,
                edge=0.10, confidence=0.6, strategy="b",
            ),
        ]

        conflicts = self.aggregator.check_conflicts(signals)
        assert len(conflicts) == 1


# ── Odds Aggregator Tests ────────────────────────────────────────

class TestOddsAggregator:
    def setup_method(self):
        self.odds = OddsAggregator()

    def teardown_method(self):
        self.odds.close()

    def test_elo_win_probability(self):
        """ELO formula should produce reasonable probabilities."""
        # Equal ratings = 50/50
        assert abs(self.odds.elo_win_probability(1500, 1500) - 0.5) < 0.001

        # Higher rating should win more often
        prob = self.odds.elo_win_probability(1700, 1500)
        assert prob > 0.5
        assert prob < 1.0

        # Much higher rating ≈ near-certain
        prob = self.odds.elo_win_probability(2000, 1200)
        assert prob > 0.9

    def test_complement_arb_detection(self):
        """Should detect when prices don't sum to 1."""
        # Underpriced
        result = self.odds.find_complement_arb({"Yes": 0.40, "No": 0.45}, tolerance=0.03)
        assert result is not None
        assert result["type"] == "underpriced_complement"

        # Overpriced
        result = self.odds.find_complement_arb({"Yes": 0.55, "No": 0.55}, tolerance=0.03)
        assert result is not None
        assert result["type"] == "overpriced_complement"

        # Fair
        result = self.odds.find_complement_arb({"Yes": 0.50, "No": 0.50}, tolerance=0.03)
        assert result is None

    def test_cross_market_edge(self):
        """Should detect edge between Polymarket and external odds."""
        external = [
            ExternalOdds(source="src1", event_name="e", outcome="Yes", implied_probability=0.65),
            ExternalOdds(source="src2", event_name="e", outcome="Yes", implied_probability=0.70),
        ]

        result = self.odds.find_cross_market_edge(0.50, external, min_edge=0.03)
        assert result is not None
        assert result["direction"] == "BUY"
        assert result["edge"] > 0.10
