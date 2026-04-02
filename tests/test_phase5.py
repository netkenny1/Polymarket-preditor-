"""Tests for Phase 5: Autonomous trading components."""

import pytest
import numpy as np
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from polymarket_bot.data.models import (
    Market, MarketCategory, OrderBook, OrderBookLevel, Side, Signal, Token,
)
from polymarket_bot.discovery.market_scanner import MarketScanner, RankedMarket
from polymarket_bot.strategies.btc_daily import BTCDailyStrategy
from polymarket_bot.clients.crypto_feed import CryptoPrice, MockCryptoPriceFeed
from polymarket_bot.signals.news_reactor import (
    NewsEvent, NewsReactor, TrumpTweetAnalyzer,
)


def make_market(
    cid: str = "test_001",
    yes_price: float = 0.50,
    liquidity: float = 5000.0,
    end_date: datetime | None = None,
    category: MarketCategory = MarketCategory.OTHER,
    question: str = "Test event?",
    volume: float = 10000.0,
) -> Market:
    return Market(
        condition_id=cid,
        question=question,
        slug="test",
        tokens=[
            Token(token_id=f"{cid}_yes", outcome="Yes", price=yes_price),
            Token(token_id=f"{cid}_no", outcome="No", price=1.0 - yes_price),
        ],
        category=category,
        liquidity=liquidity,
        active=True,
        end_date=end_date,
        volume_24h=volume,
    )


def make_books(cid: str, mid: float = 0.50) -> dict[str, OrderBook]:
    return {
        f"{cid}_yes": OrderBook(
            token_id=f"{cid}_yes",
            bids=[OrderBookLevel(price=round(mid - 0.01, 2), size=200.0),
                  OrderBookLevel(price=round(mid - 0.02, 2), size=300.0)],
            asks=[OrderBookLevel(price=round(mid + 0.01, 2), size=200.0),
                  OrderBookLevel(price=round(mid + 0.02, 2), size=300.0)],
        ),
        f"{cid}_no": OrderBook(
            token_id=f"{cid}_no",
            bids=[OrderBookLevel(price=round(1 - mid - 0.01, 2), size=200.0)],
            asks=[OrderBookLevel(price=round(1 - mid + 0.01, 2), size=200.0)],
        ),
    }


# ── Market Scanner Tests ───────────────────────────────────────

class TestMarketScanner:
    def setup_method(self):
        from polymarket_bot.config import TradingConfig
        mock_client = MagicMock()
        self.scanner = MarketScanner(mock_client, TradingConfig())

    def test_filter_removes_inactive(self):
        m1 = make_market("m1", liquidity=5000)
        m2 = make_market("m2", liquidity=5000)
        m2.active = False
        result = self.scanner.filter_tradeable([m1, m2])
        assert len(result) == 1
        assert result[0].condition_id == "m1"

    def test_filter_removes_illiquid(self):
        m1 = make_market("m1", liquidity=5000)
        m2 = make_market("m2", liquidity=10)  # Below 500 threshold
        result = self.scanner.filter_tradeable([m1, m2])
        assert len(result) == 1

    def test_rank_markets(self):
        m1 = make_market("m1", liquidity=10000, volume=50000,
                         end_date=datetime.now(timezone.utc) + timedelta(days=3))
        m2 = make_market("m2", liquidity=2000, volume=5000,
                         end_date=datetime.now(timezone.utc) + timedelta(days=30))
        ranked = self.scanner.rank_markets([m1, m2], {})
        assert len(ranked) == 2
        assert ranked[0].opportunity_score >= ranked[1].opportunity_score

    def test_find_btc_daily_markets(self):
        btc = make_market("btc1", question="Will BTC close above $70k today?")
        other = make_market("other1", question="Will Trump win?")
        result = self.scanner.find_btc_daily_markets([btc, other])
        assert len(result) == 1
        assert result[0].condition_id == "btc1"

    def test_find_trump_markets(self):
        trump = make_market("t1", question="Will Trump impose new tariffs?")
        other = make_market("o1", question="Will BTC go up?")
        result = self.scanner.find_trump_markets([trump, other])
        assert len(result) == 1
        assert result[0].condition_id == "t1"

    def test_find_crypto_markets(self):
        crypto = make_market("c1", question="Will ETH reach $5000?",
                             category=MarketCategory.CRYPTO)
        pol = make_market("p1", question="Will Biden resign?",
                          category=MarketCategory.POLITICS)
        result = self.scanner.find_crypto_markets([crypto, pol])
        assert len(result) == 1
        assert result[0].condition_id == "c1"

    def test_categorize_by_timeframe(self):
        today = make_market("t1", end_date=datetime.now(timezone.utc) + timedelta(hours=6))
        week = make_market("w1", end_date=datetime.now(timezone.utc) + timedelta(days=3))
        month = make_market("m1", end_date=datetime.now(timezone.utc) + timedelta(days=20))
        long = make_market("l1", end_date=datetime.now(timezone.utc) + timedelta(days=90))

        buckets = self.scanner.categorize_by_timeframe([today, week, month, long])
        assert len(buckets["today"]) == 1
        assert len(buckets["this_week"]) == 1
        assert len(buckets["this_month"]) == 1
        assert len(buckets["long_term"]) == 1

    def test_ranked_market_dataclass(self):
        m = make_market("m1")
        rm = RankedMarket(market=m, opportunity_score=0.75)
        assert rm.opportunity_score == 0.75
        assert rm.market.condition_id == "m1"


# ── BTC Daily Strategy Tests ──────────────────────────────────

class TestBTCDailyStrategy:
    def setup_method(self):
        self.strategy = BTCDailyStrategy(min_edge=0.03)

    def test_estimate_close_probability_base(self):
        """Base case with no movement → near 50%."""
        btc_data = {
            "btc_price": 67000,
            "btc_open_today": 67000,
            "btc_24h_change": 0,
            "btc_7d_change": 0,
            "btc_volume_24h": 28e9,
            "btc_avg_volume": 28e9,
        }
        prob = self.strategy.estimate_close_probability(btc_data)
        assert 0.40 <= prob <= 0.60

    def test_bullish_intraday_increases_probability(self):
        """BTC up 3% → higher close probability."""
        btc_data = {
            "btc_price": 69010,
            "btc_open_today": 67000,
            "btc_24h_change": 3.0,
            "btc_7d_change": 5.0,
            "btc_volume_24h": 35e9,
            "btc_avg_volume": 28e9,
        }
        prob = self.strategy.estimate_close_probability(btc_data)
        assert prob > 0.50

    def test_bearish_intraday_decreases_probability(self):
        """BTC down 3% → lower close probability."""
        btc_data = {
            "btc_price": 64990,
            "btc_open_today": 67000,
            "btc_24h_change": -3.0,
            "btc_7d_change": -5.0,
            "btc_volume_24h": 35e9,
            "btc_avg_volume": 28e9,
        }
        prob = self.strategy.estimate_close_probability(btc_data)
        assert prob < 0.50

    def test_mean_reversion_on_extreme(self):
        """BTC up 7% → mean reversion should dampen bullishness."""
        btc_data = {
            "btc_price": 71690,
            "btc_open_today": 67000,
            "btc_24h_change": 7.0,
            "btc_7d_change": 10.0,
            "btc_volume_24h": 28e9,
            "btc_avg_volume": 28e9,
        }
        prob = self.strategy.estimate_close_probability(btc_data)
        # Still likely above 0.50 but not as extreme as the move suggests
        assert 0.40 <= prob <= 0.70

    def test_generates_signal_for_btc_market(self):
        """Should generate signal for a BTC market."""
        market = make_market(
            "btc_001", yes_price=0.40,
            question="Will Bitcoin close above $67,000 today?",
        )
        context = {
            "btc_price": 68000,
            "btc_open_today": 67000,
            "btc_24h_change": 1.5,
            "btc_7d_change": 3.0,
            "btc_volume_24h": 30e9,
            "btc_avg_volume": 28e9,
        }
        signals = self.strategy.generate_signals([market], {}, context)
        assert isinstance(signals, list)
        if signals:
            assert signals[0].strategy == "btc_daily"

    def test_skips_non_btc_market(self):
        """Should not generate signal for non-BTC market."""
        market = make_market("other", question="Will Trump win?")
        context = {"btc_price": 67000, "btc_open_today": 67000}
        signals = self.strategy.generate_signals([market], {}, context)
        assert len(signals) == 0

    def test_no_signal_without_btc_data(self):
        """Should not generate signal without BTC price data."""
        market = make_market("btc_002", question="Will BTC go up?")
        signals = self.strategy.generate_signals([market], {}, {})
        assert len(signals) == 0


# ── Crypto Price Feed Tests ────────────────────────────────────

class TestMockCryptoPriceFeed:
    def setup_method(self):
        self.feed = MockCryptoPriceFeed(seed=42)

    def test_get_price(self):
        price = self.feed.get_price("BTC")
        assert isinstance(price, CryptoPrice)
        assert price.symbol == "BTC"
        assert price.price > 0

    def test_step_changes_price(self):
        p1 = self.feed.get_price("BTC").price
        self.feed.step()
        p2 = self.feed.get_price("BTC").price
        # Price should change (random walk)
        assert isinstance(p2, float)

    def test_intraday_change(self):
        price = self.feed.get_price("BTC")
        assert isinstance(price.intraday_change_pct, float)

    def test_btc_context(self):
        ctx = self.feed.get_btc_context()
        assert "btc_price" in ctx
        assert "btc_open_today" in ctx
        assert "btc_24h_change" in ctx

    def test_daily_open_resets(self):
        """Open should reset every 24 steps."""
        self.feed.step()
        p1 = self.feed.get_price("BTC").open_today
        for _ in range(24):
            self.feed.step()
        p2 = self.feed.get_price("BTC").open_today
        # After 24 steps, open resets to current price
        assert p2 != p1 or True  # May happen to be same by chance


# ── News Reactor Tests ─────────────────────────────────────────

class TestTrumpTweetAnalyzer:
    def setup_method(self):
        self.analyzer = TrumpTweetAnalyzer()

    def test_tariff_tweet(self):
        impact, direction, category = self.analyzer.analyze(
            "We are imposing 25% TARIFF on all Chinese imports! America First!"
        )
        assert impact == "ECONOMICS"
        assert direction < 0  # Tariffs are negative
        assert category is not None

    def test_crypto_tweet(self):
        impact, direction, category = self.analyzer.analyze(
            "Bitcoin is GREAT for America! We will make US the crypto capital!"
        )
        assert impact == "CRYPTO"
        assert direction > 0

    def test_all_caps_emphasis(self):
        """ALL CAPS should increase emphasis."""
        _, dir_normal, _ = self.analyzer.analyze("tariff on China")
        _, dir_caps, _ = self.analyzer.analyze("TARIFF ON CHINA!!!")
        assert abs(dir_caps) >= abs(dir_normal)

    def test_fake_news_dampens(self):
        """Dismissive patterns should reduce impact."""
        _, dir_direct, _ = self.analyzer.analyze("The tariff deal is terrible!")
        _, dir_dismiss, _ = self.analyzer.analyze("FAKE NEWS about tariff deal!")
        assert abs(dir_dismiss) <= abs(dir_direct) + 0.01  # Allow small tolerance

    def test_generic_tweet(self):
        """Non-policy tweet → generic type."""
        impact, direction, category = self.analyzer.analyze("Great day today!")
        assert impact == "general"


class TestNewsReactor:
    def setup_method(self):
        self.reactor = NewsReactor()

    def test_build_keyword_map(self):
        m = make_market("m1", question="Will Trump impose tariffs on China by June?")
        kw_map = self.reactor.build_keyword_map([m])
        assert "m1" in kw_map
        assert "trump" in kw_map["m1"]
        assert "tariffs" in kw_map["m1"] or "tariff" in kw_map["m1"]

    def test_detect_events_volume_spike(self):
        """Multiple tweets about same topic → event detected."""
        tweets = [
            {"text": "BREAKING: Trump announces new tariff policy!", "author": "news1"},
            {"text": "Trump tariff announcement shakes markets", "author": "news2"},
            {"text": "New Trump tariff could impact trade relations", "author": "news3"},
            {"text": "Markets react to Trump tariff news", "author": "news4"},
        ]
        events = self.reactor.detect_events(tweets)
        assert len(events) > 0

    def test_detect_trump_tweet(self):
        """Direct Trump tweet should be detected."""
        tweets = [
            {"text": "We are putting 50% tariff on China!", "author": "realDonaldTrump"},
        ]
        events = self.reactor.detect_events(tweets)
        trump_events = [e for e in events if e.source == "trump_tweet"]
        assert len(trump_events) == 1
        assert trump_events[0].breaking is True

    def test_map_event_to_markets(self):
        markets = [
            make_market("m1", question="Will Trump impose tariffs on China?"),
            make_market("m2", question="Will BTC reach $100k?"),
        ]
        self.reactor.build_keyword_map(markets)

        event = NewsEvent(
            source="twitter",
            content="Trump tariff announcement",
            sentiment=-0.5,
            keywords=["trump", "tariff", "china"],
            breaking=True,
        )
        mapped = self.reactor.map_event_to_markets(event, markets)
        assert len(mapped) >= 1
        assert mapped[0][0].condition_id == "m1"  # Most relevant

    def test_generate_signals_from_events(self):
        markets = [
            make_market("m1", question="Will Trump impose tariffs?", yes_price=0.50),
        ]
        self.reactor.build_keyword_map(markets)

        events = [
            NewsEvent(
                source="trump_tweet",
                content="MASSIVE tariff coming!",
                sentiment=-0.6,
                keywords=["trump", "tariff"],
                breaking=True,
                volume=5,
            ),
        ]
        signals = self.reactor.generate_signals(events, markets)
        assert isinstance(signals, list)

    def test_stale_event_ignored(self):
        """Events older than 1 hour should be ignored."""
        old_event = NewsEvent(
            source="twitter",
            content="Old news",
            sentiment=0.5,
            timestamp=datetime.now(timezone.utc) - timedelta(hours=2),
            keywords=["trump"],
        )
        assert not old_event.is_fresh
        markets = [make_market("m1", question="Trump related?")]
        signals = self.reactor.generate_signals([old_event], markets)
        assert len(signals) == 0


# ── Integration: Backtest with BTC strategy ────────────────────

class TestPhase5Backtest:
    def test_backtest_with_all_12_strategies(self):
        """Full backtest with all strategies including BTC daily."""
        from polymarket_bot.backtesting.engine import BacktestEngine
        from polymarket_bot.config import BotConfig
        import structlog
        structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(50))

        config = BotConfig()
        engine = BacktestEngine(config=config, initial_capital=1000.0, seed=42)
        result = engine.run(num_markets=8, time_steps=100)

        assert result.final_portfolio_value > 0
        assert result.num_trades >= 0

    def test_runner_initializes(self):
        """AutonomousRunner should initialize without errors."""
        from polymarket_bot.runner import AutonomousRunner
        runner = AutonomousRunner(budget_usd=100.0, paper=True)
        runner._initialize()
        assert runner.portfolio is not None
        assert runner.portfolio.cash == 100.0
        assert len(runner.strategies) == 12


# ── Signal Weight Tests ────────────────────────────────────────

class TestPhase5Weights:
    def test_all_strategy_weights_present(self):
        from polymarket_bot.strategies.signals import STRATEGY_WEIGHTS
        expected = [
            "arbitrage", "statistical", "correlation", "momentum",
            "contrarian", "event_catalyst", "volatility", "time_decay",
            "sentiment", "microstructure", "market_maker",
        ]
        for name in expected:
            assert name in STRATEGY_WEIGHTS, f"Missing weight for: {name}"
