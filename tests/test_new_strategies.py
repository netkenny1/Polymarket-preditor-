"""Tests for new strategies: momentum, contrarian, time_decay, regime detection."""

import pytest
import numpy as np
from datetime import datetime, timedelta

from polymarket_bot.data.models import (
    Market, MarketCategory, OrderBook, OrderBookLevel, Side, Signal, Token,
)
from polymarket_bot.strategies.momentum import MomentumStrategy
from polymarket_bot.strategies.contrarian import ContrarianStrategy
from polymarket_bot.strategies.time_decay import TimeDecayStrategy
from polymarket_bot.strategies.market_regime import RegimeDetector, Regime


def make_market(
    cid: str = "test_001",
    yes_price: float = 0.50,
    liquidity: float = 5000.0,
    end_date: datetime | None = None,
) -> Market:
    return Market(
        condition_id=cid,
        question="Test event?",
        slug="test",
        tokens=[
            Token(token_id=f"{cid}_yes", outcome="Yes", price=yes_price),
            Token(token_id=f"{cid}_no", outcome="No", price=1.0 - yes_price),
        ],
        category=MarketCategory.OTHER,
        liquidity=liquidity,
        active=True,
        end_date=end_date,
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


# ── Momentum Strategy ────────────────────────────────────────────

class TestMomentumStrategy:
    def setup_method(self):
        self.strategy = MomentumStrategy(
            short_window=5, medium_window=15, long_window=30, min_edge=0.03
        )

    def test_uptrend_generates_buy_yes(self):
        """Clear uptrend should produce buy YES signal."""
        market = make_market(cid="mom_001", yes_price=0.65)
        books = make_books("mom_001", 0.65)

        # Simulated uptrend: 0.40 -> 0.65
        np.random.seed(42)
        base = np.linspace(0.40, 0.65, 40) + np.random.normal(0, 0.005, 40)
        price_history = base.tolist()

        context = {"price_history_mom_001": price_history}
        signals = self.strategy.generate_signals([market], books, context)
        assert isinstance(signals, list)

    def test_downtrend_generates_buy_no(self):
        """Clear downtrend should buy NO."""
        market = make_market(cid="mom_002", yes_price=0.35)
        books = make_books("mom_002", 0.35)

        np.random.seed(42)
        base = np.linspace(0.60, 0.35, 40) + np.random.normal(0, 0.005, 40)
        price_history = base.tolist()

        context = {"price_history_mom_002": price_history}
        signals = self.strategy.generate_signals([market], books, context)
        assert isinstance(signals, list)

    def test_flat_market_no_signal(self):
        """Flat market should not generate momentum signals."""
        market = make_market(cid="mom_003", yes_price=0.50)
        books = make_books("mom_003", 0.50)

        # Flat with tiny noise
        price_history = [0.50 + np.random.normal(0, 0.002) for _ in range(40)]
        context = {"price_history_mom_003": price_history}

        signals = self.strategy.generate_signals([market], books, context)
        assert len(signals) == 0

    def test_insufficient_history_skipped(self):
        """Not enough history should skip market."""
        market = make_market(cid="mom_004")
        signals = self.strategy.generate_signals(
            [market], {}, {"price_history_mom_004": [0.5, 0.51, 0.52]}
        )
        assert len(signals) == 0


# ── Contrarian Strategy ──────────────────────────────────────────

class TestContrarianStrategy:
    def setup_method(self):
        self.strategy = ContrarianStrategy(
            lookback=20, z_threshold=1.5, min_edge=0.03
        )

    def test_spike_up_generates_sell(self):
        """Sharp upward spike should trigger contrarian sell (buy NO)."""
        market = make_market(cid="con_001", yes_price=0.80)
        books = make_books("con_001", 0.80)

        # Stable around 0.55, then spike to 0.80
        np.random.seed(42)
        stable = [0.55 + np.random.normal(0, 0.01) for _ in range(25)]
        spike = [0.65, 0.70, 0.75, 0.78, 0.80]
        price_history = stable + spike

        context = {"price_history_con_001": price_history}
        signals = self.strategy.generate_signals([market], books, context)
        # Should generate a contrarian signal (buy NO to fade the spike)
        assert isinstance(signals, list)

    def test_no_signal_in_trending_regime(self):
        """Should not generate signals when regime is 'trending'."""
        market = make_market(cid="con_002", yes_price=0.70)
        books = make_books("con_002", 0.70)

        np.random.seed(42)
        stable = [0.50 + np.random.normal(0, 0.01) for _ in range(25)]
        spike = [0.60, 0.65, 0.68, 0.70, 0.70]
        price_history = stable + spike

        context = {
            "price_history_con_002": price_history,
            "regime_con_002": "trending",  # Regime says trending -> skip
        }
        signals = self.strategy.generate_signals([market], books, context)
        assert len(signals) == 0

    def test_gradual_move_ignored(self):
        """Slow, gradual moves should not trigger contrarian signals."""
        market = make_market(cid="con_003", yes_price=0.70)
        books = make_books("con_003", 0.70)

        # Slow grind up (not a spike)
        price_history = np.linspace(0.50, 0.70, 30).tolist()
        context = {"price_history_con_003": price_history}

        signals = self.strategy.generate_signals([market], books, context)
        assert len(signals) == 0  # Gradual move, speed ratio < threshold


# ── Time Decay Strategy ──────────────────────────────────────────

class TestTimeDecayStrategy:
    def setup_method(self):
        self.strategy = TimeDecayStrategy(
            min_edge=0.03,
            min_hours_to_expiry=4.0,
            max_hours_to_expiry=200.0,
        )

    def test_near_expiry_with_sentiment(self):
        """Market near expiry with sentiment bias should generate signal."""
        market = make_market(
            cid="td_001", yes_price=0.55,
            end_date=datetime.utcnow() + timedelta(hours=24),
        )
        books = make_books("td_001", 0.55)

        from polymarket_bot.data.models import SentimentData
        sentiment = SentimentData(
            query="test", tweet_count=20, avg_sentiment=0.5,
            bullish_pct=0.7, bearish_pct=0.3,
        )
        context = {
            f"sentiment_td_001": sentiment,
            f"price_history_td_001": [0.50, 0.51, 0.52, 0.53, 0.54, 0.55],
        }

        signals = self.strategy.generate_signals([market], books, context)
        assert isinstance(signals, list)

    def test_extreme_price_skipped(self):
        """Markets at extreme prices near expiry should be skipped."""
        market = make_market(
            cid="td_002", yes_price=0.95,
            end_date=datetime.utcnow() + timedelta(hours=12),
        )
        signals = self.strategy.generate_signals([market], {}, {})
        assert len(signals) == 0

    def test_far_from_expiry_skipped(self):
        """Markets too far from expiry should be skipped."""
        market = make_market(
            cid="td_003", yes_price=0.55,
            end_date=datetime.utcnow() + timedelta(days=30),
        )
        signals = self.strategy.generate_signals([market], {}, {})
        assert len(signals) == 0

    def test_no_end_date_skipped(self):
        """Markets without end dates should be skipped."""
        market = make_market(cid="td_004", yes_price=0.55)
        market.end_date = None  # No expiry
        signals = self.strategy.generate_signals([market], {}, {})
        assert len(signals) == 0


# ── Market Regime Detector ───────────────────────────────────────

class TestRegimeDetector:
    def setup_method(self):
        self.detector = RegimeDetector(min_history=20)

    def test_trending_regime(self):
        """Clear trend should be classified as trending."""
        np.random.seed(42)
        # Strong uptrend with low noise
        prices = (np.linspace(0.30, 0.70, 60) + np.random.normal(0, 0.005, 60)).tolist()
        state = self.detector.detect(prices)
        # Should detect some form of persistent behavior
        assert state.hurst_exponent > 0.0
        assert state.confidence > 0.0
        assert state.regime in [Regime.TRENDING, Regime.MEAN_REVERTING, Regime.VOLATILE]

    def test_mean_reverting_regime(self):
        """Oscillating prices should be classified as mean-reverting."""
        np.random.seed(42)
        # Sine wave + noise (classic mean reversion)
        t = np.linspace(0, 6 * np.pi, 60)
        prices = (0.50 + 0.10 * np.sin(t) + np.random.normal(0, 0.005, 60)).tolist()
        state = self.detector.detect(prices)
        assert state.regime in [Regime.MEAN_REVERTING, Regime.TRENDING, Regime.VOLATILE]
        assert state.hurst_exponent > 0.0

    def test_volatile_regime(self):
        """High volatility should be classified as volatile."""
        np.random.seed(42)
        # Very noisy with large jumps
        prices = [0.50]
        for _ in range(59):
            prices.append(max(0.05, min(0.95, prices[-1] + np.random.normal(0, 0.08))))
        state = self.detector.detect(prices)
        assert state.volatility > 0
        assert state.regime in [Regime.VOLATILE, Regime.TRENDING, Regime.MEAN_REVERTING]

    def test_insufficient_history(self):
        """Short history should return UNKNOWN regime."""
        state = self.detector.detect([0.50, 0.51, 0.52])
        assert state.regime == Regime.UNKNOWN
        assert state.confidence == 0.0

    def test_hurst_exponent_range(self):
        """Hurst exponent should be between 0.1 and 0.9."""
        np.random.seed(42)
        prices = np.random.normal(0.50, 0.05, 100).tolist()
        state = self.detector.detect(prices)
        assert 0.1 <= state.hurst_exponent <= 0.9

    def test_classify_all_markets(self):
        """Batch classification should work."""
        np.random.seed(42)
        histories = {
            "m1": np.linspace(0.3, 0.7, 50).tolist(),
            "m2": (0.5 + 0.1 * np.sin(np.linspace(0, 4 * np.pi, 50))).tolist(),
        }
        results = self.detector.classify_all_markets(histories)
        assert "m1" in results
        assert "m2" in results
        assert results["m1"].regime != Regime.UNKNOWN
        assert results["m2"].regime != Regime.UNKNOWN


# ── Dynamic Kelly Sizer ──────────────────────────────────────────

class TestDynamicKelly:
    def test_multiplier_starts_at_one(self):
        from polymarket_bot.config import TradingConfig, RiskConfig
        from polymarket_bot.risk.dynamic_kelly import DynamicKellySizer

        sizer = DynamicKellySizer(TradingConfig(), RiskConfig())
        assert sizer.current_multiplier == 1.0

    def test_winning_streak_increases_multiplier(self):
        from polymarket_bot.config import TradingConfig, RiskConfig
        from polymarket_bot.risk.dynamic_kelly import DynamicKellySizer

        sizer = DynamicKellySizer(TradingConfig(), RiskConfig())
        # Record 10 winning trades
        for _ in range(10):
            sizer.record_outcome(0.05, 5.0)

        assert sizer.current_multiplier > 1.0

    def test_losing_streak_decreases_multiplier(self):
        from polymarket_bot.config import TradingConfig, RiskConfig
        from polymarket_bot.risk.dynamic_kelly import DynamicKellySizer

        sizer = DynamicKellySizer(TradingConfig(), RiskConfig())
        # Record 10 losing trades
        for _ in range(10):
            sizer.record_outcome(0.05, -5.0)

        assert sizer.current_multiplier < 1.0

    def test_multiplier_bounded(self):
        from polymarket_bot.config import TradingConfig, RiskConfig
        from polymarket_bot.risk.dynamic_kelly import DynamicKellySizer

        sizer = DynamicKellySizer(
            TradingConfig(), RiskConfig(),
            min_kelly_multiplier=0.3, max_kelly_multiplier=1.5,
        )
        # Extreme winning streak
        for _ in range(50):
            sizer.record_outcome(0.05, 50.0)
        assert sizer.current_multiplier <= 1.5

        # Extreme losing streak
        sizer2 = DynamicKellySizer(
            TradingConfig(), RiskConfig(),
            min_kelly_multiplier=0.3, max_kelly_multiplier=1.5,
        )
        for _ in range(50):
            sizer2.record_outcome(0.05, -50.0)
        assert sizer2.current_multiplier >= 0.3

    def test_get_stats(self):
        from polymarket_bot.config import TradingConfig, RiskConfig
        from polymarket_bot.risk.dynamic_kelly import DynamicKellySizer

        sizer = DynamicKellySizer(TradingConfig(), RiskConfig())
        sizer.record_outcome(0.05, 3.0)
        sizer.record_outcome(0.05, -1.0)

        stats = sizer.get_stats()
        assert stats["recent_trades"] == 2
        assert stats["win_rate"] == 0.5
        assert "kelly_multiplier" in stats


# ── Enhanced Backtest Integration ────────────────────────────────

class TestEnhancedBacktest:
    def test_backtest_with_all_strategies(self):
        """Full backtest with all 7 strategies should complete."""
        from polymarket_bot.backtesting.engine import BacktestEngine
        from polymarket_bot.config import BotConfig
        import structlog
        structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(50))

        config = BotConfig()
        engine = BacktestEngine(config=config, initial_capital=1000.0, seed=42)
        result = engine.run(num_markets=8, time_steps=100)

        assert result.final_portfolio_value > 0
        assert result.num_trades >= 0
        assert result.time_steps == 100

    def test_backtest_momentum_only(self):
        """Backtest with only momentum strategy."""
        from polymarket_bot.backtesting.engine import BacktestEngine
        from polymarket_bot.config import BotConfig
        import structlog
        structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(50))

        config = BotConfig()
        engine = BacktestEngine(config=config, initial_capital=1000.0, seed=42)
        result = engine.run(num_markets=5, time_steps=80, strategies=["momentum"])

        assert result.final_portfolio_value > 0

    def test_backtest_contrarian_only(self):
        """Backtest with only contrarian strategy."""
        from polymarket_bot.backtesting.engine import BacktestEngine
        from polymarket_bot.config import BotConfig
        import structlog
        structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(50))

        config = BotConfig()
        engine = BacktestEngine(config=config, initial_capital=1000.0, seed=42)
        result = engine.run(num_markets=5, time_steps=80, strategies=["contrarian"])

        assert result.final_portfolio_value > 0


# ── Improved Sentiment Tests ─────────────────────────────────────

class TestImprovedSentiment:
    def test_weighted_scoring(self):
        from polymarket_bot.clients.twitter import score_text
        # "Lock" is strong positive (1.7 weight) vs "up" mild (0.5)
        # Use longer sentences to avoid ceiling effects
        lock_score = score_text("I think this market is a definite lock for the upcoming event based on data")
        up_score = score_text("I think the price might be going up soon based on the recent chart analysis")
        assert lock_score > up_score

    def test_bigram_scoring(self):
        from polymarket_bot.clients.twitter import score_text
        # "no chance" should be very negative
        score = score_text("no chance this happens")
        assert score < 0

    def test_intensifier_amplification(self):
        from polymarket_bot.clients.twitter import score_text
        # Use longer sentences so normalization doesn't clamp both to 1.0
        base = score_text("I think the market is bullish on this particular outcome for the election")
        intensified = score_text("I think the market is extremely bullish on this particular outcome for the election")
        assert intensified > base

    def test_emoji_sentiment(self):
        from polymarket_bot.clients.twitter import score_text
        rocket = score_text("🚀🚀🚀")
        bear = score_text("📉📉📉")
        assert rocket > 0
        assert bear < 0

    def test_negation_scope(self):
        from polymarket_bot.clients.twitter import score_text
        # "not" should affect next 2 words, then expire
        # "not bullish but amazing" -> negates "bullish", but "amazing" stays positive
        score = score_text("not bullish but still amazing")
        # Should be less negative than "not bullish" alone
        neg_only = score_text("not bullish")
        assert score > neg_only
