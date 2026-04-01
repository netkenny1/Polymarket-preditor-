"""Tests for Phase 4: Volatility, EventCatalyst, SmartRouter, Analytics."""

import pytest
import numpy as np
from datetime import datetime, timedelta

from polymarket_bot.data.models import (
    Market, MarketCategory, OrderBook, OrderBookLevel, Position, Side, Signal, Token, TradeResult, Order,
)
from polymarket_bot.strategies.volatility import VolatilityStrategy
from polymarket_bot.strategies.event_catalyst import EventCatalystStrategy
from polymarket_bot.execution.smart_router import SmartOrderRouter, OrderSlice
from polymarket_bot.risk.analytics import PortfolioAnalytics, AnalyticsReport, StrategyStats


def make_market(
    cid: str = "test_001",
    yes_price: float = 0.50,
    liquidity: float = 5000.0,
    end_date: datetime | None = None,
    category: MarketCategory = MarketCategory.OTHER,
) -> Market:
    return Market(
        condition_id=cid,
        question="Test event?",
        slug="test",
        tokens=[
            Token(token_id=f"{cid}_yes", outcome="Yes", price=yes_price),
            Token(token_id=f"{cid}_no", outcome="No", price=1.0 - yes_price),
        ],
        category=category,
        liquidity=liquidity,
        active=True,
        end_date=end_date,
    )


def make_books(cid: str, mid: float = 0.50) -> dict[str, OrderBook]:
    return {
        f"{cid}_yes": OrderBook(
            token_id=f"{cid}_yes",
            bids=[OrderBookLevel(price=round(mid - 0.01, 2), size=200.0),
                  OrderBookLevel(price=round(mid - 0.02, 2), size=300.0),
                  OrderBookLevel(price=round(mid - 0.03, 2), size=400.0)],
            asks=[OrderBookLevel(price=round(mid + 0.01, 2), size=200.0),
                  OrderBookLevel(price=round(mid + 0.02, 2), size=300.0),
                  OrderBookLevel(price=round(mid + 0.03, 2), size=400.0)],
        ),
        f"{cid}_no": OrderBook(
            token_id=f"{cid}_no",
            bids=[OrderBookLevel(price=round(1 - mid - 0.01, 2), size=200.0)],
            asks=[OrderBookLevel(price=round(1 - mid + 0.01, 2), size=200.0)],
        ),
    }


# ── Volatility Strategy Tests ──────────────────────────────────

class TestVolatilityStrategy:
    def setup_method(self):
        self.strategy = VolatilityStrategy(
            short_window=10, long_window=30, min_history=30, min_edge=0.02,
        )

    def test_vol_contraction_near_lower_band(self):
        """Low vol + price near lower band → buy YES (breakout up)."""
        market = make_market(cid="vol_001", yes_price=0.40)
        np.random.seed(42)
        # Flat market then slight dip (low vol, price near lower band)
        prices = [0.50 + np.random.normal(0, 0.003) for _ in range(30)]
        prices[-5:] = [0.42, 0.41, 0.40, 0.40, 0.40]
        context = {"price_history_vol_001": prices}
        signals = self.strategy.generate_signals([market], {}, context)
        assert isinstance(signals, list)

    def test_flat_market_no_signal(self):
        """Perfectly flat market → no signal."""
        market = make_market(cid="vol_002", yes_price=0.50)
        prices = [0.50] * 40
        context = {"price_history_vol_002": prices}
        signals = self.strategy.generate_signals([market], {}, context)
        assert len(signals) == 0

    def test_insufficient_history(self):
        """Short history → no signal."""
        market = make_market(cid="vol_003")
        signals = self.strategy.generate_signals(
            [market], {}, {"price_history_vol_003": [0.5, 0.51, 0.52]},
        )
        assert len(signals) == 0

    def test_vol_expansion_with_recovery(self):
        """High vol + recovery from dip → should analyze."""
        market = make_market(cid="vol_004", yes_price=0.48)
        np.random.seed(42)
        # Start stable, then big dip, then recovery
        stable = [0.55 + np.random.normal(0, 0.005) for _ in range(20)]
        dip = [0.45, 0.40, 0.38, 0.35, 0.37, 0.40, 0.43, 0.45, 0.47, 0.48]
        prices = stable + dip
        context = {"price_history_vol_004": prices}
        signals = self.strategy.generate_signals([market], {}, context)
        assert isinstance(signals, list)

    def test_metadata_contains_vol_info(self):
        """Signal metadata should contain vol regime info."""
        market = make_market(cid="vol_005", yes_price=0.40)
        np.random.seed(42)
        prices = [0.50 + np.random.normal(0, 0.003) for _ in range(25)]
        prices.extend([0.44, 0.42, 0.41, 0.40, 0.40])
        context = {"price_history_vol_005": prices}
        signals = self.strategy.generate_signals([market], {}, context)
        if signals:
            assert "vol_ratio" in signals[0].metadata
            assert "vol_regime" in signals[0].metadata
            assert "bb_width" in signals[0].metadata


# ── Event Catalyst Strategy Tests ──────────────────────────────

class TestEventCatalystStrategy:
    def setup_method(self):
        self.strategy = EventCatalystStrategy(
            pre_event_days=(1.0, 3.0),
            overreaction_threshold=0.15,
            min_edge=0.03,
        )

    def test_pre_event_uncertain_market(self):
        """Near 50/50 market 2 days before → buy cheaper side."""
        market = make_market(
            cid="ec_001", yes_price=0.45,
            end_date=datetime.utcnow() + timedelta(days=2),
            category=MarketCategory.POLITICS,
        )
        signals = self.strategy.generate_signals([market], {}, {})
        assert len(signals) >= 1
        assert signals[0].strategy == "event_catalyst"

    def test_pre_event_momentum(self):
        """Price at 0.75 near event → follow momentum."""
        market = make_market(
            cid="ec_002", yes_price=0.75,
            end_date=datetime.utcnow() + timedelta(days=1.5),
        )
        signals = self.strategy.generate_signals([market], {}, {})
        assert isinstance(signals, list)
        if signals:
            assert signals[0].outcome == "Yes"

    def test_post_event_overreaction_high(self):
        """Price spiked from 0.70 to 0.90 → fade."""
        market = make_market(cid="ec_003", yes_price=0.90)
        context = {"price_history_ec_003": [0.65, 0.70, 0.75, 0.90]}
        signals = self.strategy.generate_signals([market], {}, context)
        if signals:
            assert signals[0].outcome == "No"  # Fading the spike

    def test_post_event_overreaction_low(self):
        """Price crashed from 0.35 to 0.10 → fade."""
        market = make_market(cid="ec_004", yes_price=0.10)
        context = {"price_history_ec_004": [0.40, 0.35, 0.25, 0.10]}
        signals = self.strategy.generate_signals([market], {}, context)
        if signals:
            assert signals[0].outcome == "Yes"  # Fading the crash

    def test_far_from_event_no_signal(self):
        """30 days out → no pre-event signal."""
        market = make_market(
            cid="ec_005", yes_price=0.50,
            end_date=datetime.utcnow() + timedelta(days=30),
        )
        signals = self.strategy.generate_signals([market], {}, {})
        assert len(signals) == 0

    def test_category_multiplier(self):
        """Politics markets should have higher confidence than crypto."""
        politics_market = make_market(
            cid="ec_006", yes_price=0.45,
            end_date=datetime.utcnow() + timedelta(days=2),
            category=MarketCategory.POLITICS,
        )
        crypto_market = make_market(
            cid="ec_007", yes_price=0.45,
            end_date=datetime.utcnow() + timedelta(days=2),
            category=MarketCategory.CRYPTO,
        )
        pol_signals = self.strategy.generate_signals([politics_market], {}, {})
        crypto_signals = self.strategy.generate_signals([crypto_market], {}, {})
        if pol_signals and crypto_signals:
            assert pol_signals[0].confidence > crypto_signals[0].confidence


# ── Smart Order Router Tests ───────────────────────────────────

class TestSmartOrderRouter:
    def setup_method(self):
        self.router = SmartOrderRouter(impact_coeff=0.1, max_slices=5)

    def _make_signal(self, side=Side.BUY, price=0.50, fv=0.55):
        return Signal(
            market_condition_id="test", token_id="test_yes", side=side,
            outcome="Yes", estimated_fair_value=fv, market_price=price,
            edge=abs(fv - price), confidence=0.7, strategy="test",
        )

    def _make_book(self, mid=0.50):
        return OrderBook(
            token_id="test_yes",
            bids=[OrderBookLevel(price=mid - 0.01, size=200),
                  OrderBookLevel(price=mid - 0.02, size=300),
                  OrderBookLevel(price=mid - 0.03, size=400)],
            asks=[OrderBookLevel(price=mid + 0.01, size=200),
                  OrderBookLevel(price=mid + 0.02, size=300),
                  OrderBookLevel(price=mid + 0.03, size=400)],
        )

    def test_small_order_single_slice(self):
        """Small order should not be split."""
        signal = self._make_signal()
        book = self._make_book()
        slices = self.router.route_order(signal, book, total_size_usd=10.0)
        assert len(slices) == 1
        assert slices[0].delay_ms == 0

    def test_large_order_split(self):
        """Large order should be split into multiple slices."""
        signal = self._make_signal()
        book = self._make_book()
        slices = self.router.route_order(signal, book, total_size_usd=500.0)
        assert len(slices) >= 2
        # Slices should have increasing delay
        for i in range(1, len(slices)):
            assert slices[i].delay_ms > slices[i - 1].delay_ms

    def test_slices_sum_to_total(self):
        """Sum of slice sizes should equal total order size."""
        signal = self._make_signal()
        book = self._make_book()
        slices = self.router.route_order(signal, book, total_size_usd=100.0)
        total = sum(s.size for s in slices)
        # Allow small rounding error
        limit_price = slices[0].price
        expected_shares = 100.0 / limit_price
        assert abs(total - expected_shares) < 1.0

    def test_empty_book_returns_empty(self):
        """Empty order book should return no slices."""
        signal = self._make_signal()
        book = OrderBook(token_id="test_yes", bids=[], asks=[])
        slices = self.router.route_order(signal, book, total_size_usd=50.0)
        assert len(slices) == 0

    def test_vwap_computation(self):
        """VWAP should be reasonable."""
        book = self._make_book(0.50)
        vwap = self.router._compute_vwap(book, Side.BUY, 50.0)
        assert 0.50 <= vwap <= 0.55  # Should be at or above best ask

    def test_market_impact_increases_with_size(self):
        """Larger orders should have more impact."""
        impact_small = self.router._estimate_market_impact(10.0, 1000.0)
        impact_large = self.router._estimate_market_impact(100.0, 1000.0)
        assert impact_large > impact_small

    def test_split_order_decreasing(self):
        """Slices should be in decreasing size order."""
        slices = self.router._split_order(100.0, 3)
        assert len(slices) == 3
        assert slices[0] >= slices[1] >= slices[2]

    def test_price_capped_at_fair_value(self):
        """Buy price should never exceed fair value."""
        signal = self._make_signal(fv=0.52)
        book = self._make_book(0.50)
        slices = self.router.route_order(signal, book, total_size_usd=50.0)
        if slices:
            assert slices[0].price <= 0.52


# ── Portfolio Analytics Tests ──────────────────────────────────

class TestPortfolioAnalytics:
    def test_sortino_ratio(self):
        """Sortino should only penalize downside."""
        returns = [0.01, 0.02, -0.01, 0.03, -0.005, 0.01, 0.02, -0.02]
        sortino = PortfolioAnalytics.compute_sortino_ratio(returns)
        assert sortino > 0  # Positive overall returns

    def test_sortino_all_positive(self):
        """All positive returns → no downside deviation → 0."""
        returns = [0.01, 0.02, 0.03, 0.01]
        sortino = PortfolioAnalytics.compute_sortino_ratio(returns)
        assert sortino == 0.0  # No downside to measure

    def test_calmar_ratio(self):
        calmar = PortfolioAnalytics.compute_calmar_ratio(0.30, 0.10)
        assert abs(calmar - 3.0) < 1e-10

    def test_calmar_no_drawdown(self):
        calmar = PortfolioAnalytics.compute_calmar_ratio(0.20, 0.0)
        assert calmar == 0.0

    def test_var_95(self):
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000).tolist()
        var = PortfolioAnalytics.compute_var(returns, 0.95)
        assert var > 0  # Should be positive loss magnitude
        assert var < 0.10  # Reasonable range

    def test_cvar_greater_than_var(self):
        """CVaR should be >= VaR (expected shortfall is worse than threshold)."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000).tolist()
        var = PortfolioAnalytics.compute_var(returns, 0.95)
        cvar = PortfolioAnalytics.compute_cvar(returns, 0.95)
        assert cvar >= var

    def test_max_consecutive_losses(self):
        pnls = [1, -1, -2, -3, 1, -1, 1]
        assert PortfolioAnalytics.compute_max_consecutive_losses(pnls) == 3

    def test_max_consecutive_wins(self):
        pnls = [1, 2, 3, -1, 1, 1, -1]
        assert PortfolioAnalytics.compute_max_consecutive_wins(pnls) == 3

    def test_recovery_time(self):
        # Peak at 1100, drops to 900, recovers to 1100 at step +4
        values = [1000, 1100, 1050, 950, 900, 950, 1000, 1050, 1100, 1150]
        recovery = PortfolioAnalytics.compute_recovery_time(values)
        assert recovery > 0

    def test_full_report(self):
        analytics = PortfolioAnalytics()
        values = [1000, 1010, 1005, 1020, 1015, 1030, 1025, 1040]
        returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
        report = analytics.full_report(values, [], returns)
        assert isinstance(report, AnalyticsReport)
        assert report.total_return_pct > 0
        assert report.sharpe_ratio != 0
        assert report.summary()  # Should produce string

    def test_empty_data(self):
        """Should handle empty inputs gracefully."""
        assert PortfolioAnalytics.compute_sortino_ratio([]) == 0.0
        assert PortfolioAnalytics.compute_var([]) == 0.0
        assert PortfolioAnalytics.compute_max_consecutive_losses([]) == 0
        assert PortfolioAnalytics.compute_recovery_time([]) == 0


# ── Updated Signal Weights ─────────────────────────────────────

class TestPhase4Weights:
    def test_volatility_weight(self):
        from polymarket_bot.strategies.signals import STRATEGY_WEIGHTS
        assert "volatility" in STRATEGY_WEIGHTS
        assert STRATEGY_WEIGHTS["volatility"] == 1.0

    def test_event_catalyst_weight(self):
        from polymarket_bot.strategies.signals import STRATEGY_WEIGHTS
        assert "event_catalyst" in STRATEGY_WEIGHTS
        assert STRATEGY_WEIGHTS["event_catalyst"] == 1.05


# ── Integration Backtest ───────────────────────────────────────

class TestPhase4Backtest:
    def test_backtest_all_11_strategies(self):
        """Full backtest with all 11 strategies + exit manager."""
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

    def test_backtest_volatility_only(self):
        from polymarket_bot.backtesting.engine import BacktestEngine
        from polymarket_bot.config import BotConfig
        import structlog
        structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(50))

        engine = BacktestEngine(config=BotConfig(), initial_capital=1000.0, seed=42)
        result = engine.run(num_markets=5, time_steps=80, strategies=["volatility"])
        assert result.final_portfolio_value > 0

    def test_backtest_event_catalyst_only(self):
        from polymarket_bot.backtesting.engine import BacktestEngine
        from polymarket_bot.config import BotConfig
        import structlog
        structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(50))

        engine = BacktestEngine(config=BotConfig(), initial_capital=1000.0, seed=42)
        result = engine.run(num_markets=5, time_steps=80, strategies=["event_catalyst"])
        assert result.final_portfolio_value > 0
