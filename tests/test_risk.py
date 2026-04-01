"""Tests for risk management and position sizing."""

import pytest

from polymarket_bot.config import RiskConfig, TradingConfig
from polymarket_bot.data.models import (
    Order,
    OrderStatus,
    Position,
    Side,
    Signal,
    TradeResult,
)
from polymarket_bot.risk.manager import RiskManager
from polymarket_bot.risk.portfolio import Portfolio
from polymarket_bot.risk.position_sizer import PositionSizer


# ── Position Sizer Tests ─────────────────────────────────────────

class TestPositionSizer:
    def setup_method(self):
        self.trading = TradingConfig(
            kelly_fraction=0.25,
            max_single_position_usd=100.0,
            max_portfolio_exposure_usd=1000.0,
        )
        self.risk = RiskConfig(position_limit_per_market_pct=0.10)
        self.sizer = PositionSizer(self.trading, self.risk)

    def test_kelly_positive_edge(self):
        """Positive edge should produce positive Kelly fraction."""
        signal = Signal(
            market_condition_id="m1", token_id="t1", side=Side.BUY,
            outcome="Yes", estimated_fair_value=0.65, market_price=0.50,
            edge=0.15, confidence=0.8, strategy="test",
        )
        kelly = self.sizer.calculate_kelly_fraction(signal)
        assert kelly > 0
        assert kelly <= 0.25  # Capped at max

    def test_kelly_no_edge(self):
        """No edge should produce zero Kelly."""
        signal = Signal(
            market_condition_id="m1", token_id="t1", side=Side.BUY,
            outcome="Yes", estimated_fair_value=0.50, market_price=0.50,
            edge=0.0, confidence=0.5, strategy="test",
        )
        kelly = self.sizer.calculate_kelly_fraction(signal)
        assert kelly == 0.0

    def test_kelly_negative_edge(self):
        """Negative edge should produce zero Kelly (don't bet)."""
        signal = Signal(
            market_condition_id="m1", token_id="t1", side=Side.BUY,
            outcome="Yes", estimated_fair_value=0.40, market_price=0.50,
            edge=-0.10, confidence=0.5, strategy="test",
        )
        kelly = self.sizer.calculate_kelly_fraction(signal)
        assert kelly == 0.0

    def test_position_size_respects_max(self):
        """Position size should not exceed max single position."""
        signal = Signal(
            market_condition_id="m1", token_id="t1", side=Side.BUY,
            outcome="Yes", estimated_fair_value=0.90, market_price=0.50,
            edge=0.40, confidence=1.0, strategy="test",
        )
        size = self.sizer.calculate_position_size(
            signal, portfolio_value=10000.0, current_exposure=0.0
        )
        assert size <= self.trading.max_single_position_usd

    def test_position_size_scales_with_confidence(self):
        """Lower confidence should produce smaller position."""
        high_conf = Signal(
            market_condition_id="m1", token_id="t1", side=Side.BUY,
            outcome="Yes", estimated_fair_value=0.65, market_price=0.50,
            edge=0.15, confidence=0.9, strategy="test",
        )
        low_conf = Signal(
            market_condition_id="m1", token_id="t1", side=Side.BUY,
            outcome="Yes", estimated_fair_value=0.65, market_price=0.50,
            edge=0.15, confidence=0.3, strategy="test",
        )

        size_high = self.sizer.calculate_position_size(high_conf, 1000.0, 0.0)
        size_low = self.sizer.calculate_position_size(low_conf, 1000.0, 0.0)

        assert size_high > size_low

    def test_zero_portfolio_value(self):
        """Zero portfolio should produce zero size."""
        signal = Signal(
            market_condition_id="m1", token_id="t1", side=Side.BUY,
            outcome="Yes", estimated_fair_value=0.65, market_price=0.50,
            edge=0.15, confidence=0.8, strategy="test",
        )
        assert self.sizer.calculate_position_size(signal, 0.0, 0.0) == 0.0

    def test_shares_calculation(self):
        """Dollar to shares conversion should be correct."""
        shares = self.sizer.calculate_size_in_shares(100.0, 0.50)
        assert shares == 200.0

        shares = self.sizer.calculate_size_in_shares(100.0, 0.25)
        assert shares == 400.0


# ── Portfolio Tests ──────────────────────────────────────────────

class TestPortfolio:
    def setup_method(self):
        self.portfolio = Portfolio(initial_cash=1000.0)

    def _make_fill(self, token_id: str, side: Side, price: float, size: float) -> TradeResult:
        order = Order(
            order_id="test_order",
            market_condition_id="m1",
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            status=OrderStatus.FILLED,
            filled_size=size,
            strategy="test",
        )
        return TradeResult(order=order, success=True, fill_price=price, fill_size=size, fees=0.0)

    def test_initial_state(self):
        assert self.portfolio.cash == 1000.0
        assert self.portfolio.total_value == 1000.0
        assert len(self.portfolio.positions) == 0

    def test_buy_creates_position(self):
        fill = self._make_fill("t1", Side.BUY, 0.50, 100.0)
        self.portfolio.process_fill(fill)

        assert "t1" in self.portfolio.positions
        assert self.portfolio.positions["t1"].size == 100.0
        assert self.portfolio.positions["t1"].avg_entry_price == 0.50
        assert self.portfolio.cash == 950.0  # 1000 - (0.50 * 100)

    def test_sell_closes_position(self):
        # Buy
        buy = self._make_fill("t1", Side.BUY, 0.50, 100.0)
        self.portfolio.process_fill(buy)

        # Sell at profit
        sell = self._make_fill("t1", Side.SELL, 0.70, 100.0)
        self.portfolio.process_fill(sell)

        assert "t1" not in self.portfolio.positions
        assert self.portfolio.realized_pnl == pytest.approx(20.0)  # (0.70 - 0.50) * 100
        assert self.portfolio.cash == pytest.approx(1020.0)

    def test_price_update(self):
        fill = self._make_fill("t1", Side.BUY, 0.50, 100.0)
        self.portfolio.process_fill(fill)

        self.portfolio.update_prices({"t1": 0.60})

        assert self.portfolio.positions["t1"].unrealized_pnl == pytest.approx(10.0)
        assert self.portfolio.positions["t1"].current_price == 0.60

    def test_drawdown_tracking(self):
        fill = self._make_fill("t1", Side.BUY, 0.50, 100.0)
        self.portfolio.process_fill(fill)

        # Price goes up (new peak)
        self.portfolio.update_prices({"t1": 0.80})
        assert self.portfolio.peak_value > 1000.0

        # Price drops (drawdown)
        self.portfolio.update_prices({"t1": 0.30})
        assert self.portfolio.drawdown_pct > 0

    def test_snapshot(self):
        fill = self._make_fill("t1", Side.BUY, 0.50, 100.0)
        self.portfolio.process_fill(fill)

        snap = self.portfolio.take_snapshot()
        assert snap.total_value == self.portfolio.total_value
        assert snap.num_positions == 1
        assert len(self.portfolio.snapshots) == 1

    def test_multiple_positions(self):
        fill1 = self._make_fill("t1", Side.BUY, 0.50, 50.0)
        fill2 = self._make_fill("t2", Side.BUY, 0.30, 100.0)

        self.portfolio.process_fill(fill1)
        self.portfolio.process_fill(fill2)

        assert len(self.portfolio.positions) == 2
        assert self.portfolio.cash == pytest.approx(1000.0 - 25.0 - 30.0)


# ── Risk Manager Tests ───────────────────────────────────────────

class TestRiskManager:
    def setup_method(self):
        self.trading = TradingConfig(
            max_portfolio_exposure_usd=500.0,
            max_single_position_usd=100.0,
            max_positions=5,
            kelly_fraction=0.25,
        )
        self.risk_config = RiskConfig(
            max_drawdown_pct=0.15,
            max_daily_loss_usd=50.0,
            stop_loss_pct=0.30,
            trailing_stop_pct=0.20,
        )
        self.portfolio = Portfolio(initial_cash=1000.0)
        self.sizer = PositionSizer(self.trading, self.risk_config)
        self.risk = RiskManager(self.risk_config, self.trading, self.portfolio, self.sizer)

    def _make_signal(self, edge: float = 0.10, confidence: float = 0.7) -> Signal:
        return Signal(
            market_condition_id="m1", token_id="t1", side=Side.BUY,
            outcome="Yes", estimated_fair_value=0.50 + edge,
            market_price=0.50, edge=edge, confidence=confidence,
            strategy="test",
        )

    def test_approve_good_signal(self):
        signal = self._make_signal(edge=0.15, confidence=0.8)
        approved, size, reason = self.risk.check_signal(signal)
        assert approved
        assert size > 0
        assert reason == "approved"

    def test_reject_after_max_drawdown(self):
        """Should halt trading after max drawdown."""
        # Simulate loss to trigger drawdown
        self.portfolio.cash = 800.0  # 20% drawdown > 15% limit
        self.portfolio.peak_value = 1000.0

        signal = self._make_signal()
        approved, _, reason = self.risk.check_signal(signal)
        assert not approved
        assert "drawdown" in reason.lower()
        assert self.risk.trading_halted

    def test_reject_after_daily_loss(self):
        """Should reject after daily loss limit hit."""
        self.risk.daily_pnl = -55.0  # Over $50 limit

        signal = self._make_signal()
        approved, _, reason = self.risk.check_signal(signal)
        assert not approved
        assert "daily loss" in reason.lower()

    def test_reject_max_positions(self):
        """Should reject when max positions reached."""
        # Fill up positions
        for i in range(5):
            self.portfolio.positions[f"t{i}"] = Position(
                market_condition_id=f"m{i}", token_id=f"t{i}",
                outcome="Yes", size=10.0, avg_entry_price=0.50,
            )

        signal = self._make_signal()
        approved, _, reason = self.risk.check_signal(signal)
        assert not approved
        assert "max positions" in reason.lower()

    def test_stop_loss_trigger(self):
        """Should detect positions that hit stop loss."""
        self.portfolio.positions["t1"] = Position(
            market_condition_id="m1", token_id="t1",
            outcome="Yes", size=100.0, avg_entry_price=0.50,
            current_price=0.30,  # 40% loss > 30% stop
        )

        stops = self.risk.check_stop_losses()
        assert len(stops) == 1
        assert stops[0]["type"] == "stop_loss"

    def test_trailing_stop_trigger(self):
        """Should detect positions that hit trailing stop."""
        pos = Position(
            market_condition_id="m1", token_id="t1",
            outcome="Yes", size=100.0, avg_entry_price=0.50,
            current_price=0.65,
            max_price_seen=0.85,  # Dropped 23.5% from peak > 20% trailing
        )
        self.portfolio.positions["t1"] = pos

        stops = self.risk.check_stop_losses()
        assert len(stops) == 1
        assert stops[0]["type"] == "trailing_stop"

    def test_risk_summary(self):
        summary = self.risk.get_risk_summary()
        assert "trading_halted" in summary
        assert "drawdown_pct" in summary
        assert "portfolio_value" in summary
        assert summary["portfolio_value"] == 1000.0
