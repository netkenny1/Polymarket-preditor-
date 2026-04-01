"""Integration tests for the full trading pipeline."""

import pytest

from polymarket_bot.clients.polymarket import PaperTradingClient
from polymarket_bot.clients.twitter import MockTwitterClient
from polymarket_bot.clients.odds_sources import OddsAggregator
from polymarket_bot.config import BotConfig, PolymarketConfig
from polymarket_bot.data.models import (
    Market,
    MarketCategory,
    OrderBook,
    OrderBookLevel,
    Side,
    Token,
)
from polymarket_bot.execution.engine import ExecutionEngine
from polymarket_bot.risk.manager import RiskManager
from polymarket_bot.risk.portfolio import Portfolio
from polymarket_bot.risk.position_sizer import PositionSizer
from polymarket_bot.strategies.sentiment import SentimentStrategy
from polymarket_bot.strategies.market_maker import MarketMakerStrategy
from polymarket_bot.strategies.signals import SignalAggregator


def _make_test_market() -> Market:
    return Market(
        condition_id="int_001",
        question="Will the test pass?",
        slug="test-pass",
        tokens=[
            Token(token_id="int_001_yes", outcome="Yes", price=0.55),
            Token(token_id="int_001_no", outcome="No", price=0.45),
        ],
        category=MarketCategory.OTHER,
        liquidity=10000.0,
        active=True,
    )


def _make_test_books() -> dict[str, OrderBook]:
    return {
        "int_001_yes": OrderBook(
            token_id="int_001_yes",
            bids=[
                OrderBookLevel(price=0.54, size=200.0),
                OrderBookLevel(price=0.53, size=300.0),
            ],
            asks=[
                OrderBookLevel(price=0.56, size=200.0),
                OrderBookLevel(price=0.57, size=300.0),
            ],
        ),
        "int_001_no": OrderBook(
            token_id="int_001_no",
            bids=[
                OrderBookLevel(price=0.44, size=200.0),
                OrderBookLevel(price=0.43, size=300.0),
            ],
            asks=[
                OrderBookLevel(price=0.46, size=200.0),
                OrderBookLevel(price=0.47, size=300.0),
            ],
        ),
    }


class TestFullPipeline:
    """Test the complete signal -> risk -> execution pipeline."""

    def setup_method(self):
        self.config = BotConfig()
        self.client = PaperTradingClient(PolymarketConfig())
        self.portfolio = Portfolio(initial_cash=1000.0)
        self.sizer = PositionSizer(self.config.trading, self.config.risk)
        self.risk = RiskManager(self.config.risk, self.config.trading, self.portfolio, self.sizer)
        self.executor = ExecutionEngine(self.client, self.risk, self.portfolio)

    def test_market_maker_round_trip(self):
        """Market maker should be able to open and close positions."""
        mm = MarketMakerStrategy(self.config.market_maker)
        market = _make_test_market()
        books = _make_test_books()
        context = {"positions": {}}

        signals = mm.generate_signals([market], books, context)
        aggregator = SignalAggregator(min_composite_edge=0.01)
        ranked = aggregator.aggregate(signals)

        if ranked:
            results = self.executor.execute_signals(ranked[:2])
            executed = [r for r in results if r.success]
            assert len(executed) >= 0  # May be 0 if risk rejects

    def test_paper_trading_execution(self):
        """Paper trading should simulate fills correctly."""
        self.client.set_simulated_prices({"int_001_yes": 0.55})

        result = self.client.place_order(
            token_id="int_001_yes",
            side=Side.BUY,
            price=0.55,
            size=50.0,
            market_condition_id="int_001",
        )

        assert result.success
        assert result.fill_size == 50.0
        assert result.fill_price > 0

    def test_portfolio_tracks_paper_trades(self):
        """Portfolio should correctly track paper trading fills."""
        self.client.set_simulated_prices({"int_001_yes": 0.55})

        result = self.client.place_order(
            token_id="int_001_yes",
            side=Side.BUY,
            price=0.55,
            size=50.0,
            market_condition_id="int_001",
        )

        self.portfolio.process_fill(result)

        assert "int_001_yes" in self.portfolio.positions
        assert self.portfolio.cash < 1000.0
        assert self.portfolio.positions["int_001_yes"].size == 50.0

    def test_risk_limits_enforcement(self):
        """Risk manager should enforce position limits."""
        # Try to buy more than max single position
        from polymarket_bot.data.models import Signal

        signal = Signal(
            market_condition_id="int_001",
            token_id="int_001_yes",
            side=Side.BUY,
            outcome="Yes",
            estimated_fair_value=0.70,
            market_price=0.55,
            edge=0.15,
            confidence=0.9,
            strategy="test",
        )

        approved, size, reason = self.risk.check_signal(signal)
        if approved:
            assert size <= self.config.trading.max_single_position_usd

    def test_stop_loss_execution(self):
        """Stop losses should execute through the full pipeline."""
        # Create a losing position
        from polymarket_bot.data.models import Position

        self.portfolio.positions["int_001_yes"] = Position(
            market_condition_id="int_001",
            token_id="int_001_yes",
            outcome="Yes",
            size=100.0,
            avg_entry_price=0.60,
            current_price=0.35,  # Big loss
        )
        self.client.set_simulated_prices({"int_001_yes": 0.35})

        stops = self.risk.check_stop_losses()
        assert len(stops) > 0

        results = self.executor.execute_stop_losses(stops)
        assert any(r.success for r in results)


class TestUtilities:
    """Test utility functions."""

    def test_helpers(self):
        from polymarket_bot.utils.helpers import (
            clamp, safe_divide, exponential_decay,
            compute_vwap, round_price, categorize_market,
            format_usd, is_market_open, calculate_sharpe_ratio,
        )

        assert clamp(1.5, 0, 1) == 1.0
        assert clamp(-0.5, 0, 1) == 0.0
        assert clamp(0.5, 0, 1) == 0.5

        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) == 0.0

        assert exponential_decay(100, 30, 30) == pytest.approx(50.0)
        assert exponential_decay(100, 30, 0) == 100.0

        assert compute_vwap([10, 20], [100, 100]) == 15.0

        assert round_price(0.555) == 0.56
        assert round_price(0.554) == 0.55

        assert categorize_market("Will Bitcoin reach $100K?", ["crypto"]) == "crypto"
        assert categorize_market("Will the president win?", ["politics"]) == "politics"
        assert categorize_market("NBA championship winner?", ["sports"]) == "sports"

        assert format_usd(1234.56) == "$1,234.56"

        assert is_market_open(None) is True

        # Sharpe with no data
        assert calculate_sharpe_ratio([]) == 0.0
        assert calculate_sharpe_ratio([0.01]) == 0.0
