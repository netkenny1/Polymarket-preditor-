"""Backtesting engine for evaluating strategy performance.

Simulates the full trading pipeline with historical/simulated data
to measure strategy profitability before risking real capital.

Key metrics tracked:
- Total return and annualized return
- Sharpe ratio
- Maximum drawdown
- Win rate
- Average profit per trade
- Profit factor (gross profit / gross loss)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import structlog

from polymarket_bot.backtesting.simulator import MarketSimulator, SimulatedMarket
from polymarket_bot.clients.polymarket import PaperTradingClient
from polymarket_bot.config import (
    ArbitrageConfig,
    BotConfig,
    MarketMakerConfig,
    PolymarketConfig,
    SentimentConfig,
)
from polymarket_bot.clients.odds_sources import OddsAggregator
from polymarket_bot.clients.twitter import MockTwitterClient
from polymarket_bot.data.models import Market, OrderBook, Side, TradeResult
from polymarket_bot.execution.engine import ExecutionEngine
from polymarket_bot.risk.manager import RiskManager
from polymarket_bot.risk.portfolio import Portfolio
from polymarket_bot.risk.position_sizer import PositionSizer
from polymarket_bot.strategies.arbitrage import ArbitrageStrategy
from polymarket_bot.strategies.contrarian import ContrarianStrategy
from polymarket_bot.strategies.market_maker import MarketMakerStrategy
from polymarket_bot.strategies.market_regime import RegimeDetector
from polymarket_bot.strategies.momentum import MomentumStrategy
from polymarket_bot.strategies.sentiment import SentimentStrategy
from polymarket_bot.strategies.signals import SignalAggregator
from polymarket_bot.strategies.statistical import StatisticalStrategy
from polymarket_bot.strategies.time_decay import TimeDecayStrategy
from polymarket_bot.risk.dynamic_kelly import DynamicKellySizer
from polymarket_bot.utils.helpers import calculate_sharpe_ratio

logger = structlog.get_logger()


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    total_return_pct: float = 0.0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    num_trades: int = 0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0
    final_portfolio_value: float = 0.0
    initial_portfolio_value: float = 0.0
    time_steps: int = 0
    portfolio_values: list[float] = field(default_factory=list)
    daily_returns: list[float] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"  BACKTEST RESULTS\n"
            f"{'='*60}\n"
            f"  Initial Capital:    ${self.initial_portfolio_value:,.2f}\n"
            f"  Final Value:        ${self.final_portfolio_value:,.2f}\n"
            f"  Total P&L:          ${self.total_pnl:,.2f}\n"
            f"  Total Return:       {self.total_return_pct:+.2%}\n"
            f"  Sharpe Ratio:       {self.sharpe_ratio:.2f}\n"
            f"  Max Drawdown:       {self.max_drawdown_pct:.2%}\n"
            f"  Win Rate:           {self.win_rate:.1%}\n"
            f"  Total Trades:       {self.num_trades}\n"
            f"  Profit Factor:      {self.profit_factor:.2f}\n"
            f"  Avg Trade P&L:      ${self.avg_trade_pnl:.2f}\n"
            f"  Best Trade:         ${self.best_trade_pnl:.2f}\n"
            f"  Worst Trade:        ${self.worst_trade_pnl:.2f}\n"
            f"  Time Steps:         {self.time_steps}\n"
            f"{'='*60}\n"
        )


class BacktestEngine:
    """Run strategy backtests on simulated market data.

    Creates a full simulated environment with:
    - Simulated markets with known true probabilities
    - Price evolution via mean-reverting random walks
    - Paper trading execution
    - Full risk management
    """

    def __init__(
        self,
        config: BotConfig | None = None,
        initial_capital: float = 1000.0,
        seed: int = 42,
    ) -> None:
        self.config = config or BotConfig()
        self.initial_capital = initial_capital
        self.simulator = MarketSimulator(seed=seed)

    def run(
        self,
        num_markets: int = 10,
        time_steps: int = 200,
        strategies: list[str] | None = None,
    ) -> BacktestResult:
        """Run a full backtest simulation.

        Args:
            num_markets: Number of simulated markets.
            time_steps: Number of price updates to simulate.
            strategies: List of strategy names to use. Default: all.

        Returns:
            BacktestResult with performance metrics.
        """
        enabled = strategies or [
            "sentiment", "statistical", "market_maker", "arbitrage",
            "momentum", "contrarian", "time_decay",
        ]

        # ── Setup ────────────────────────────────────────────────
        sim_markets = self.simulator.create_simulated_markets(num_markets)

        client = PaperTradingClient(PolymarketConfig())
        portfolio = Portfolio(initial_cash=self.initial_capital)
        dynamic_sizer = DynamicKellySizer(self.config.trading, self.config.risk)
        risk_mgr = RiskManager(self.config.risk, self.config.trading, portfolio, dynamic_sizer)
        executor = ExecutionEngine(client, risk_mgr, portfolio)
        aggregator = SignalAggregator(min_composite_edge=self.config.trading.min_edge_threshold)
        regime_detector = RegimeDetector()

        # Initialize strategies
        strat_instances = []
        mock_twitter = MockTwitterClient()
        odds_agg = OddsAggregator()

        if "sentiment" in enabled:
            sim_sentiment_config = SentimentConfig(
                min_tweets=3,
                volume_spike_threshold=self.config.sentiment.volume_spike_threshold,
                sentiment_threshold=self.config.sentiment.sentiment_threshold,
                decay_half_life_minutes=self.config.sentiment.decay_half_life_minutes,
                keywords_per_market=self.config.sentiment.keywords_per_market,
            )
            strat_instances.append(SentimentStrategy(mock_twitter, sim_sentiment_config))
        if "statistical" in enabled:
            strat_instances.append(StatisticalStrategy(odds_agg, self.config.trading.min_edge_threshold))
        if "market_maker" in enabled:
            strat_instances.append(MarketMakerStrategy(self.config.market_maker))
        if "arbitrage" in enabled:
            strat_instances.append(ArbitrageStrategy(self.config.arbitrage, odds_agg))
        if "momentum" in enabled:
            strat_instances.append(MomentumStrategy(min_edge=self.config.trading.min_edge_threshold))
        if "contrarian" in enabled:
            strat_instances.append(ContrarianStrategy(min_edge=self.config.trading.min_edge_threshold))
        if "time_decay" in enabled:
            strat_instances.append(TimeDecayStrategy(min_edge=self.config.trading.min_edge_threshold))

        # ── Simulation Loop ──────────────────────────────────────
        portfolio_values = [self.initial_capital]
        all_trades: list[TradeResult] = []

        for step in range(time_steps):
            # Reset daily P&L each "day" (every 24 steps ≈ hourly data for 1 day)
            if step % 24 == 0:
                risk_mgr.daily_pnl = 0.0

            # 1. Advance prices
            self.simulator.step_prices(sim_markets)

            # 2. Update simulated prices and volumes in paper client
            prices = {}
            volumes = {}
            for sim in sim_markets:
                for token in sim.market.tokens:
                    prices[token.token_id] = token.price
                    volumes[token.token_id] = sim.market.volume_24h
            client.set_simulated_prices(prices)
            client.set_simulated_volumes(volumes)
            portfolio.update_prices(prices)

            # 3. Generate order books, context, and detect regimes
            markets = [sim.market for sim in sim_markets]
            order_books: dict[str, OrderBook] = {}
            context: dict[str, Any] = {"positions": {tid: pos for tid, pos in portfolio.positions.items()}}

            for sim in sim_markets:
                books = self.simulator.generate_order_book(sim)
                order_books.update(books)

                # Add sentiment context
                sentiment = self.simulator.generate_sentiment(sim)
                context[f"sentiment_{sim.market.condition_id}"] = sentiment

                # Add price history for momentum/contrarian/mean-reversion
                context[f"price_history_{sim.market.condition_id}"] = sim.price_history.copy()

                # Detect market regime
                if len(sim.price_history) >= 30:
                    regime_state = regime_detector.detect(sim.price_history)
                    context[f"regime_{sim.market.condition_id}"] = regime_state.regime.value

            # 4. Generate signals from all strategies
            all_signals = []
            for strat in strat_instances:
                try:
                    signals = strat.generate_signals(markets, order_books, context)
                    all_signals.extend(signals)
                except Exception as e:
                    logger.debug("strategy_error", strategy=strat.name, error=str(e))

            # 5. Aggregate and rank signals
            ranked_signals = aggregator.aggregate(all_signals)

            # 6. Execute top signals (limit to avoid over-trading)
            top_signals = ranked_signals[:7]
            results = executor.execute_signals(top_signals)
            for r in results:
                if r.success:
                    all_trades.append(r)
                    # Feed outcome to dynamic Kelly sizer
                    edge_used = next(
                        (s.edge for s in top_signals if s.token_id == r.order.token_id), 0.05
                    )
                    # Approximate immediate P&L for Kelly tracking
                    current_price = prices.get(r.order.token_id, r.fill_price)
                    pnl_approx = (current_price - r.fill_price) * r.fill_size if r.order.side == Side.BUY else (r.fill_price - current_price) * r.fill_size
                    dynamic_sizer.record_outcome(edge_used, pnl_approx)

            # 7. Check stop losses and liquidation
            if risk_mgr.needs_liquidation():
                liq_orders = risk_mgr.get_liquidation_orders()
                if liq_orders:
                    liq_results = executor.execute_stop_losses(liq_orders)
                    all_trades.extend([r for r in liq_results if r.success])
                    # Reset halt after liquidation so bot can resume with cash
                    risk_mgr.trading_halted = False
                    risk_mgr.halt_reason = ""
                    # Reset peak to current value to allow recovery trading
                    portfolio.peak_value = portfolio.total_value

            stops = risk_mgr.check_stop_losses()
            if stops:
                stop_results = executor.execute_stop_losses(stops)
                all_trades.extend([r for r in stop_results if r.success])

            # 8. Record portfolio value
            portfolio_values.append(portfolio.total_value)

            # Log progress every 50 steps
            if (step + 1) % 50 == 0:
                logger.info(
                    "backtest_progress",
                    step=step + 1,
                    portfolio_value=round(portfolio.total_value, 2),
                    num_trades=len(all_trades),
                    pnl=round(portfolio.total_pnl, 2),
                )

        # ── Compute Results ──────────────────────────────────────
        result = self._compute_results(
            portfolio, all_trades, portfolio_values, time_steps
        )

        odds_agg.close()
        return result

    def _compute_results(
        self,
        portfolio: Portfolio,
        trades: list[TradeResult],
        portfolio_values: list[float],
        time_steps: int,
    ) -> BacktestResult:
        """Compute performance metrics from backtest."""
        # Daily returns
        daily_returns = []
        for i in range(1, len(portfolio_values)):
            if portfolio_values[i - 1] > 0:
                ret = (portfolio_values[i] - portfolio_values[i - 1]) / portfolio_values[i - 1]
                daily_returns.append(ret)

        # Use portfolio-value-based P&L (more accurate than per-trade)
        # Group trades into round-trips where possible
        total_pnl = portfolio.total_value - self.initial_capital
        num_trades = len(trades)

        # Calculate win/loss from portfolio value changes between trades
        step_pnls = []
        for i in range(1, len(portfolio_values)):
            change = portfolio_values[i] - portfolio_values[i - 1]
            if abs(change) > 0.001:
                step_pnls.append(change)

        gross_profit = sum(p for p in step_pnls if p > 0) if step_pnls else 0
        gross_loss = abs(sum(p for p in step_pnls if p < 0)) if step_pnls else 0

        # Max drawdown from portfolio value series
        peak = self.initial_capital
        max_dd = 0.0
        for val in portfolio_values:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return BacktestResult(
            total_return_pct=portfolio.return_pct,
            total_pnl=portfolio.total_pnl,
            sharpe_ratio=calculate_sharpe_ratio(daily_returns) if daily_returns else 0.0,
            max_drawdown_pct=max_dd,
            win_rate=sum(1 for p in step_pnls if p > 0) / len(step_pnls) if step_pnls else 0,
            num_trades=num_trades,
            profit_factor=gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0,
            avg_trade_pnl=total_pnl / num_trades if num_trades > 0 else 0,
            best_trade_pnl=max(step_pnls) if step_pnls else 0,
            worst_trade_pnl=min(step_pnls) if step_pnls else 0,
            final_portfolio_value=portfolio.total_value,
            initial_portfolio_value=self.initial_capital,
            time_steps=time_steps,
            portfolio_values=portfolio_values,
            daily_returns=daily_returns,
        )
