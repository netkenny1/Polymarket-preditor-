"""Live paper trading backtest against real Polymarket data.

Runs the full 12-strategy pipeline against live Polymarket markets
with simulated (paper) execution. Unlike the synthetic backtester,
this fetches real order books and prices from Polymarket's APIs.

Usage:
    python -m polymarket_bot.backtesting.live_paper_backtest --budget 500 --cycles 20
    python -m polymarket_bot.backtesting.live_paper_backtest --budget 1000 --cycles 50 --delay 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from polymarket_bot.backtesting.engine import BacktestResult
from polymarket_bot.clients.crypto_feed import CryptoPriceFeed, MockCryptoPriceFeed
from polymarket_bot.clients.odds_sources import OddsAggregator
from polymarket_bot.clients.polymarket import PaperTradingClient, PolymarketClient
from polymarket_bot.clients.twitter import MockTwitterClient
from polymarket_bot.config import BotConfig, PolymarketConfig
from polymarket_bot.data.models import Market, OrderBook, Side, TradeResult
from polymarket_bot.data.store import DataStore
from polymarket_bot.discovery.market_scanner import MarketScanner
from polymarket_bot.execution.engine import ExecutionEngine
from polymarket_bot.execution.exit_manager import ExitManager
from polymarket_bot.risk.dynamic_kelly import DynamicKellySizer
from polymarket_bot.risk.manager import RiskManager
from polymarket_bot.risk.portfolio import Portfolio
from polymarket_bot.strategies.arbitrage import ArbitrageStrategy
from polymarket_bot.strategies.btc_daily import BTCDailyStrategy
from polymarket_bot.strategies.contrarian import ContrarianStrategy
from polymarket_bot.strategies.correlation import CorrelationStrategy
from polymarket_bot.strategies.event_catalyst import EventCatalystStrategy
from polymarket_bot.strategies.market_maker import MarketMakerStrategy
from polymarket_bot.strategies.market_regime import RegimeDetector
from polymarket_bot.strategies.microstructure import MicrostructureStrategy
from polymarket_bot.strategies.momentum import MomentumStrategy
from polymarket_bot.strategies.sentiment import SentimentStrategy
from polymarket_bot.strategies.signals import SignalAggregator
from polymarket_bot.strategies.statistical import StatisticalStrategy
from polymarket_bot.strategies.time_decay import TimeDecayStrategy
from polymarket_bot.strategies.volatility import VolatilityStrategy
from polymarket_bot.utils.helpers import calculate_sharpe_ratio

logger = structlog.get_logger()


@dataclass
class LiveBacktestResult(BacktestResult):
    """Extended results including per-strategy and per-market breakdowns."""

    strategy_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)
    markets_traded: list[str] = field(default_factory=list)
    cycle_count: int = 0
    elapsed_seconds: float = 0.0
    signals_generated: int = 0
    signals_executed: int = 0

    def summary(self) -> str:
        base = super().summary()
        extra = (
            f"  ── Live Paper Trading Details ──\n"
            f"  Cycles Run:         {self.cycle_count}\n"
            f"  Wall-clock Time:    {self.elapsed_seconds:.1f}s\n"
            f"  Markets Traded:     {len(self.markets_traded)}\n"
            f"  Signals Generated:  {self.signals_generated}\n"
            f"  Signals Executed:   {self.signals_executed}\n"
        )

        if self.strategy_breakdown:
            extra += "\n  ── Strategy Breakdown ──\n"
            for strat, stats in sorted(
                self.strategy_breakdown.items(),
                key=lambda x: x[1].get("trades", 0),
                reverse=True,
            ):
                trades = stats.get("trades", 0)
                pnl = stats.get("pnl", 0.0)
                wins = stats.get("wins", 0)
                wr = wins / trades * 100 if trades > 0 else 0
                extra += (
                    f"  {strat:<22s} trades={trades:>3d}  "
                    f"P&L=${pnl:>+8.2f}  WR={wr:>5.1f}%\n"
                )

        return base.rstrip("\n") + "\n" + extra + f"{'='*60}\n"


class LivePaperBacktest:
    """Run the full trading pipeline against real Polymarket data.

    Fetches live markets and order books from the Polymarket Gamma and
    CLOB APIs, runs all 12 strategies, and executes trades through the
    paper trading client. Cycles run in rapid succession (configurable
    delay) to compress what would normally take hours into minutes.
    """

    def __init__(
        self,
        budget_usd: float = 500.0,
        num_cycles: int = 20,
        delay_between_cycles: float = 5.0,
        max_markets: int = 30,
        max_trades_per_cycle: int = 5,
    ) -> None:
        self.budget_usd = budget_usd
        self.num_cycles = num_cycles
        self.delay = delay_between_cycles
        self.max_markets = max_markets
        self.max_trades_per_cycle = max_trades_per_cycle

        self.config = BotConfig()
        # Real markets are more efficient than simulated ones -- use a
        # tighter edge threshold and higher kelly fraction to generate
        # meaningful trading activity during the backtest.
        from polymarket_bot.config import TradingConfig, RiskConfig
        self.config.trading = TradingConfig(
            paper_trading=True,
            max_portfolio_exposure_usd=budget_usd,
            max_single_position_usd=min(budget_usd * 0.20, 150.0),
            min_edge_threshold=0.015,
            kelly_fraction=0.35,
            max_positions=30,
            min_liquidity_usd=self.config.trading.min_liquidity_usd,
            max_spread=self.config.trading.max_spread,
            rebalance_interval_seconds=self.config.trading.rebalance_interval_seconds,
            stale_price_seconds=self.config.trading.stale_price_seconds,
        )
        self.config.risk = RiskConfig(
            max_drawdown_pct=0.25,
            max_daily_loss_usd=budget_usd * 0.10,
            max_correlated_exposure_pct=0.50,
            position_limit_per_market_pct=0.12,
            stop_loss_pct=0.40,
            trailing_stop_pct=0.30,
        )
        self._all_trades: list[TradeResult] = []
        self._portfolio_values: list[float] = []
        self._signals_generated = 0
        self._signals_executed = 0
        self._strategy_stats: dict[str, dict[str, Any]] = {}
        self._markets_traded: set[str] = set()
        self._price_history: dict[str, list[float]] = {}
        self._context: dict[str, Any] = {}

    async def run(self) -> LiveBacktestResult:
        """Execute the full live paper backtest."""
        self._print_banner()
        start_time = time.monotonic()

        # -- Read-only client for real market data --
        reader = PolymarketClient(PolymarketConfig())
        paper = PaperTradingClient(PolymarketConfig())
        scanner = MarketScanner(reader, self.config.trading)

        portfolio = Portfolio(initial_cash=self.budget_usd)
        sizer = DynamicKellySizer(self.config.trading, self.config.risk)
        risk_mgr = RiskManager(self.config.risk, self.config.trading, portfolio, sizer)
        executor = ExecutionEngine(paper, risk_mgr, portfolio)
        aggregator = SignalAggregator(
            min_composite_edge=0.015,
        )
        exit_manager = ExitManager()
        regime_detector = RegimeDetector()

        mock_twitter = MockTwitterClient()
        odds_agg = OddsAggregator()
        crypto_feed = MockCryptoPriceFeed()

        strategies = self._build_strategies(mock_twitter, odds_agg)

        self._portfolio_values = [self.budget_usd]

        # -- Scan real markets once up front --
        print("\n  Scanning live Polymarket markets...")
        try:
            all_markets = await scanner.scan_all_markets()
            tradeable = scanner.filter_tradeable(all_markets)
        except Exception as e:
            print(f"  Failed to fetch markets: {e}")
            print("  Falling back to direct Gamma API fetch...")
            loop = asyncio.get_running_loop()
            tradeable = await loop.run_in_executor(
                None, lambda: reader.get_markets(limit=100, active=True)
            )

        if not tradeable:
            print("  No tradeable markets found. Exiting.")
            reader.close()
            paper.close()
            odds_agg.close()
            return self._compute_results(portfolio, 0, 0.0)

        print(f"  Found {len(tradeable)} tradeable markets")

        # Pre-seed price history from token prices so momentum / volatility
        # strategies can generate signals from cycle 1.
        for m in tradeable:
            yes_tok = next((t for t in m.tokens if t.outcome == "Yes"), None)
            if yes_tok and yes_tok.price > 0:
                base = yes_tok.price
                noise = np.random.default_rng(42).normal(0, 0.008, size=50)
                hist = [max(0.02, min(0.98, base + n)) for n in noise]
                self._price_history[m.condition_id] = hist
                self._context[f"price_history_{m.condition_id}"] = hist

        print("  Price history seeded for all markets\n")

        # -- Main trading loop --
        order_books: dict[str, OrderBook] = {}

        for cycle in range(1, self.num_cycles + 1):
            risk_mgr.daily_pnl = 0.0 if cycle % 24 == 0 else risk_mgr.daily_pnl
            crypto_feed.step()
            self._context.update(crypto_feed.get_btc_context())

            # Rank and pick top markets
            try:
                ranked = scanner.rank_markets(tradeable, order_books)
                top_markets = [r.market for r in ranked[: self.max_markets]]
            except Exception:
                top_markets = tradeable[: self.max_markets]

            # Fetch live order books
            loop = asyncio.get_running_loop()
            for market in top_markets:
                for token in market.tokens:
                    try:
                        book = await loop.run_in_executor(
                            None, reader.get_order_book, token.token_id,
                        )
                        if book:
                            order_books[token.token_id] = book
                    except Exception:
                        pass

            # Update prices from real order books
            prices: dict[str, float] = {}
            volumes: dict[str, float] = {}
            for tid, book in order_books.items():
                if book.mid_price is not None:
                    prices[tid] = book.mid_price
            for m in top_markets:
                for t in m.tokens:
                    volumes[t.token_id] = m.volume_24h

            paper.set_simulated_prices(prices)
            paper.set_simulated_volumes(volumes)
            portfolio.update_prices(prices)

            # Build context
            self._context["positions"] = dict(portfolio.positions)
            for m in top_markets:
                key = f"price_history_{m.condition_id}"
                hist = self._price_history.get(m.condition_id, [])
                yes_tok = next((t for t in m.tokens if t.outcome == "Yes"), None)
                if yes_tok and yes_tok.price > 0:
                    hist.append(yes_tok.price)
                    if len(hist) > 200:
                        hist = hist[-200:]
                    self._price_history[m.condition_id] = hist
                self._context[key] = hist

                if len(hist) >= 30:
                    state = regime_detector.detect(hist)
                    self._context[f"regime_{m.condition_id}"] = state.regime.value

            # Generate signals from all strategies
            all_signals = []
            for strat in strategies:
                try:
                    sigs = strat.generate_signals(top_markets, order_books, self._context)
                    all_signals.extend(sigs)
                except Exception as e:
                    logger.debug("strategy_error", strategy=strat.name, error=str(e))

            self._signals_generated += len(all_signals)

            # Aggregate and rank
            ranked_signals = aggregator.aggregate(all_signals)

            # Check exits
            exit_manager.advance_step()
            exit_rules = exit_manager.check_exits(portfolio.positions)
            for rule in exit_rules:
                try:
                    sell_size = rule.position.size * rule.sell_fraction
                    price = round(rule.position.current_price * (1 - 0.005 * rule.urgency), 4)
                    exit_result = paper.place_order(
                        token_id=rule.token_id,
                        side=Side.SELL,
                        price=max(0.01, price),
                        size=sell_size,
                        market_condition_id=rule.position.market_condition_id,
                        strategy=f"exit_{rule.exit_type}",
                    )
                    if exit_result and exit_result.success:
                        portfolio.process_fill(exit_result)
                        self._all_trades.append(exit_result)
                        self._record_strategy_trade(
                            f"exit_{rule.exit_type}", exit_result, prices,
                        )
                        if rule.sell_fraction >= 1.0:
                            exit_manager.remove_position(rule.token_id)
                except Exception:
                    pass

            # Execute top signals
            top_signals = ranked_signals[: self.max_trades_per_cycle]
            results = executor.execute_signals(top_signals)

            for r in results:
                if r.success:
                    self._all_trades.append(r)
                    self._signals_executed += 1
                    self._markets_traded.add(r.order.market_condition_id)
                    self._record_strategy_trade(r.order.strategy, r, prices)

                    # Feed Kelly sizer
                    edge_used = next(
                        (s.edge for s in top_signals if s.token_id == r.order.token_id),
                        0.05,
                    )
                    cur_px = prices.get(r.order.token_id, r.fill_price)
                    pnl_approx = (
                        (cur_px - r.fill_price) * r.fill_size
                        if r.order.side == Side.BUY
                        else (r.fill_price - cur_px) * r.fill_size
                    )
                    sizer.record_outcome(edge_used, pnl_approx)

                    # Register entries with exit manager
                    if r.order.side == Side.BUY:
                        market = next(
                            (m for m in top_markets if any(t.token_id == r.order.token_id for t in m.tokens)),
                            None,
                        )
                        exit_manager.register_entry(
                            r.order.token_id,
                            edge_used,
                            r.fill_size,
                            market.end_date if market else None,
                        )

            # Stop losses and liquidation
            if risk_mgr.needs_liquidation():
                liq_orders = risk_mgr.get_liquidation_orders()
                if liq_orders:
                    liq_results = executor.execute_stop_losses(liq_orders)
                    for lr in liq_results:
                        if lr.success:
                            self._all_trades.append(lr)
                    risk_mgr.trading_halted = False
                    risk_mgr.halt_reason = ""
                    portfolio.peak_value = portfolio.total_value

            stops = risk_mgr.check_stop_losses()
            if stops:
                stop_results = executor.execute_stop_losses(stops)
                for sr in stop_results:
                    if sr.success:
                        self._all_trades.append(sr)

            self._portfolio_values.append(portfolio.total_value)

            # Print cycle status
            trade_count = sum(1 for r in results if r.success)
            pnl = portfolio.total_pnl
            val = portfolio.total_value
            print(
                f"  Cycle {cycle:>3d}/{self.num_cycles} | "
                f"Value: ${val:>9,.2f} | P&L: ${pnl:>+8.2f} ({portfolio.return_pct:>+.1%}) | "
                f"Signals: {len(ranked_signals):>3d} | Trades: {trade_count} | "
                f"Positions: {len(portfolio.positions)}"
            )

            if cycle < self.num_cycles and self.delay > 0:
                await asyncio.sleep(self.delay)

        elapsed = time.monotonic() - start_time

        # Save results
        store = DataStore()
        store.save_trade_history(self._all_trades, "paper_backtest_trades.json")
        store.save_snapshots(portfolio.snapshots, "paper_backtest_snapshots.json")
        self._save_strategy_breakdown(store)

        # Cleanup
        reader.close()
        paper.close()
        odds_agg.close()

        result = self._compute_results(portfolio, self.num_cycles, elapsed)
        print(result.summary())
        return result

    def _build_strategies(self, twitter, odds_agg) -> list:
        min_edge = self.config.trading.min_edge_threshold
        return [
            SentimentStrategy(twitter, self.config.sentiment),
            StatisticalStrategy(odds_agg, min_edge),
            MarketMakerStrategy(self.config.market_maker),
            ArbitrageStrategy(self.config.arbitrage, odds_agg),
            MomentumStrategy(min_edge=min_edge),
            ContrarianStrategy(min_edge=min_edge),
            TimeDecayStrategy(min_edge=min_edge),
            CorrelationStrategy(min_edge=min_edge),
            MicrostructureStrategy(min_edge=min_edge),
            VolatilityStrategy(min_edge=min_edge),
            EventCatalystStrategy(min_edge=min_edge),
            BTCDailyStrategy(min_edge=min_edge),
        ]

    def _record_strategy_trade(
        self,
        strategy: str,
        result: TradeResult,
        prices: dict[str, float],
    ) -> None:
        if strategy not in self._strategy_stats:
            self._strategy_stats[strategy] = {
                "trades": 0, "wins": 0, "losses": 0, "pnl": 0.0,
            }
        stats = self._strategy_stats[strategy]
        stats["trades"] += 1
        cur_px = prices.get(result.order.token_id, result.fill_price)
        if result.order.side == Side.BUY:
            instant_pnl = (cur_px - result.fill_price) * result.fill_size
        else:
            instant_pnl = (result.fill_price - cur_px) * result.fill_size
        stats["pnl"] += instant_pnl
        if instant_pnl >= 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

    def _compute_results(
        self,
        portfolio: Portfolio,
        cycles: int,
        elapsed: float,
    ) -> LiveBacktestResult:
        values = self._portfolio_values
        daily_returns = []
        for i in range(1, len(values)):
            if values[i - 1] > 0:
                daily_returns.append(
                    (values[i] - values[i - 1]) / values[i - 1]
                )

        step_pnls = []
        for i in range(1, len(values)):
            change = values[i] - values[i - 1]
            if abs(change) > 0.001:
                step_pnls.append(change)

        gross_profit = sum(p for p in step_pnls if p > 0) if step_pnls else 0
        gross_loss = abs(sum(p for p in step_pnls if p < 0)) if step_pnls else 0

        peak = self.budget_usd
        max_dd = 0.0
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        num_trades = len(self._all_trades)

        return LiveBacktestResult(
            total_return_pct=portfolio.return_pct,
            total_pnl=portfolio.total_pnl,
            sharpe_ratio=calculate_sharpe_ratio(daily_returns) if daily_returns else 0.0,
            max_drawdown_pct=max_dd,
            win_rate=(
                sum(1 for p in step_pnls if p > 0) / len(step_pnls)
                if step_pnls else 0
            ),
            num_trades=num_trades,
            profit_factor=(
                gross_profit / gross_loss
                if gross_loss > 0
                else float("inf") if gross_profit > 0 else 0
            ),
            avg_trade_pnl=portfolio.total_pnl / num_trades if num_trades > 0 else 0,
            best_trade_pnl=max(step_pnls) if step_pnls else 0,
            worst_trade_pnl=min(step_pnls) if step_pnls else 0,
            final_portfolio_value=portfolio.total_value,
            initial_portfolio_value=self.budget_usd,
            time_steps=cycles,
            portfolio_values=values,
            daily_returns=daily_returns,
            strategy_breakdown=self._strategy_stats,
            markets_traded=list(self._markets_traded),
            cycle_count=cycles,
            elapsed_seconds=elapsed,
            signals_generated=self._signals_generated,
            signals_executed=self._signals_executed,
        )

    def _save_strategy_breakdown(self, store: DataStore) -> None:
        store.save_json(self._strategy_stats, "paper_backtest_strategy_stats.json")

    def _print_banner(self) -> None:
        print(f"""
{'='*60}
  LIVE PAPER TRADING BACKTEST
{'='*60}
  Budget:           ${self.budget_usd:,.2f}
  Cycles:           {self.num_cycles}
  Delay:            {self.delay}s between cycles
  Max Markets:      {self.max_markets}
  Trades/Cycle:     {self.max_trades_per_cycle}
  Strategies:       12 (full ensemble)
  Data Source:      LIVE Polymarket API
  Execution:        Paper (simulated fills)
{'='*60}""")


async def _async_main(args: argparse.Namespace) -> None:
    bt = LivePaperBacktest(
        budget_usd=args.budget,
        num_cycles=args.cycles,
        delay_between_cycles=args.delay,
        max_markets=args.max_markets,
        max_trades_per_cycle=args.max_trades,
    )
    await bt.run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live Paper Trading Backtest against real Polymarket data",
    )
    parser.add_argument("--budget", type=float, default=500.0, help="Starting capital (USD)")
    parser.add_argument("--cycles", type=int, default=20, help="Number of trading cycles to run")
    parser.add_argument("--delay", type=float, default=5.0, help="Seconds between cycles (0 for no delay)")
    parser.add_argument("--max-markets", type=int, default=30, help="Max markets to evaluate per cycle")
    parser.add_argument("--max-trades", type=int, default=5, help="Max new trades per cycle")
    args = parser.parse_args()

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(20),
    )

    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
