#!/usr/bin/env python3
"""Monte Carlo backtest runner — runs N simulations with different seeds.

Shows detailed results: per-seed P&L, trade logs, per-strategy attribution,
portfolio equity curves, and aggregate statistics.

Usage:
    python run_monte_carlo.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import structlog

from polymarket_bot.backtesting.engine import BacktestEngine, BacktestResult
from polymarket_bot.backtesting.simulator import MarketSimulator
from polymarket_bot.clients.economic_data import MockEconomicDataClient
from polymarket_bot.clients.odds_sources import OddsAggregator
from polymarket_bot.clients.polymarket import PaperTradingClient
from polymarket_bot.clients.twitter import MockTwitterClient
from polymarket_bot.config import (
    BotConfig, PolymarketConfig, SentimentConfig,
)
from polymarket_bot.data.models import Market, OrderBook, Side, TradeResult
from polymarket_bot.execution.engine import ExecutionEngine
from polymarket_bot.execution.exit_manager import ExitManager
from polymarket_bot.narrative.strategy import NarrativeStrategy
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

# Suppress noisy logs during backtests
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(40))

# ── Configuration ────────────────────────────────────────────────

INITIAL_CAPITAL = 100.0
NUM_SEEDS = 20
NUM_MARKETS = 10
TIME_STEPS = 200  # ~8 simulated days at hourly resolution
STEPS_PER_DAY = 24


@dataclass
class TradeLog:
    """A single logged trade."""
    step: int
    day: int
    side: str
    token_id: str
    strategy: str
    price: float
    size: float
    cost: float
    portfolio_value: float


@dataclass
class DetailedResult:
    """Extended backtest result with trade logs and per-strategy attribution."""
    seed: int
    result: BacktestResult
    trades: list[TradeLog] = field(default_factory=list)
    strategy_pnl: dict = field(default_factory=dict)
    strategy_trades: dict = field(default_factory=dict)
    daily_values: list[float] = field(default_factory=list)


def run_single_backtest(seed: int) -> DetailedResult:
    """Run one full backtest with detailed trade logging."""
    config = BotConfig()
    simulator = MarketSimulator(seed=seed)
    sim_markets = simulator.create_simulated_markets(NUM_MARKETS)

    client = PaperTradingClient(PolymarketConfig())
    portfolio = Portfolio(initial_cash=INITIAL_CAPITAL)
    dynamic_sizer = DynamicKellySizer(config.trading, config.risk)
    risk_mgr = RiskManager(config.risk, config.trading, portfolio, dynamic_sizer)
    executor = ExecutionEngine(client, risk_mgr, portfolio)
    aggregator = SignalAggregator(min_composite_edge=config.trading.min_edge_threshold)
    regime_detector = RegimeDetector()
    exit_manager = ExitManager()

    # All 13 strategies
    mock_twitter = MockTwitterClient()
    odds_agg = OddsAggregator()
    mock_econ = MockEconomicDataClient(seed=seed)
    min_edge = config.trading.min_edge_threshold

    sim_sentiment_config = SentimentConfig(
        min_tweets=3,
        volume_spike_threshold=config.sentiment.volume_spike_threshold,
        sentiment_threshold=config.sentiment.sentiment_threshold,
        decay_half_life_minutes=config.sentiment.decay_half_life_minutes,
        keywords_per_market=config.sentiment.keywords_per_market,
    )

    strategies = [
        SentimentStrategy(mock_twitter, sim_sentiment_config),
        StatisticalStrategy(odds_agg, min_edge),
        MarketMakerStrategy(config.market_maker),
        ArbitrageStrategy(config.arbitrage, odds_agg),
        MomentumStrategy(min_edge=min_edge),
        ContrarianStrategy(min_edge=min_edge),
        TimeDecayStrategy(min_edge=min_edge),
        CorrelationStrategy(min_edge=min_edge),
        MicrostructureStrategy(min_edge=min_edge),
        VolatilityStrategy(min_edge=min_edge),
        EventCatalystStrategy(min_edge=min_edge),
        BTCDailyStrategy(min_edge=min_edge),
        NarrativeStrategy(economic_client=mock_econ, min_edge=min_edge, min_events=2),
    ]

    # Tracking
    portfolio_values = [INITIAL_CAPITAL]
    trade_logs: list[TradeLog] = []
    strategy_pnl: dict[str, float] = {}
    strategy_trades: dict[str, int] = {}
    daily_values: list[float] = [INITIAL_CAPITAL]
    all_trades: list[TradeResult] = []

    for step in range(TIME_STEPS):
        day = step // STEPS_PER_DAY

        if step % STEPS_PER_DAY == 0:
            risk_mgr.daily_pnl = 0.0

        # 1. Advance prices + economic data
        simulator.step_prices(sim_markets)
        if step % 5 == 0:
            mock_econ.step()

        # 2. Update prices
        prices = {}
        volumes = {}
        for sim in sim_markets:
            for token in sim.market.tokens:
                prices[token.token_id] = token.price
                volumes[token.token_id] = sim.market.volume_24h
        client.set_simulated_prices(prices)
        client.set_simulated_volumes(volumes)
        portfolio.update_prices(prices)

        # 3. Build context
        markets = [sim.market for sim in sim_markets]
        order_books: dict[str, OrderBook] = {}
        context: dict = {"positions": dict(portfolio.positions)}

        for sim in sim_markets:
            books = simulator.generate_order_book(sim)
            order_books.update(books)
            sentiment = simulator.generate_sentiment(sim)
            context[f"sentiment_{sim.market.condition_id}"] = sentiment
            context[f"price_history_{sim.market.condition_id}"] = sim.price_history.copy()
            if len(sim.price_history) >= 30:
                regime_state = regime_detector.detect(sim.price_history)
                context[f"regime_{sim.market.condition_id}"] = regime_state.regime.value

        # Add economic context for narrative strategy
        context["economic_indicators"] = mock_econ.get_all_indicators()

        # Add BTC price context for btc_daily strategy (simulated)
        rng = np.random.RandomState(seed + step)
        if step == 0:
            btc_open = 60000.0 + rng.normal(0, 2000)
            btc_price = btc_open
        else:
            btc_price = context.get("btc_price", 60000.0) * (1 + rng.normal(0.0002, 0.005))
            if step % STEPS_PER_DAY == 0:
                btc_open = btc_price  # New day open
            else:
                btc_open = context.get("btc_open_today", btc_price)
        context["btc_price"] = btc_price
        context["btc_open_today"] = btc_open

        # 4. Generate signals
        all_signals = []
        for strat in strategies:
            try:
                signals = strat.generate_signals(markets, order_books, context)
                all_signals.extend(signals)
            except Exception:
                pass

        # 5. Aggregate and execute
        ranked_signals = aggregator.aggregate(all_signals)
        top_signals = ranked_signals[:7]
        results = executor.execute_signals(top_signals)

        for r in results:
            if r.success:
                all_trades.append(r)
                strat_name = r.order.strategy or "unknown"
                strategy_trades[strat_name] = strategy_trades.get(strat_name, 0) + 1

                trade_logs.append(TradeLog(
                    step=step,
                    day=day,
                    side=r.order.side.value,
                    token_id=r.order.token_id[:20],
                    strategy=strat_name,
                    price=round(r.fill_price, 4),
                    size=round(r.fill_size, 2),
                    cost=round(r.net_cost, 2),
                    portfolio_value=round(portfolio.total_value, 2),
                ))

                edge_used = next(
                    (s.edge for s in top_signals if s.token_id == r.order.token_id), 0.05
                )
                current_price = prices.get(r.order.token_id, r.fill_price)
                pnl_approx = (
                    (current_price - r.fill_price) * r.fill_size
                    if r.order.side == Side.BUY
                    else (r.fill_price - current_price) * r.fill_size
                )
                dynamic_sizer.record_outcome(edge_used, pnl_approx)

        # 6. Exit management
        for r in results:
            if r.success and r.order.side == Side.BUY:
                sig_edge = next(
                    (s.edge for s in top_signals if s.token_id == r.order.token_id), 0.05
                )
                sim_market = next(
                    (sm for sm in sim_markets
                     if any(t.token_id == r.order.token_id for t in sm.market.tokens)),
                    None,
                )
                end_date = sim_market.market.end_date if sim_market else None
                exit_manager.register_entry(r.order.token_id, sig_edge, r.fill_size, end_date)

        exit_manager.advance_step()
        exit_rules = exit_manager.check_exits(portfolio.positions)
        for rule in exit_rules:
            try:
                sell_size = rule.position.size * rule.sell_fraction
                price = round(rule.position.current_price * (1 - 0.005 * rule.urgency), 4)
                exit_result = client.place_order(
                    token_id=rule.token_id, side=Side.SELL,
                    price=max(0.01, price), size=sell_size,
                    market_condition_id=rule.position.market_condition_id,
                    strategy=f"exit_{rule.exit_type}",
                )
                if exit_result and exit_result.success:
                    portfolio.process_fill(exit_result)
                    all_trades.append(exit_result)
                    if rule.sell_fraction >= 1.0:
                        exit_manager.remove_position(rule.token_id)

                    trade_logs.append(TradeLog(
                        step=step, day=day, side="SELL",
                        token_id=rule.token_id[:20],
                        strategy=f"exit_{rule.exit_type}",
                        price=round(exit_result.fill_price, 4),
                        size=round(exit_result.fill_size, 2),
                        cost=round(exit_result.net_cost, 2),
                        portfolio_value=round(portfolio.total_value, 2),
                    ))
            except Exception:
                pass

        # 7. Risk management
        if risk_mgr.needs_liquidation():
            liq_orders = risk_mgr.get_liquidation_orders()
            if liq_orders:
                liq_results = executor.execute_stop_losses(liq_orders)
                all_trades.extend([r for r in liq_results if r.success])
                risk_mgr.trading_halted = False
                risk_mgr.halt_reason = ""
                portfolio.peak_value = portfolio.total_value

        stops = risk_mgr.check_stop_losses()
        if stops:
            stop_results = executor.execute_stop_losses(stops)
            all_trades.extend([r for r in stop_results if r.success])

        # 8. Record
        portfolio_values.append(portfolio.total_value)
        if (step + 1) % STEPS_PER_DAY == 0:
            daily_values.append(round(portfolio.total_value, 2))

    # Compute per-strategy P&L attribution from trade logs
    # (approximate — tracks cost basis by strategy)
    for strat_name in strategy_trades:
        buy_cost = sum(
            t.cost for t in trade_logs
            if t.strategy == strat_name and t.side == "BUY"
        )
        strategy_pnl[strat_name] = -buy_cost  # Net cost (negative = invested)

    # Build result
    daily_returns = []
    for i in range(1, len(portfolio_values)):
        if portfolio_values[i - 1] > 0:
            daily_returns.append(
                (portfolio_values[i] - portfolio_values[i - 1]) / portfolio_values[i - 1]
            )

    step_pnls = []
    for i in range(1, len(portfolio_values)):
        change = portfolio_values[i] - portfolio_values[i - 1]
        if abs(change) > 0.001:
            step_pnls.append(change)

    gross_profit = sum(p for p in step_pnls if p > 0) if step_pnls else 0
    gross_loss = abs(sum(p for p in step_pnls if p < 0)) if step_pnls else 0
    peak = INITIAL_CAPITAL
    max_dd = 0.0
    for val in portfolio_values:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    result = BacktestResult(
        total_return_pct=portfolio.return_pct,
        total_pnl=portfolio.total_pnl,
        sharpe_ratio=calculate_sharpe_ratio(daily_returns) if daily_returns else 0.0,
        max_drawdown_pct=max_dd,
        win_rate=sum(1 for p in step_pnls if p > 0) / len(step_pnls) if step_pnls else 0,
        num_trades=len(all_trades),
        profit_factor=gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0,
        avg_trade_pnl=portfolio.total_pnl / len(all_trades) if all_trades else 0,
        best_trade_pnl=max(step_pnls) if step_pnls else 0,
        worst_trade_pnl=min(step_pnls) if step_pnls else 0,
        final_portfolio_value=portfolio.total_value,
        initial_portfolio_value=INITIAL_CAPITAL,
        time_steps=TIME_STEPS,
        portfolio_values=portfolio_values,
        daily_returns=daily_returns,
    )

    odds_agg.close()

    return DetailedResult(
        seed=seed,
        result=result,
        trades=trade_logs,
        strategy_pnl=strategy_pnl,
        strategy_trades=strategy_trades,
        daily_values=daily_values,
    )


def print_equity_curve(values: list[float], width: int = 50) -> str:
    """ASCII art equity curve."""
    if len(values) < 2:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1
    lines = []
    for v in values:
        bar_len = int((v - mn) / rng * width)
        bar = "#" * bar_len
        lines.append(f"  ${v:>8.2f} |{bar}")
    return "\n".join(lines)


def main():
    print(f"""
{'='*72}
  MONTE CARLO BACKTEST — POLYMARKET TRADING BOT
  {NUM_SEEDS} simulations | ${INITIAL_CAPITAL} initial | {TIME_STEPS} steps ({TIME_STEPS // STEPS_PER_DAY} sim days)
  13 strategies including narrative analysis
{'='*72}
""")

    all_results: list[DetailedResult] = []
    start_time = time.time()

    for i, seed in enumerate(range(1, NUM_SEEDS + 1)):
        t0 = time.time()
        dr = run_single_backtest(seed)
        elapsed = time.time() - t0
        r = dr.result

        status = "+" if r.total_pnl > 0 else "-"
        print(
            f"  Seed {seed:>2d}: ${r.final_portfolio_value:>8.2f} "
            f"({r.total_return_pct:>+7.1%}) "
            f"| {r.num_trades:>3d} trades "
            f"| Sharpe {r.sharpe_ratio:>5.2f} "
            f"| MaxDD {r.max_drawdown_pct:>5.1%} "
            f"| WR {r.win_rate:>4.0%} "
            f"| {elapsed:.1f}s"
        )
        all_results.append(dr)

    total_time = time.time() - start_time

    # ── Aggregate Statistics ────────────────────────────────────
    returns = [r.result.total_return_pct for r in all_results]
    final_values = [r.result.final_portfolio_value for r in all_results]
    sharpes = [r.result.sharpe_ratio for r in all_results]
    drawdowns = [r.result.max_drawdown_pct for r in all_results]
    win_rates = [r.result.win_rate for r in all_results]
    trade_counts = [r.result.num_trades for r in all_results]
    profitable = sum(1 for r in returns if r > 0)

    print(f"""
{'='*72}
  AGGREGATE RESULTS ({NUM_SEEDS} simulations)
{'='*72}

  Starting Capital:     ${INITIAL_CAPITAL:,.2f}

  RETURNS
  -------
  Mean Return:          {np.mean(returns):>+8.2%}
  Median Return:        {np.median(returns):>+8.2%}
  Std Dev:              {np.std(returns):>8.2%}
  Best Seed:            {np.max(returns):>+8.2%}  (seed {returns.index(max(returns))+1})
  Worst Seed:           {np.min(returns):>+8.2%}  (seed {returns.index(min(returns))+1})
  Profitable Seeds:     {profitable}/{NUM_SEEDS} ({profitable/NUM_SEEDS:.0%})

  FINAL PORTFOLIO VALUES
  ----------------------
  Mean:                 ${np.mean(final_values):>8.2f}
  Median:               ${np.median(final_values):>8.2f}
  Min:                  ${np.min(final_values):>8.2f}
  Max:                  ${np.max(final_values):>8.2f}
  P10:                  ${np.percentile(final_values, 10):>8.2f}
  P90:                  ${np.percentile(final_values, 90):>8.2f}

  RISK METRICS
  ------------
  Mean Sharpe:          {np.mean(sharpes):>8.2f}
  Mean Max Drawdown:    {np.mean(drawdowns):>8.2%}
  Worst Drawdown:       {np.max(drawdowns):>8.2%}
  Mean Win Rate:        {np.mean(win_rates):>8.1%}

  TRADING ACTIVITY
  ----------------
  Mean Trades/Sim:      {np.mean(trade_counts):>8.1f}
  Total Trades:         {sum(trade_counts):>8d}
  Avg P&L per Trade:    ${np.mean([r.result.avg_trade_pnl for r in all_results]):>8.4f}

  Runtime:              {total_time:.1f}s ({total_time/NUM_SEEDS:.1f}s per sim)
""")

    # ── Per-Strategy Attribution (across all seeds) ─────────────
    all_strategy_trades: dict[str, int] = {}
    for dr in all_results:
        for strat, count in dr.strategy_trades.items():
            all_strategy_trades[strat] = all_strategy_trades.get(strat, 0) + count

    if all_strategy_trades:
        print(f"  STRATEGY TRADE COUNTS (across all {NUM_SEEDS} seeds)")
        print(f"  {'Strategy':<25s} {'Trades':>8s} {'Avg/Sim':>8s}")
        print(f"  {'-'*25} {'-'*8} {'-'*8}")
        for strat, count in sorted(all_strategy_trades.items(), key=lambda x: -x[1]):
            print(f"  {strat:<25s} {count:>8d} {count/NUM_SEEDS:>8.1f}")
        print()

    # ── Show detailed trade log for best seed ───────────────────
    best_idx = returns.index(max(returns))
    best = all_results[best_idx]
    print(f"{'='*72}")
    print(f"  DETAILED TRADE LOG — Best Seed #{best.seed} ({best.result.total_return_pct:+.1%})")
    print(f"{'='*72}")
    print(f"  {'Day':>3s} {'Step':>4s} {'Side':>4s} {'Strategy':<20s} {'Price':>7s} {'Size':>6s} {'Cost':>8s} {'Portfolio':>10s}")
    print(f"  {'-'*3} {'-'*4} {'-'*4} {'-'*20} {'-'*7} {'-'*6} {'-'*8} {'-'*10}")

    for t in best.trades[:60]:  # Show first 60 trades
        print(
            f"  {t.day:>3d} {t.step:>4d} {t.side:>4s} "
            f"{t.strategy:<20s} "
            f"${t.price:>6.4f} {t.size:>6.1f} ${t.cost:>7.2f} ${t.portfolio_value:>9.2f}"
        )

    if len(best.trades) > 60:
        print(f"  ... ({len(best.trades) - 60} more trades)")
    print()

    # ── Daily portfolio values for best seed ────────────────────
    if best.daily_values:
        print(f"  DAILY PORTFOLIO VALUES — Seed #{best.seed}")
        print(f"  {'Day':>3s}  {'Value':>10s}  {'Change':>8s}")
        print(f"  {'-'*3}  {'-'*10}  {'-'*8}")
        for i, val in enumerate(best.daily_values):
            change = val - best.daily_values[i - 1] if i > 0 else 0
            print(f"  {i:>3d}  ${val:>9.2f}  ${change:>+7.2f}")
        print()

    # ── Equity curve for best seed (ASCII) ──────────────────────
    print(f"  EQUITY CURVE — Seed #{best.seed} (sampled)")
    values = best.result.portfolio_values
    # Sample ~20 points
    sample_interval = max(1, len(values) // 20)
    sampled = values[::sample_interval]
    print(print_equity_curve(sampled, width=45))
    print()

    # ── Show worst seed too ─────────────────────────────────────
    worst_idx = returns.index(min(returns))
    worst = all_results[worst_idx]
    print(f"  WORST SEED #{worst.seed} — Trade Summary")
    print(f"  Final: ${worst.result.final_portfolio_value:.2f} ({worst.result.total_return_pct:+.1%})")
    print(f"  Trades: {worst.result.num_trades} | MaxDD: {worst.result.max_drawdown_pct:.1%}")
    print()

    # ── Distribution of returns ─────────────────────────────────
    print(f"  RETURN DISTRIBUTION")
    buckets = [
        ("< -20%", sum(1 for r in returns if r < -0.20)),
        ("-20% to -10%", sum(1 for r in returns if -0.20 <= r < -0.10)),
        ("-10% to 0%", sum(1 for r in returns if -0.10 <= r < 0)),
        ("0% to +25%", sum(1 for r in returns if 0 <= r < 0.25)),
        ("+25% to +50%", sum(1 for r in returns if 0.25 <= r < 0.50)),
        ("+50% to +100%", sum(1 for r in returns if 0.50 <= r < 1.00)),
        ("+100% to +200%", sum(1 for r in returns if 1.00 <= r < 2.00)),
        ("> +200%", sum(1 for r in returns if r >= 2.00)),
    ]
    for label, count in buckets:
        bar = "#" * (count * 3)
        print(f"  {label:>16s}: {count:>2d} {bar}")

    print(f"\n{'='*72}")
    print(f"  Simulation complete. {profitable}/{NUM_SEEDS} seeds profitable.")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
