#!/usr/bin/env python3
"""30-day stressed-market backtest — hourly steps, bearish macro, higher vol.

Simulates one calendar month (720 hourly steps) with $100 initial capital,
10 seeds, 15 markets, and macro/spread/volatility overrides vs. Monte Carlo.

Usage:
    python run_month_sim.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import structlog

from polymarket_bot.backtesting.engine import BacktestResult
from polymarket_bot.backtesting.simulator import MarketSimulator, SimulatedMarket
from polymarket_bot.clients.economic_data import MockEconomicDataClient
from polymarket_bot.clients.odds_sources import OddsAggregator
from polymarket_bot.clients.polymarket import PaperTradingClient
from polymarket_bot.clients.twitter import MockTwitterClient
from polymarket_bot.config import (
    BotConfig,
    PolymarketConfig,
    SentimentConfig,
)
from polymarket_bot.data.models import OrderBook, OrderBookLevel, Side, TradeResult
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

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(40))

# ── Month sim configuration ─────────────────────────────────────

INITIAL_CAPITAL = 100.0
NUM_SEEDS = 10
NUM_MARKETS = 15
TIME_STEPS = 720  # 30 days * 24 h
STEPS_PER_DAY = 24
NUM_DAYS = TIME_STEPS // STEPS_PER_DAY  # 30


class BearishMockEconomicDataClient(MockEconomicDataClient):
    """Mock macro with bearish equilibrium (tariffs, hot CPI, weak sentiment)."""

    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed=seed)
        # Align imports/exports so exports - imports stays near -95 after step()
        self._equilibrium.update({
            "trade_balance": -95.0,
            "cpi_yoy": 5.8,
            "unemployment_rate": 5.2,
            "fed_funds_rate": 4.75,
            "consumer_sentiment": 48.0,
            "tariff_rate_avg": 18.0,
            "imports": 350.0,
            "exports": 255.0,
        })
        self._current = dict(self._equilibrium)
        self._previous = dict(self._equilibrium)


class StressedMarketSimulator(MarketSimulator):
    """Higher vol, slower mean reversion, wider spreads."""

    MEAN_REVERSION = 0.003
    VOL_LO, VOL_HI = 0.03, 0.10
    SPREAD_LO, SPREAD_HI = 0.02, 0.08

    def create_simulated_markets(self, count: int = 10) -> list[SimulatedMarket]:
        markets = super().create_simulated_markets(count)
        for sim in markets:
            sim.volatility = float(self.rng.uniform(self.VOL_LO, self.VOL_HI))
        return markets

    def step_prices(self, sim_markets: list[SimulatedMarket]) -> None:
        for sim in sim_markets:
            current = sim.price_history[-1]
            reversion = self.MEAN_REVERSION * (sim.true_probability - current)
            noise = self.rng.normal(0, sim.volatility)
            new_price = current + reversion + noise
            new_price = max(0.02, min(0.98, new_price))
            sim.price_history.append(new_price)
            for token in sim.market.tokens:
                if token.outcome == "Yes":
                    token.price = new_price
                else:
                    token.price = 1.0 - new_price

    def generate_order_book(self, sim: SimulatedMarket, depth: int = 5) -> dict[str, OrderBook]:
        books = {}
        for token in sim.market.tokens:
            mid = token.price
            spread = self.rng.uniform(self.SPREAD_LO, self.SPREAD_HI)
            bids = []
            asks = []
            for i in range(depth):
                bid_price = max(0.01, mid - spread / 2 - i * 0.01)
                ask_price = min(0.99, mid + spread / 2 + i * 0.01)
                bid_size = self.rng.uniform(10, 200) * (depth - i) / depth
                ask_size = self.rng.uniform(10, 200) * (depth - i) / depth
                bids.append(OrderBookLevel(price=round(bid_price, 2), size=round(bid_size, 2)))
                asks.append(OrderBookLevel(price=round(ask_price, 2), size=round(ask_size, 2)))
            bids.sort(key=lambda x: x.price, reverse=True)
            asks.sort(key=lambda x: x.price)
            books[token.token_id] = OrderBook(
                token_id=token.token_id,
                bids=bids,
                asks=asks,
            )
        return books


@dataclass
class TradeLog:
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
    seed: int
    result: BacktestResult
    trades: list[TradeLog] = field(default_factory=list)
    strategy_pnl: dict = field(default_factory=dict)
    strategy_trades: dict = field(default_factory=dict)
    daily_values: list[float] = field(default_factory=list)
    weekly_pnl: list[float] = field(default_factory=list)


def weekly_pnl_from_daily(daily_values: list[float]) -> list[float]:
    """Four calendar weeks: P&L = value at end of week - value at week start."""
    weeks: list[float] = []
    for w in range(4):
        start_i = w * 7
        end_i = (w + 1) * 7
        if end_i < len(daily_values):
            weeks.append(daily_values[end_i] - daily_values[start_i])
        else:
            weeks.append(0.0)
    return weeks


def run_single_backtest(seed: int) -> DetailedResult:
    config = BotConfig()
    simulator = StressedMarketSimulator(seed=seed)
    sim_markets = simulator.create_simulated_markets(NUM_MARKETS)

    client = PaperTradingClient(PolymarketConfig())
    portfolio = Portfolio(initial_cash=INITIAL_CAPITAL)
    dynamic_sizer = DynamicKellySizer(config.trading, config.risk)
    risk_mgr = RiskManager(config.risk, config.trading, portfolio, dynamic_sizer)
    executor = ExecutionEngine(client, risk_mgr, portfolio)
    aggregator = SignalAggregator(min_composite_edge=0.015)
    regime_detector = RegimeDetector()
    exit_manager = ExitManager()

    mock_twitter = MockTwitterClient()
    odds_agg = OddsAggregator()
    mock_econ = BearishMockEconomicDataClient(seed=seed)
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

        simulator.step_prices(sim_markets)
        if step % 5 == 0:
            mock_econ.step()

        prices = {}
        volumes = {}
        for sim in sim_markets:
            for token in sim.market.tokens:
                prices[token.token_id] = token.price
                volumes[token.token_id] = sim.market.volume_24h
        client.set_simulated_prices(prices)
        client.set_simulated_volumes(volumes)
        portfolio.update_prices(prices)

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

        context["economic_indicators"] = mock_econ.get_all_indicators()

        rng = np.random.RandomState(seed + step)
        if step == 0:
            btc_open = 60000.0 + rng.normal(0, 3000)
            btc_price = btc_open
        else:
            btc_price = context.get("btc_price", 60000.0) * (1 + rng.normal(0.0001, 0.02))
            if step % STEPS_PER_DAY == 0:
                btc_open = btc_price
            else:
                btc_open = context.get("btc_open_today", btc_price)
        context["btc_price"] = btc_price
        context["btc_open_today"] = btc_open

        all_signals = []
        for strat in strategies:
            try:
                signals = strat.generate_signals(markets, order_books, context)
                all_signals.extend(signals)
            except Exception:
                pass

        ranked_signals = aggregator.aggregate(all_signals)
        top_signals = ranked_signals[:7]
        results = executor.execute_signals(top_signals)

        for r in results:
            if r.success:
                all_trades.append(r)
                strat_name = r.order.strategy or "unknown"
                strategy_trades[strat_name] = strategy_trades.get(strat_name, 0) + 1

                trade_logs.append(
                    TradeLog(
                        step=step,
                        day=day,
                        side=r.order.side.value,
                        token_id=r.order.token_id[:20],
                        strategy=strat_name,
                        price=round(r.fill_price, 4),
                        size=round(r.fill_size, 2),
                        cost=round(r.net_cost, 2),
                        portfolio_value=round(portfolio.total_value, 2),
                    )
                )

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

        for r in results:
            if r.success and r.order.side == Side.BUY:
                sig_edge = next(
                    (s.edge for s in top_signals if s.token_id == r.order.token_id), 0.05
                )
                sim_market = next(
                    (
                        sm
                        for sm in sim_markets
                        if any(t.token_id == r.order.token_id for t in sm.market.tokens)
                    ),
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
                    token_id=rule.token_id,
                    side=Side.SELL,
                    price=max(0.01, price),
                    size=sell_size,
                    market_condition_id=rule.position.market_condition_id,
                    strategy=f"exit_{rule.exit_type}",
                )
                if exit_result and exit_result.success:
                    portfolio.process_fill(exit_result)
                    all_trades.append(exit_result)
                    if rule.sell_fraction >= 1.0:
                        exit_manager.remove_position(rule.token_id)

                    trade_logs.append(
                        TradeLog(
                            step=step,
                            day=day,
                            side="SELL",
                            token_id=rule.token_id[:20],
                            strategy=f"exit_{rule.exit_type}",
                            price=round(exit_result.fill_price, 4),
                            size=round(exit_result.fill_size, 2),
                            cost=round(exit_result.net_cost, 2),
                            portfolio_value=round(portfolio.total_value, 2),
                        )
                    )
            except Exception:
                pass

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

        portfolio_values.append(portfolio.total_value)
        if (step + 1) % STEPS_PER_DAY == 0:
            daily_values.append(round(portfolio.total_value, 2))

    for strat_name in strategy_trades:
        buy_cost = sum(
            t.cost for t in trade_logs if t.strategy == strat_name and t.side == "BUY"
        )
        strategy_pnl[strat_name] = -buy_cost

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
        profit_factor=gross_profit / gross_loss
        if gross_loss > 0
        else float("inf")
        if gross_profit > 0
        else 0,
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
    weekly = weekly_pnl_from_daily(daily_values)

    return DetailedResult(
        seed=seed,
        result=result,
        trades=trade_logs,
        strategy_pnl=strategy_pnl,
        strategy_trades=strategy_trades,
        daily_values=daily_values,
        weekly_pnl=weekly,
    )


def print_equity_curve(values: list[float], width: int = 50) -> str:
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


def print_weekly_table(results: list[DetailedResult]) -> None:
    print(f"\n{'=' * 72}")
    print(f"  WEEKLY P&L (Weeks 1–4, seven days each; stressed macro)")
    print(f"{'=' * 72}\n")
    hdr = f"  {'Seed':>4s} │ {'W1':>10s} {'W2':>10s} {'W3':>10s} {'W4':>10s} │ {'4W Sum':>10s}"
    print(hdr)
    print(f"  {'─' * 4}─┼{'─' * 43}┼{'─' * 10}")

    def fmt(v: float) -> str:
        return f"${v:+.2f}"

    for dr in results:
        w = dr.weekly_pnl
        s4 = sum(w) if w else 0.0
        print(
            f"  {dr.seed:>4d} │ {fmt(w[0]):>10s} {fmt(w[1]):>10s} {fmt(w[2]):>10s} {fmt(w[3]):>10s} │ {fmt(s4):>10s}"
        )

    print(f"  {'─' * 4}─┼{'─' * 43}┼{'─' * 10}")
    w1 = [dr.weekly_pnl[0] for dr in results]
    w2 = [dr.weekly_pnl[1] for dr in results]
    w3 = [dr.weekly_pnl[2] for dr in results]
    w4 = [dr.weekly_pnl[3] for dr in results]
    print(
        f"\n  Aggregate week means: W1 ${np.mean(w1):+.2f} | W2 ${np.mean(w2):+.2f} | "
        f"W3 ${np.mean(w3):+.2f} | W4 ${np.mean(w4):+.2f}"
    )
    if NUM_DAYS >= 29:
        rem = []
        for dr in results:
            dv = dr.daily_values
            if len(dv) > 29:
                rem.append(dv[-1] - dv[28])
        if rem:
            print(
                f"  Days 29–30 remainder (mean across seeds): ${np.mean(rem):+.2f} "
                f"(final − end of week 4)\n"
            )
        else:
            print()
    else:
        print()


def print_projected_capital_scaling(results: list[DetailedResult]) -> None:
    """Scale each seed's outcome multiplier to other starting capitals."""
    capitals = [100.0, 500.0, 1000.0]
    mults = [dr.result.final_portfolio_value / INITIAL_CAPITAL for dr in results]
    mean_mult = float(np.mean(mults))
    med_mult = float(np.median(mults))
    p10_mult = float(np.percentile(mults, 10))
    p90_mult = float(np.percentile(mults, 90))

    print(f"{'=' * 72}")
    print("  MONTHLY PROJECTED FINAL VALUE (linear scale from sim at $100)")
    print("  Assumes same return multiple as $100 run (position limits unchanged).")
    print(f"{'=' * 72}\n")
    print(f"  {'Start':>8s} │ {'Mean final':>14s} {'Median':>14s} {'P10':>14s} {'P90':>14s}")
    print(f"  {'─' * 8}─┼{'─' * 14}┼{'─' * 14}┼{'─' * 14}┼{'─' * 14}")
    for c in capitals:
        print(
            f"  ${c:>7.0f} │ ${c * mean_mult:>13.2f} ${c * med_mult:>13.2f} "
            f"${c * p10_mult:>13.2f} ${c * p90_mult:>13.2f}"
        )
    print(
        f"\n  Implied mean monthly return: {(mean_mult - 1.0) * 100:+.2f}% "
        f"(median {(med_mult - 1.0) * 100:+.2f}%)\n"
    )


def main() -> None:
    sim_line = (
        f"  Simulator: vol {StressedMarketSimulator.VOL_LO:.2f}–{StressedMarketSimulator.VOL_HI:.2f}, "
        f"mean reversion {StressedMarketSimulator.MEAN_REVERSION}, "
        f"spreads {StressedMarketSimulator.SPREAD_LO:.2f}–{StressedMarketSimulator.SPREAD_HI:.2f}"
    )
    print(
        f"""
{'=' * 72}
  30-DAY STRESSED-MARKET SIMULATION — POLYMARKET BOT
  Seeds: {NUM_SEEDS} | Initial: ${INITIAL_CAPITAL:.0f} | Steps: {TIME_STEPS} ({NUM_DAYS} days, hourly)
  Markets: {NUM_MARKETS} | Macro: bearish (CPI 5.8%, tariffs ~18%, sentiment 48)
{sim_line}
{'=' * 72}
""",
        flush=True,
    )

    all_results: list[DetailedResult] = []
    start_time = time.time()

    for seed in range(1, NUM_SEEDS + 1):
        t0 = time.time()
        dr = run_single_backtest(seed)
        elapsed = time.time() - t0
        r = dr.result
        print(
            f"  Seed {seed:>2d}: ${r.final_portfolio_value:>8.2f} "
            f"({r.total_return_pct:>+7.1%}) "
            f"| {r.num_trades:>4d} trades "
            f"| Sharpe {r.sharpe_ratio:>5.2f} "
            f"| MaxDD {r.max_drawdown_pct:>5.1%} "
            f"| {elapsed:.1f}s",
            flush=True,
        )
        all_results.append(dr)

    total_time = time.time() - start_time
    returns = [r.result.total_return_pct for r in all_results]
    final_values = [r.result.final_portfolio_value for r in all_results]
    sharpes = [r.result.sharpe_ratio for r in all_results]
    drawdowns = [r.result.max_drawdown_pct for r in all_results]
    win_rates = [r.result.win_rate for r in all_results]
    trade_counts = [r.result.num_trades for r in all_results]
    profitable = sum(1 for r in returns if r > 0)

    print(
        f"""
{'=' * 72}
  AGGREGATE STATS ({NUM_SEEDS} seeds × {NUM_DAYS} days)
{'=' * 72}

  Starting capital:     ${INITIAL_CAPITAL:,.2f}

  Returns
  -------
  Mean:                 {np.mean(returns):>+8.2%}
  Median:               {np.median(returns):>+8.2%}
  Std:                  {np.std(returns):>8.2%}
  Best seed:            {np.max(returns):>+8.2%}  (seed {returns.index(max(returns)) + 1})
  Worst seed:           {np.min(returns):>+8.2%}  (seed {returns.index(min(returns)) + 1})
  Profitable seeds:     {profitable}/{NUM_SEEDS}

  Final portfolio ($)
  -------------------
  Mean:                 ${np.mean(final_values):>8.2f}
  Median:               ${np.median(final_values):>8.2f}
  Min / Max:            ${np.min(final_values):>8.2f} / ${np.max(final_values):>8.2f}

  Risk
  ----
  Mean Sharpe:          {np.mean(sharpes):>8.2f}
  Mean max drawdown:    {np.mean(drawdowns):>8.2%}
  Worst drawdown:       {np.max(drawdowns):>8.2%}
  Mean win rate (step): {np.mean(win_rates):>8.1%}

  Activity
  --------
  Mean trades/sim:      {np.mean(trade_counts):>8.1f}
  Total trades:         {sum(trade_counts):>8d}
  Runtime:              {total_time:.1f}s ({total_time / NUM_SEEDS:.1f}s per seed)
"""
    )

    all_strategy_trades: dict[str, int] = {}
    for dr in all_results:
        for strat, count in dr.strategy_trades.items():
            all_strategy_trades[strat] = all_strategy_trades.get(strat, 0) + count

    print(f"{'=' * 72}")
    print("  STRATEGY ATTRIBUTION (trade counts, all seeds)")
    print(f"{'=' * 72}\n")
    print(f"  {'Strategy':<28s} {'Trades':>8s} {'Avg/seed':>10s}")
    print(f"  {'-' * 28} {'-' * 8} {'-' * 10}")
    for strat, count in sorted(all_strategy_trades.items(), key=lambda x: -x[1]):
        print(f"  {strat:<28s} {count:>8d} {count / NUM_SEEDS:>10.1f}")
    print()

    print_weekly_table(all_results)

    best_idx = int(np.argmax(returns))
    worst_idx = int(np.argmin(returns))
    best = all_results[best_idx]
    worst = all_results[worst_idx]

    print(f"{'=' * 72}")
    print(f"  DAILY EQUITY — BEST SEED #{best.seed} ({best.result.total_return_pct:+.1%})")
    print(f"{'=' * 72}")
    print(print_equity_curve(best.daily_values, width=50))
    print()

    print(f"{'=' * 72}")
    print(f"  DAILY EQUITY — WORST SEED #{worst.seed} ({worst.result.total_return_pct:+.1%})")
    print(f"{'=' * 72}")
    print(print_equity_curve(worst.daily_values, width=50))
    print()

    print_projected_capital_scaling(all_results)

    print(f"{'=' * 72}")
    print(f"  Done. {profitable}/{NUM_SEEDS} seeds profitable after {NUM_DAYS} stressed sim days.")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
