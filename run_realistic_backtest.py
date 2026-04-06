#!/usr/bin/env python3
"""Realistic backtest — zero hindsight bias, realistic fills.

Uses RealisticMarketSimulator (no mean-reversion toward true_prob,
sentiment from price not truth, persistent order books).

All context data derived from CURRENT OBSERVABLE PRICE only.
true_probability is used ONLY for market resolution at simulation end.

Usage:
    python run_realistic_backtest.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

import numpy as np
import structlog

from polymarket_bot.backtesting.simulator import RealisticMarketSimulator
from polymarket_bot.clients.economic_data import MockEconomicDataClient
from polymarket_bot.clients.odds_sources import EloRating, ExternalOdds, OddsAggregator, PollData
from polymarket_bot.clients.polymarket import PaperTradingClient
from polymarket_bot.clients.twitter import MockTwitterClient
from polymarket_bot.config import BotConfig, PolymarketConfig, SentimentConfig
from polymarket_bot.data.models import OrderBook, Side, TradeResult
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

# ── Configuration ────────────────────────────────────────────────
INITIAL_CAPITAL = 100.0
NUM_SEEDS = 20
NUM_MARKETS = 12
TIME_STEPS = 200
STEPS_PER_DAY = 24


def run_single(seed: int) -> dict:
    """Run one realistic backtest with zero hindsight bias."""
    config = BotConfig()
    simulator = RealisticMarketSimulator(seed=seed)
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

    portfolio_values = [INITIAL_CAPITAL]
    strategy_trades: dict[str, int] = {}
    total_trades = 0
    total_fills = 0

    # Persistent context state — real polls/odds don't wildly change every hour
    persistent_polls: dict[str, list] = {}
    persistent_elo: dict[str, dict] = {}
    persistent_ext_odds: dict[str, float] = {}
    btc_price = 60000.0
    btc_open = btc_price

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

        # ── PERSISTENT CONTEXT — polls/odds update slowly, not every step ──
        # Real polls release every few days. External odds update hourly.
        # We update polls once per day (every STEPS_PER_DAY) and
        # external odds every 4 steps, with small random walk updates.
        for sim in sim_markets:
            cid = sim.market.condition_id
            candidates = [t.outcome for t in sim.market.tokens]
            current_yes = sim.price_history[-1] if sim.price_history else 0.5

            if sim.market.category.value == "politics" and len(candidates) >= 2:
                if cid not in persistent_polls or step % STEPS_PER_DAY == 0:
                    polls = []
                    for c in candidates:
                        base_pct = current_yes if c == "Yes" else (1.0 - current_yes)
                        pct = max(0.05, min(0.95, base_pct + rng.normal(0, 0.06)))
                        polls.append(PollData(pollster=f"Mock-{rng.randint(1,100)}", candidate=c, pct=pct, sample_size=800))
                    persistent_polls[cid] = polls
                context[f"polls_{cid}"] = persistent_polls[cid]

            elif sim.market.category.value == "sports" and len(candidates) >= 2:
                if cid not in persistent_elo or step % STEPS_PER_DAY == 0:
                    elo_diff = 400 * (current_yes - 0.5)
                    persistent_elo[cid] = {
                        candidates[0]: EloRating(team=candidates[0], rating=1500 + elo_diff / 2 + rng.normal(0, 50), sport="generic"),
                        candidates[1]: EloRating(team=candidates[1], rating=1500 - elo_diff / 2 + rng.normal(0, 50), sport="generic"),
                    }
                context[f"elo_{cid}"] = persistent_elo[cid]

            # External odds update every 4 steps with small drift
            if cid not in persistent_ext_odds or step % 4 == 0:
                if cid in persistent_ext_odds:
                    prev = persistent_ext_odds[cid]
                    drift = rng.normal(0, 0.01)
                    persistent_ext_odds[cid] = max(0.05, min(0.95, prev + drift))
                else:
                    persistent_ext_odds[cid] = max(0.05, min(0.95, current_yes + rng.normal(0, 0.05)))
            ext_prob = persistent_ext_odds[cid]
            context[f"external_odds_{cid}"] = [
                ExternalOdds(source="MockBook", event_name=sim.market.question, outcome="Yes", implied_probability=ext_prob),
                ExternalOdds(source="MockBook", event_name=sim.market.question, outcome="No", implied_probability=1.0 - ext_prob),
            ]

        if step == 0:
            btc_open = 60000.0 + rng.normal(0, 3000)
            btc_price = btc_open
        else:
            btc_price = btc_price * (1 + rng.normal(0, 0.005))
            if step % STEPS_PER_DAY == 0:
                btc_open = btc_price
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
            total_trades += 1
            if r.success:
                total_fills += 1
                strat_name = r.order.strategy or "unknown"
                strategy_trades[strat_name] = strategy_trades.get(strat_name, 0) + 1

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
                    if rule.sell_fraction >= 1.0:
                        exit_manager.remove_position(rule.token_id)
            except Exception:
                pass

        stops = risk_mgr.check_stop_losses()
        if stops:
            executor.execute_stop_losses(stops)

        if risk_mgr.trading_halted:
            liq = risk_mgr.get_liquidation_orders()
            if liq:
                executor.execute_stop_losses(liq)
            risk_mgr.trading_halted = False
            portfolio.peak_value = portfolio.total_value

        portfolio_values.append(portfolio.total_value)
        if step % STEPS_PER_DAY == STEPS_PER_DAY - 1:
            pass  # day end

    # Force-close all open positions to get true liquidated returns
    for tid, pos in list(portfolio.positions.items()):
        if pos.size > 0:
            close_r = client.place_order(tid, Side.SELL, pos.current_price, pos.size,
                                         pos.market_condition_id, "close_end")
            if close_r.success:
                portfolio.process_fill(close_r)

    final = portfolio.total_value
    ret = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL
    equity = np.array(portfolio_values)
    daily_returns = np.diff(equity) / equity[:-1]
    sharpe = calculate_sharpe_ratio(list(daily_returns))
    peak = np.maximum.accumulate(equity)
    drawdowns = (peak - equity) / np.where(peak > 0, peak, 1)
    max_dd = float(np.max(drawdowns))
    wins = sum(1 for r in daily_returns if r > 0)
    total = len(daily_returns)
    wr = wins / total if total > 0 else 0

    return {
        "seed": seed,
        "final": final,
        "return_pct": ret,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "win_rate": wr,
        "trades": total_fills,
        "attempted": total_trades,
        "fill_rate": client.fill_rate,
        "strategy_trades": strategy_trades,
    }


def main():
    print()
    print("=" * 72)
    print("  REALISTIC BACKTEST — ZERO HINDSIGHT BIAS")
    print("  RealisticMarketSimulator | No true_prob leakage")
    print("  Realistic fill simulation | 2% fees")
    print(f"  {NUM_SEEDS} seeds | ${INITIAL_CAPITAL} initial | {TIME_STEPS} steps | {NUM_MARKETS} markets")
    print("=" * 72)
    print()

    results = []
    t0 = time.time()
    for seed in range(1, NUM_SEEDS + 1):
        st = time.time()
        r = run_single(seed)
        elapsed = time.time() - st
        results.append(r)
        print(
            f"  Seed {seed:2d}: ${r['final']:8.2f} ({r['return_pct']:+7.1%}) | "
            f"Sharpe {r['sharpe']:5.2f} | MaxDD {r['max_dd']:.1%} | "
            f"WR {r['win_rate']:.0%} | "
            f"Fills {r['trades']:3d}/{r['attempted']:3d} ({r['fill_rate']:.0%}) | "
            f"{elapsed:.1f}s"
        )

    total_time = time.time() - t0

    # ── Aggregate ─────────────────────────────────────────────────
    returns = [r["return_pct"] for r in results]
    finals = [r["final"] for r in results]
    sharpes = [r["sharpe"] for r in results]
    drawdowns = [r["max_dd"] for r in results]
    fill_rates = [r["fill_rate"] for r in results]
    profitable = sum(1 for r in returns if r > 0)
    losing = sum(1 for r in returns if r <= 0)

    print()
    print("=" * 72)
    print("  AGGREGATE RESULTS")
    print("=" * 72)
    print(f"  Mean Return:       {np.mean(returns):+.1%}")
    print(f"  Median Return:     {np.median(returns):+.1%}")
    print(f"  Best Seed:         {max(returns):+.1%}")
    print(f"  Worst Seed:        {min(returns):+.1%}")
    print(f"  Std Dev:           {np.std(returns):.1%}")
    print(f"  Profitable Seeds:  {profitable}/{NUM_SEEDS} ({profitable/NUM_SEEDS*100:.0f}%)")
    print(f"  Losing Seeds:      {losing}/{NUM_SEEDS}")
    print(f"  Mean Sharpe:       {np.mean(sharpes):.2f}")
    print(f"  Mean Max Drawdown: {np.mean(drawdowns):.1%}")
    print(f"  Worst Drawdown:    {max(drawdowns):.1%}")
    print(f"  Mean Fill Rate:    {np.mean(fill_rates):.0%}")
    print(f"  Mean Final:        ${np.mean(finals):.2f}")
    print(f"  Median Final:      ${np.median(finals):.2f}")
    print(f"  Total Time:        {total_time:.0f}s")

    # ── Strategy attribution ──────────────────────────────────────
    all_strat: dict[str, int] = {}
    for r in results:
        for s, c in r["strategy_trades"].items():
            all_strat[s] = all_strat.get(s, 0) + c

    print()
    print("  STRATEGY TRADE COUNTS (across all seeds)")
    print("  " + "-" * 50)
    sorted_strats = sorted(all_strat.items(), key=lambda x: x[1], reverse=True)
    for name, count in sorted_strats[:25]:
        avg = count / NUM_SEEDS
        print(f"  {name:55s} {count:5d}  ({avg:5.1f}/sim)")

    # ── Return distribution ───────────────────────────────────────
    print()
    print("  RETURN DISTRIBUTION")
    bins = [(-999, -0.20), (-0.20, -0.10), (-0.10, 0), (0, 0.10), (0.10, 0.25), (0.25, 0.50), (0.50, 999)]
    labels = ["< -20%", "-20% to -10%", "-10% to 0%", "0% to +10%", "+10% to +25%", "+25% to +50%", "> +50%"]
    for (lo, hi), label in zip(bins, labels):
        cnt = sum(1 for r in returns if lo <= r < hi)
        bar = "#" * (cnt * 3)
        print(f"  {label:>15s}: {cnt:2d} {bar}")

    # ── Honest assessment ─────────────────────────────────────────
    print()
    print("=" * 72)
    print("  HONEST ASSESSMENT")
    print("=" * 72)
    mean_ret = np.mean(returns)
    if mean_ret > 0.10:
        verdict = "POSITIVE — mean return above 10%. But verify against live paper trading."
    elif mean_ret > 0:
        verdict = "MARGINAL — mean return positive but small. Fees and slippage may eliminate edge in live."
    else:
        verdict = "NEGATIVE — strategies do not show edge without hindsight bias."
    print(f"  Verdict: {verdict}")
    print()
    print(f"  Simulation complete. {profitable}/{NUM_SEEDS} seeds profitable.")
    print("=" * 72)


if __name__ == "__main__":
    main()
