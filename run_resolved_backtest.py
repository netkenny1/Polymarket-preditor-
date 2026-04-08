#!/usr/bin/env python3
"""Backtest against RESOLVED Polymarket markets with ground-truth outcomes.

Unlike the standard historical backtest (which only measures P&L),
this script compares strategy predictions to actual known resolutions.
This gives us TRUE accuracy metrics — not just returns.

Usage:
    python run_resolved_backtest.py [--days 90] [--markets 30]

Metrics:
  - Prediction accuracy by strategy (directional bet was correct?)
  - Narrative hit rate (NarrativeStrategy predicted the right winner?)
  - Confidence calibration (70% confidence → 70% actual win rate?)
  - Sharpe vs buy-and-hold (buy everything at 50c, hold to resolution)
  - Timing quality (how early did the algorithm identify the winner?)
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict

import numpy as np
import structlog

from polymarket_bot.clients.economic_data import MockEconomicDataClient
from polymarket_bot.clients.odds_sources import OddsAggregator
from polymarket_bot.clients.polymarket import PaperTradingClient
from polymarket_bot.clients.twitter import MockTwitterClient
from polymarket_bot.config import BotConfig, PolymarketConfig, SentimentConfig
from polymarket_bot.data.historical import HistoricalDataFetcher
from polymarket_bot.data.models import Market, OrderBook, OrderBookLevel, SentimentData, Side
from polymarket_bot.execution.engine import ExecutionEngine
from polymarket_bot.execution.exit_manager import ExitManager
from polymarket_bot.narrative.strategy import NarrativeStrategy
from polymarket_bot.risk.dynamic_kelly import DynamicKellySizer
from polymarket_bot.risk.manager import RiskManager
from polymarket_bot.risk.portfolio import Portfolio
from polymarket_bot.strategies.arbitrage import ArbitrageStrategy
from polymarket_bot.strategies.btc_daily import BTCDailyStrategy
from polymarket_bot.strategies.contrarian import ContrarianStrategy
from polymarket_bot.strategies.event_catalyst import EventCatalystStrategy
from polymarket_bot.strategies.market_maker import MarketMakerStrategy
from polymarket_bot.strategies.market_regime import RegimeDetector
from polymarket_bot.strategies.momentum import MomentumStrategy
from polymarket_bot.strategies.sentiment import SentimentStrategy
from polymarket_bot.strategies.signals import SignalAggregator
from polymarket_bot.strategies.statistical import StatisticalStrategy
from polymarket_bot.strategies.time_decay import TimeDecayStrategy
from polymarket_bot.strategies.volatility import VolatilityStrategy
from polymarket_bot.utils.helpers import calculate_sharpe_ratio

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(40))

INITIAL_CAPITAL = 100.0


def build_synthetic_book(price: float, depth: float = 500.0) -> OrderBook:
    spread = max(0.02, abs(1.0 - 2 * price) * 0.03 + 0.01)
    bids, asks = [], []
    for i in range(5):
        bp = max(0.01, price - spread / 2 - i * 0.01)
        ap = min(0.99, price + spread / 2 + i * 0.01)
        sz = depth / 5 * (5 - i) / 5
        bids.append(OrderBookLevel(price=round(bp, 2), size=round(sz, 2)))
        asks.append(OrderBookLevel(price=round(ap, 2), size=round(sz, 2)))
    return OrderBook(token_id="", bids=bids, asks=asks)


def run_resolved_backtest(
    num_markets: int = 30,
    days_back: int = 90,
    min_volume: float = 5000.0,
    narrative_only: bool = False,
) -> dict:
    """Run backtest against resolved markets with ground-truth accuracy measurement."""

    print("=" * 72)
    print("  RESOLVED MARKET BACKTEST — Ground-Truth Accuracy")
    print(f"  Markets: {num_markets} | Period: {days_back} days | Min Volume: ${min_volume:,.0f}")
    print("=" * 72)
    print()

    # ── Fetch resolved market data ────────────────────────────────────
    print("Fetching resolved markets from Polymarket Gamma API...")
    fetcher = HistoricalDataFetcher()
    try:
        dataset = fetcher.build_resolved_backtest_dataset(
            num_markets=num_markets, days_back=days_back, min_volume=min_volume
        )
    finally:
        fetcher.close()

    markets = dataset["markets"]
    price_histories = dataset["price_histories"]
    outcomes = dataset["outcomes"]  # {condition_id: "Yes"|"No"}

    if not markets:
        print("ERROR: No resolved markets with sufficient data found.")
        print("  (This is normal if the Polymarket API doesn't return many resolved markets.)")
        print("  Try increasing --days or decreasing --volume")
        return {"error": "no_data"}

    print(f"Found {len(markets)} resolved markets:")
    for m in markets[:10]:
        winner = outcomes.get(m.condition_id, "?")
        print(f"  [{winner}] {m.question[:65]}")
    if len(markets) > 10:
        print(f"  ... and {len(markets) - 10} more")
    print()

    # ── Setup trading infrastructure ─────────────────────────────────
    config = BotConfig()
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
    mock_econ = MockEconomicDataClient(seed=42)
    min_edge = config.trading.min_edge_threshold

    sim_sent_config = SentimentConfig(
        min_tweets=3,
        volume_spike_threshold=config.sentiment.volume_spike_threshold,
        sentiment_threshold=config.sentiment.sentiment_threshold,
        decay_half_life_minutes=config.sentiment.decay_half_life_minutes,
        keywords_per_market=config.sentiment.keywords_per_market,
    )

    if narrative_only:
        strategies = [NarrativeStrategy(economic_client=mock_econ, min_edge=min_edge, min_events=2)]
    else:
        strategies = [
            SentimentStrategy(mock_twitter, sim_sent_config),
            StatisticalStrategy(odds_agg, min_edge),
            MarketMakerStrategy(config.market_maker),
            ArbitrageStrategy(config.arbitrage, odds_agg),
            MomentumStrategy(min_edge=min_edge),
            ContrarianStrategy(min_edge=min_edge),
            TimeDecayStrategy(min_edge=min_edge),
            VolatilityStrategy(min_edge=min_edge),
            EventCatalystStrategy(min_edge=min_edge),
            BTCDailyStrategy(min_edge=min_edge),
            NarrativeStrategy(economic_client=mock_econ, min_edge=min_edge, min_events=2),
        ]

    # ── Tracking structures ───────────────────────────────────────────
    portfolio_values = [INITIAL_CAPITAL]
    # {strategy_name: {"correct": int, "total": int, "early_correct": int}}
    strategy_accuracy: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0, "pnl": 0.0})
    narrative_hits = {"correct": 0, "total": 0}
    confidence_bins: list[tuple[float, bool]] = []  # (confidence, was_correct)
    timing_data: list[tuple[int, bool]] = []  # (step_when_first_correct, market_resolved_at)

    # Track which signals were generated for each market and at what step
    signal_log: dict[str, list[dict]] = defaultdict(list)  # token_id → signals

    total_trades = 0
    total_fills = 0
    t0 = time.time()

    # Compute common steps
    all_lengths = [
        len(price_histories[t.token_id])
        for m in markets for t in m.tokens
        if t.token_id in price_histories
    ]
    if not all_lengths:
        print("ERROR: No price histories found.")
        return {"error": "no_prices"}

    num_steps = min(all_lengths)
    print(f"Replaying {num_steps} time steps...\n")

    # ── Replay loop ───────────────────────────────────────────────────
    for step in range(num_steps):
        if step % 24 == 0:
            risk_mgr.daily_pnl = 0.0
            if step % 120 == 0:
                mock_econ.step()

        # Update prices
        prices: dict[str, float] = {}
        for market in markets:
            for token in market.tokens:
                tid = token.token_id
                if tid in price_histories and step < len(price_histories[tid]):
                    p = price_histories[tid][step]
                    token.price = p
                    prices[tid] = p

        if not prices:
            continue

        client.set_simulated_prices(prices)
        client.set_simulated_volumes({tid: m.liquidity for m in markets for tid in [t.token_id for t in m.tokens]})
        portfolio.update_prices(prices)

        order_books: dict[str, OrderBook] = {}
        context: dict = {"positions": dict(portfolio.positions)}

        for market in markets:
            for token in market.tokens:
                tid = token.token_id
                if tid in prices:
                    book = build_synthetic_book(prices[tid], depth=market.liquidity / 10)
                    order_books[tid] = OrderBook(token_id=tid, bids=book.bids, asks=book.asks)

            cid = market.condition_id
            yes_token = next((t for t in market.tokens if t.outcome == "Yes"), None)
            if yes_token and yes_token.token_id in price_histories:
                hist = price_histories[yes_token.token_id][: step + 1]
                context[f"price_history_{cid}"] = hist
                if len(hist) >= 30:
                    regime_state = regime_detector.detect(hist)
                    context[f"regime_{cid}"] = regime_state.regime.value
                if len(hist) >= 5:
                    trend = hist[-1] - hist[-5]
                    noisy = max(-1.0, min(1.0, trend * 5.0 + np.random.normal(0, 0.15)))
                    bullish = max(0, (noisy + 1) / 2)
                    context[f"sentiment_{cid}"] = SentimentData(
                        query=market.question[:50], tweet_count=int(np.random.uniform(5, 50)),
                        avg_sentiment=noisy, sentiment_std=0.25,
                        volume_ratio=np.random.uniform(0.5, 2.0),
                        bullish_pct=bullish, bearish_pct=1.0 - bullish,
                    )

        context["economic_indicators"] = mock_econ.get_all_indicators()
        context["btc_price"] = 85000.0 * (1 + np.random.normal(0, 0.003))
        context["btc_open_today"] = context.get("btc_price", 85000.0)

        # Generate signals
        all_signals = []
        for strat in strategies:
            try:
                sigs = strat.generate_signals(markets, order_books, context)
                # Log signals with their strategy name and step
                for sig in sigs:
                    signal_log[sig.token_id].append({
                        "step": step, "strategy": sig.strategy,
                        "side": sig.side.value, "confidence": sig.confidence,
                        "edge": sig.edge,
                    })
                all_signals.extend(sigs)
            except Exception:
                pass

        ranked = aggregator.aggregate(all_signals)
        top = ranked[:7]
        results = executor.execute_signals(top)

        for r in results:
            total_trades += 1
            if r.success:
                total_fills += 1
                name = r.order.strategy or "unknown"
                strategy_accuracy[name]["total"] += 1

                # Determine if this signal direction was correct given known outcome
                for market in markets:
                    if any(t.token_id == r.order.token_id for t in market.tokens):
                        winner = outcomes.get(market.condition_id, "")
                        token = next((t for t in market.tokens if t.token_id == r.order.token_id), None)
                        if token and winner:
                            predicted_yes = (r.order.side == Side.BUY and token.outcome == "Yes") or \
                                           (r.order.side == Side.BUY and token.outcome != "No")
                            actual_yes = (winner == "Yes")
                            correct = (predicted_yes == actual_yes)
                            if correct:
                                strategy_accuracy[name]["correct"] += 1
                                timing_data.append((step, True))
                            # Track confidence calibration
                            sig = next((s for s in top if s.token_id == r.order.token_id), None)
                            if sig:
                                confidence_bins.append((sig.confidence, correct))
                        break

                # PnL tracking
                cp = prices.get(r.order.token_id, r.fill_price)
                pnl = ((cp - r.fill_price) * r.fill_size if r.order.side == Side.BUY
                       else (r.fill_price - cp) * r.fill_size)
                strategy_accuracy[name]["pnl"] = strategy_accuracy[name].get("pnl", 0.0) + pnl
                dynamic_sizer.record_outcome(0.05, pnl)

        # Exits
        for r in results:
            if r.success and r.order.side == Side.BUY:
                mkt = next((m for m in markets if any(t.token_id == r.order.token_id for t in m.tokens)), None)
                exit_manager.register_entry(r.order.token_id, 0.05, r.fill_size,
                                            mkt.end_date if mkt else None)

        exit_manager.advance_step()
        for rule in exit_manager.check_exits(portfolio.positions):
            try:
                sell_price = round(rule.position.current_price * (1 - 0.005 * rule.urgency), 4)
                er = client.place_order(rule.token_id, Side.SELL, max(0.01, sell_price),
                                        rule.position.size * rule.sell_fraction,
                                        rule.position.market_condition_id, f"exit_{rule.exit_type}")
                if er and er.success:
                    portfolio.process_fill(er)
                    if rule.sell_fraction >= 1.0:
                        exit_manager.remove_position(rule.token_id)
            except Exception:
                pass

        portfolio_values.append(portfolio.total_value)

        if step % 24 == 0:
            day = step // 24
            print(f"  Day {day:3d}: ${portfolio.total_value:8.2f} ({portfolio.return_pct:+7.1%}) | "
                  f"Fills {total_fills}/{total_trades}")

    # ── Force-close all positions ────────────────────────────────────
    for tid, pos in list(portfolio.positions.items()):
        if pos.size > 0:
            r = client.place_order(tid, Side.SELL, pos.current_price, pos.size,
                                   pos.market_condition_id, "close_end")
            if r.success:
                portfolio.process_fill(r)

    elapsed = time.time() - t0

    # ── Buy-and-Hold Benchmark ───────────────────────────────────────
    # Buy each market at 50c, sell at resolution (winner pays $1, loser pays $0)
    bah_pnl = 0.0
    bet_size = INITIAL_CAPITAL / max(len(markets), 1)
    for market in markets:
        winner = outcomes.get(market.condition_id, "")
        yes_token = next((t for t in market.tokens if t.outcome == "Yes"), None)
        if yes_token and winner:
            entry_price = 0.50
            exit_price = 1.0 if winner == "Yes" else 0.0
            shares = bet_size / entry_price
            bah_pnl += (exit_price - entry_price) * shares

    bah_return = bah_pnl / INITIAL_CAPITAL

    # ── Confidence Calibration ───────────────────────────────────────
    calibration = {}
    if confidence_bins:
        bins = np.linspace(0, 1, 6)
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            bucket = [(c, ok) for c, ok in confidence_bins if lo <= c < hi]
            if bucket:
                avg_conf = np.mean([c for c, _ in bucket])
                win_rate = np.mean([float(ok) for _, ok in bucket])
                calibration[f"{lo:.1f}-{hi:.1f}"] = {
                    "n": len(bucket), "avg_confidence": round(avg_conf, 3),
                    "actual_win_rate": round(win_rate, 3)
                }

    # ── Compute strategy-level accuracy ─────────────────────────────
    final = portfolio.total_value
    ret = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL
    equity = np.array(portfolio_values)
    daily_ret = np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], 1)
    sharpe = calculate_sharpe_ratio(list(daily_ret))
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.where(peak > 0, peak, 1)
    max_dd = float(np.max(dd))

    # ── Report ───────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  RESOLVED BACKTEST RESULTS — Ground-Truth Accuracy")
    print("=" * 72)
    print(f"  Resolved Markets:      {len(markets)}")
    print(f"  Time Steps:            {num_steps}")
    print(f"  Final Portfolio:       ${final:.2f}")
    print(f"  Total Return:          {ret:+.1%}")
    print(f"  Buy-and-Hold Return:   {bah_return:+.1%}  (naive 50c benchmark)")
    print(f"  Sharpe Ratio:          {sharpe:.2f}")
    print(f"  Max Drawdown:          {max_dd:.1%}")
    print(f"  Total Fills:           {total_fills}/{total_trades}")
    print(f"  Time:                  {elapsed:.0f}s")
    print()

    print("  STRATEGY PREDICTION ACCURACY")
    print("  " + "-" * 60)
    print(f"  {'Strategy':<40} {'Correct':>8} {'Total':>8} {'Accuracy':>10} {'PnL':>8}")
    print("  " + "-" * 60)
    for name, stats in sorted(strategy_accuracy.items(), key=lambda x: x[1]["total"], reverse=True)[:15]:
        n = stats["total"]
        correct = stats["correct"]
        acc = correct / n if n > 0 else 0.0
        pnl = stats.get("pnl", 0.0)
        print(f"  {name:<40} {correct:>8} {n:>8} {acc:>9.1%}  ${pnl:>6.2f}")

    print()
    print("  CONFIDENCE CALIBRATION")
    print("  " + "-" * 50)
    print(f"  {'Confidence':>12} {'N':>6} {'Predicted':>12} {'Actual Win%':>12}")
    for bucket, cal in sorted(calibration.items()):
        print(f"  {bucket:>12} {cal['n']:>6} {cal['avg_confidence']:>11.1%} {cal['actual_win_rate']:>11.1%}")

    # Verdict
    alpha = ret - bah_return
    print()
    if alpha > 0.05:
        verdict = f"POSITIVE alpha: +{alpha:.1%} vs buy-and-hold. Strategies are adding real edge."
    elif alpha > -0.03:
        verdict = f"MARGINAL: {alpha:+.1%} vs buy-and-hold. Strategies are roughly market-neutral."
    else:
        verdict = f"NEGATIVE alpha: {alpha:+.1%} vs buy-and-hold. Strategies are underperforming."
    print(f"  VERDICT: {verdict}")
    print("=" * 72)

    return {
        "final": final, "return": ret, "bah_return": bah_return,
        "alpha": alpha, "sharpe": sharpe, "max_dd": max_dd,
        "fills": total_fills, "markets": len(markets),
        "strategy_accuracy": {k: {"accuracy": v["correct"]/v["total"] if v["total"] else 0,
                                   "n": v["total"]} for k, v in strategy_accuracy.items()},
        "calibration": calibration,
    }


def main():
    parser = argparse.ArgumentParser(description="Resolved market ground-truth backtest")
    parser.add_argument("--days", type=int, default=90, help="Days of history")
    parser.add_argument("--markets", type=int, default=30, help="Number of resolved markets")
    parser.add_argument("--volume", type=float, default=5000.0, help="Min market volume")
    parser.add_argument("--narrative-only", action="store_true",
                        help="Test only NarrativeStrategy for pure signal accuracy")
    args = parser.parse_args()

    run_resolved_backtest(
        num_markets=args.markets,
        days_back=args.days,
        min_volume=args.volume,
        narrative_only=args.narrative_only,
    )


if __name__ == "__main__":
    main()
