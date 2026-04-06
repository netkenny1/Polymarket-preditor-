#!/usr/bin/env python3
"""Backtest against REAL Polymarket historical data.

Fetches actual price histories from the Polymarket CLOB API,
replays them step by step, and runs the full strategy ensemble
against real market dynamics.

Usage:
    python run_historical_backtest.py [--days 14] [--markets 15]
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import structlog

from polymarket_bot.clients.economic_data import MockEconomicDataClient
from polymarket_bot.clients.odds_sources import OddsAggregator
from polymarket_bot.clients.polymarket import PaperTradingClient
from polymarket_bot.clients.twitter import MockTwitterClient
from polymarket_bot.config import BotConfig, PolymarketConfig, SentimentConfig
from polymarket_bot.data.historical import HistoricalDataFetcher
from polymarket_bot.data.models import (
    Market,
    OrderBook,
    OrderBookLevel,
    SentimentData,
    Side,
)
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

INITIAL_CAPITAL = 100.0


def build_synthetic_book(price: float, depth: float = 500.0) -> OrderBook:
    """Build a realistic order book from a historical price point."""
    spread = max(0.02, abs(1.0 - 2 * price) * 0.03 + 0.01)
    bids, asks = [], []
    for i in range(5):
        bp = max(0.01, price - spread / 2 - i * 0.01)
        ap = min(0.99, price + spread / 2 + i * 0.01)
        sz = depth / 5 * (5 - i) / 5
        bids.append(OrderBookLevel(price=round(bp, 2), size=round(sz, 2)))
        asks.append(OrderBookLevel(price=round(ap, 2), size=round(sz, 2)))
    return OrderBook(token_id="", bids=bids, asks=asks)


def run_historical_backtest(
    num_markets: int = 15,
    days_back: int = 14,
    min_liquidity: float = 2000.0,
) -> dict:
    """Run backtest against real historical Polymarket data."""

    print("=" * 72)
    print("  HISTORICAL DATA BACKTEST — Real Polymarket Prices")
    print(f"  Markets: {num_markets} | Period: {days_back} days | Min Liquidity: ${min_liquidity}")
    print("=" * 72)
    print()

    # ── Fetch Data ────────────────────────────────────────────────
    print("Fetching historical data from Polymarket API...")
    fetcher = HistoricalDataFetcher()
    try:
        dataset = fetcher.build_backtest_dataset(
            num_markets=num_markets,
            days_back=days_back,
            min_liquidity=min_liquidity,
            interval="1h",
        )
    finally:
        fetcher.close()

    markets = dataset["markets"]
    price_histories = dataset["price_histories"]
    timestamps = dataset["timestamps"]

    if not markets:
        print("ERROR: No markets with sufficient data found.")
        return {"error": "no_data"}

    # Find common time range
    all_lengths = [len(price_histories[t.token_id])
                   for m in markets for t in m.tokens
                   if t.token_id in price_histories]
    if not all_lengths:
        print("ERROR: No price data found.")
        return {"error": "no_prices"}

    num_steps = min(all_lengths)
    print(f"Found {len(markets)} markets, {num_steps} time steps")
    for m in markets:
        print(f"  - {m.question[:70]}")
    print()

    # ── Setup ─────────────────────────────────────────────────────
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
    t0 = time.time()

    # ── Replay Loop ───────────────────────────────────────────────
    for step in range(num_steps):
        if step % 24 == 0:
            risk_mgr.daily_pnl = 0.0
            if step % 120 == 0:
                mock_econ.step()

        # Update prices from historical data
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

        volumes = {tid: m.liquidity for m in markets for tid in [t.token_id for t in m.tokens]}
        client.set_simulated_prices(prices)
        client.set_simulated_volumes(volumes)
        portfolio.update_prices(prices)

        # Build order books and context from real prices
        order_books: dict[str, OrderBook] = {}
        context: dict = {"positions": dict(portfolio.positions)}

        for market in markets:
            for token in market.tokens:
                tid = token.token_id
                if tid in prices:
                    book = build_synthetic_book(prices[tid], depth=market.liquidity / 10)
                    book = OrderBook(token_id=tid, bids=book.bids, asks=book.asks)
                    order_books[tid] = book

            cid = market.condition_id
            # Build price history from actual data up to this step
            yes_token = next((t for t in market.tokens if t.outcome == "Yes"), None)
            if yes_token and yes_token.token_id in price_histories:
                hist = price_histories[yes_token.token_id][:step + 1]
                context[f"price_history_{cid}"] = hist
                if len(hist) >= 30:
                    regime_state = regime_detector.detect(hist)
                    context[f"regime_{cid}"] = regime_state.regime.value

                # Sentiment from real price trend
                if len(hist) >= 5:
                    trend = hist[-1] - hist[-5]
                    noisy = max(-1.0, min(1.0, trend * 5.0 + np.random.normal(0, 0.2)))
                    bullish = max(0, (noisy + 1) / 2)
                    context[f"sentiment_{cid}"] = SentimentData(
                        query=market.question[:50],
                        tweet_count=int(np.random.uniform(5, 50)),
                        avg_sentiment=noisy,
                        sentiment_std=0.3,
                        volume_ratio=np.random.uniform(0.5, 2.0),
                        bullish_pct=bullish,
                        bearish_pct=1.0 - bullish,
                    )

        context["economic_indicators"] = mock_econ.get_all_indicators()
        context["btc_price"] = 85000.0 * (1 + np.random.normal(0, 0.003))
        context["btc_open_today"] = context.get("btc_price", 85000.0)

        # Generate and execute signals
        all_signals = []
        for strat in strategies:
            try:
                signals = strat.generate_signals(markets, order_books, context)
                all_signals.extend(signals)
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
                strategy_trades[name] = strategy_trades.get(name, 0) + 1
                edge = next((s.edge for s in top if s.token_id == r.order.token_id), 0.05)
                cp = prices.get(r.order.token_id, r.fill_price)
                pnl = ((cp - r.fill_price) * r.fill_size if r.order.side == Side.BUY
                       else (r.fill_price - cp) * r.fill_size)
                dynamic_sizer.record_outcome(edge, pnl)

        # Exit management
        for r in results:
            if r.success and r.order.side == Side.BUY:
                edge = next((s.edge for s in top if s.token_id == r.order.token_id), 0.05)
                mkt = next((m for m in markets if any(t.token_id == r.order.token_id for t in m.tokens)), None)
                exit_manager.register_entry(r.order.token_id, edge, r.fill_size,
                                            mkt.end_date if mkt else None)

        exit_manager.advance_step()
        for rule in exit_manager.check_exits(portfolio.positions):
            try:
                sell_size = rule.position.size * rule.sell_fraction
                price = round(rule.position.current_price * (1 - 0.005 * rule.urgency), 4)
                er = client.place_order(rule.token_id, Side.SELL, max(0.01, price), sell_size,
                                        rule.position.market_condition_id, f"exit_{rule.exit_type}")
                if er and er.success:
                    portfolio.process_fill(er)
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

        if step % 24 == 0:
            day = step // 24
            print(f"  Day {day:3d}: ${portfolio.total_value:8.2f} ({portfolio.return_pct:+7.1%}) | "
                  f"Fills {total_fills}/{total_trades}")

    # ── Force close at end ────────────────────────────────────────
    for tid, pos in list(portfolio.positions.items()):
        if pos.size > 0:
            r = client.place_order(tid, Side.SELL, pos.current_price, pos.size,
                                   pos.market_condition_id, "close_end")
            if r.success:
                portfolio.process_fill(r)

    elapsed = time.time() - t0
    final = portfolio.total_value
    ret = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL
    equity = np.array(portfolio_values)
    daily_ret = np.diff(equity) / equity[:-1]
    sharpe = calculate_sharpe_ratio(list(daily_ret))
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.where(peak > 0, peak, 1)
    max_dd = float(np.max(dd))

    # ── Report ────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  HISTORICAL BACKTEST RESULTS")
    print("=" * 72)
    print(f"  Period:            {days_back} days of real Polymarket data")
    print(f"  Markets:           {len(markets)}")
    print(f"  Time Steps:        {num_steps} ({num_steps/24:.0f} days)")
    print(f"  Final Value:       ${final:.2f}")
    print(f"  Return:            {ret:+.1%}")
    print(f"  Sharpe:            {sharpe:.2f}")
    print(f"  Max Drawdown:      {max_dd:.1%}")
    print(f"  Total Fills:       {total_fills}/{total_trades} ({client.fill_rate:.0%} fill rate)")
    print(f"  Time:              {elapsed:.0f}s")
    print()

    print("  STRATEGY ATTRIBUTION")
    print("  " + "-" * 50)
    for name, count in sorted(strategy_trades.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {name:55s} {count:4d}")

    verdict = ""
    if ret > 0.05:
        verdict = "POSITIVE edge detected on real data. Promising for live trading."
    elif ret > -0.03:
        verdict = "MARGINAL — roughly break-even. Strategies need more edge for live profitability."
    else:
        verdict = "NEGATIVE — strategies lost money on real data. Do NOT deploy live without changes."

    print()
    print(f"  VERDICT: {verdict}")
    print("=" * 72)

    return {
        "final": final, "return": ret, "sharpe": sharpe,
        "max_dd": max_dd, "trades": total_fills, "markets": len(markets),
        "strategy_trades": strategy_trades,
    }


def main():
    parser = argparse.ArgumentParser(description="Historical Polymarket backtest")
    parser.add_argument("--days", type=int, default=14, help="Days of history")
    parser.add_argument("--markets", type=int, default=15, help="Number of markets")
    parser.add_argument("--liquidity", type=float, default=2000.0, help="Min market liquidity")
    args = parser.parse_args()

    run_historical_backtest(
        num_markets=args.markets,
        days_back=args.days,
        min_liquidity=args.liquidity,
    )


if __name__ == "__main__":
    main()
