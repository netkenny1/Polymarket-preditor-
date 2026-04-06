"""Autonomous Polymarket trading runner.

Give it a budget, it finds markets, makes trades, and runs 24/7.

Usage:
    python -m polymarket_bot.runner --budget 100 --paper
    python -m polymarket_bot.runner --budget 100 --live  # Requires API keys
"""

from __future__ import annotations

import asyncio
import os
import signal as signal_mod
import sys
import time as time_mod
from datetime import datetime, timezone
from typing import Any

import structlog

from polymarket_bot.clients.crypto_feed import CryptoPriceFeed, MockCryptoPriceFeed
from polymarket_bot.clients.polymarket import PaperTradingClient, PolymarketClient
from polymarket_bot.clients.odds_sources import OddsAggregator
from polymarket_bot.clients.twitter import MockTwitterClient, TwitterClient
from polymarket_bot.config import BotConfig
from polymarket_bot.data.models import Market, OrderBook, Side
from polymarket_bot.discovery.market_scanner import MarketScanner
from polymarket_bot.execution.engine import ExecutionEngine
from polymarket_bot.execution.exit_manager import ExitManager
from polymarket_bot.risk.dynamic_kelly import DynamicKellySizer
from polymarket_bot.risk.manager import RiskManager
from polymarket_bot.risk.portfolio import Portfolio
from polymarket_bot.signals.news_reactor import NewsReactor
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
from polymarket_bot.narrative.strategy import NarrativeStrategy
from polymarket_bot.clients.economic_data import MockEconomicDataClient, EconomicDataClient

logger = structlog.get_logger()


class AutonomousRunner:
    """Fully autonomous Polymarket trading bot.

    Give it a budget, it finds markets, generates signals, and trades.
    Runs continuously with configurable intervals.
    """

    def __init__(
        self,
        budget_usd: float = 100.0,
        paper: bool = True,
        scan_interval_minutes: int = 15,
        trade_interval_minutes: int = 5,
        max_daily_trades: int = 20,
        categories: list[str] | None = None,
    ) -> None:
        self.budget_usd = budget_usd
        self.paper = paper
        self.scan_interval_minutes = scan_interval_minutes
        self.trade_interval_minutes = trade_interval_minutes
        self.max_daily_trades = max_daily_trades
        self.categories = categories
        self._running = False
        self._cycle_count = 0
        self._daily_trade_count = 0
        self._last_scan_time = 0.0
        self._last_daily_reset: datetime | None = None

        # Components initialized in _initialize()
        self.config: BotConfig = BotConfig()
        self.client: PolymarketClient | PaperTradingClient | None = None
        self.portfolio: Portfolio | None = None
        self.risk_manager: RiskManager | None = None
        self.executor: ExecutionEngine | None = None
        self.scanner: MarketScanner | None = None
        self.aggregator: SignalAggregator | None = None
        self.exit_manager: ExitManager | None = None
        self.news_reactor: NewsReactor | None = None
        self.crypto_feed: CryptoPriceFeed | None = None
        self.regime_detector: RegimeDetector | None = None
        self.strategies: list[Any] = []
        self._markets: list[Market] = []
        self._order_books: dict[str, OrderBook] = {}
        self._context: dict[str, Any] = {}

    async def start(self) -> None:
        """Main entry point. Runs forever until stopped."""
        self._print_banner()
        self._initialize()
        self._running = True

        # Handle graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal_mod.SIGINT, signal_mod.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))

        print("\n🟢 Bot is LIVE. Press Ctrl+C to stop.\n")

        while self._running:
            try:
                await self._trading_cycle()
                await asyncio.sleep(self.trade_interval_minutes * 60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("cycle_error", error=str(e))
                print(f"\n⚠️  Error in trading cycle: {e}")
                await asyncio.sleep(30)  # Back off on error

        await self.shutdown()

    def _initialize(self) -> None:
        """Set up all components."""
        self.config = BotConfig()

        # Client
        if self.paper:
            self.client = PaperTradingClient(self.config.polymarket)
        else:
            self.client = PolymarketClient(self.config.polymarket)

        # Portfolio & risk (try to restore from saved state for live trading)
        state_file = "portfolio_state.json"
        if os.path.exists(state_file) and not self.paper:
            try:
                self.portfolio = Portfolio.load_state(state_file)
                logger.info("portfolio_state_loaded", value=self.portfolio.total_value, positions=len(self.portfolio.positions))
            except Exception as e:
                logger.error("portfolio_state_load_failed", error=str(e))
                self.portfolio = Portfolio(initial_cash=self.budget_usd)
        else:
            self.portfolio = Portfolio(initial_cash=self.budget_usd)
        sizer = DynamicKellySizer(self.config.trading, self.config.risk)
        self.risk_manager = RiskManager(
            self.config.risk, self.config.trading, self.portfolio, sizer,
        )
        self.executor = ExecutionEngine(self.client, self.risk_manager, self.portfolio)

        # Discovery
        self.scanner = MarketScanner(self.client, self.config.trading)

        # Signal processing
        self.aggregator = SignalAggregator(
            min_composite_edge=self.config.trading.min_edge_threshold,
        )
        self.exit_manager = ExitManager()
        self.regime_detector = RegimeDetector()

        # News reactor
        twitter = MockTwitterClient() if self.paper else TwitterClient(self.config.twitter)
        self.news_reactor = NewsReactor(twitter)

        # Crypto feed
        self.crypto_feed = MockCryptoPriceFeed() if self.paper else CryptoPriceFeed()

        # Economic data feed
        self.economic_client = MockEconomicDataClient() if self.paper else EconomicDataClient()

        # Strategies (12 total)
        odds_agg = OddsAggregator()
        min_edge = self.config.trading.min_edge_threshold
        self.strategies = [
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
            NarrativeStrategy(
                economic_client=self.economic_client,
                news_reactor=self.news_reactor,
                min_edge=min_edge,
            ),
        ]

        logger.info(
            "initialized",
            budget=self.budget_usd,
            paper=self.paper,
            strategies=len(self.strategies),
        )

    async def _trading_cycle(self) -> None:
        """One full trading cycle."""
        self._cycle_count += 1
        self._check_daily_reset()

        loop = asyncio.get_running_loop()
        now = datetime.now(timezone.utc)

        # 1. Scan/refresh markets periodically
        if time_mod.time() - self._last_scan_time > self.scan_interval_minutes * 60:
            print(f"\n📡 Scanning markets... (cycle #{self._cycle_count})")
            try:
                all_markets = await self.scanner.scan_all_markets()
                self._markets = self.scanner.filter_tradeable(all_markets)

                # Categorize what we found
                by_time = self.scanner.categorize_by_timeframe(self._markets)
                btc_markets = self.scanner.find_btc_daily_markets(self._markets)
                trump_markets = self.scanner.find_trump_markets(self._markets)
                crypto_markets = self.scanner.find_crypto_markets(self._markets)

                print(f"   Found {len(self._markets)} tradeable markets:")
                print(f"   - Today: {len(by_time['today'])} | Week: {len(by_time['this_week'])} | Month: {len(by_time['this_month'])}")
                print(f"   - BTC daily: {len(btc_markets)} | Trump: {len(trump_markets)} | Crypto: {len(crypto_markets)}")

                self.news_reactor.build_keyword_map(self._markets)
                self._last_scan_time = time_mod.time()
            except Exception as e:
                logger.error("scan_failed", error=str(e))

        if not self._markets:
            return

        # 2. Update crypto prices and economic data
        if isinstance(self.crypto_feed, MockCryptoPriceFeed):
            self.crypto_feed.step()
        btc_ctx = self.crypto_feed.get_btc_context()
        self._context.update(btc_ctx)

        # Update economic data for narrative strategy
        if isinstance(self.economic_client, MockEconomicDataClient):
            self.economic_client.step()
        self._context["economic_indicators"] = self.economic_client.get_all_indicators()

        # Feed news events into context for narrative strategy
        if self.news_reactor and hasattr(self.news_reactor, '_event_history'):
            self._context["news_events"] = self.news_reactor._event_history[-50:]

        # 3. Fetch order books for top markets
        try:
            ranked = self.scanner.rank_markets(self._markets, self._order_books)
            top_markets = [r.market for r in ranked[:30]]  # Focus on top 30

            for market in top_markets:
                for token in market.tokens:
                    try:
                        book = await loop.run_in_executor(
                            None, self.client.get_order_book, token.token_id,
                        )
                        if book:
                            self._order_books[token.token_id] = book
                    except Exception:
                        pass
        except Exception as e:
            logger.debug("order_book_error", error=str(e))
            top_markets = self._markets[:30]

        # 4. Build context
        prices = {}
        for tid, book in self._order_books.items():
            if book.mid_price is not None:
                prices[tid] = book.mid_price
        self.portfolio.update_prices(prices)
        self._context["positions"] = dict(self.portfolio.positions)

        # 5. Detect regime for markets with history
        for m in top_markets:
            ph = self._context.get(f"price_history_{m.condition_id}", [])
            if len(ph) >= 30:
                state = self.regime_detector.detect(ph)
                self._context[f"regime_{m.condition_id}"] = state.regime.value

        # 6. Generate signals from all strategies
        all_signals = []
        for strat in self.strategies:
            try:
                sigs = strat.generate_signals(top_markets, self._order_books, self._context)
                all_signals.extend(sigs)
            except Exception as e:
                logger.debug("strategy_error", strategy=strat.name, error=str(e))

        # 7. Aggregate and rank
        ranked_signals = self.aggregator.aggregate(all_signals)

        # 8. Check exit conditions
        self.exit_manager.advance_step()
        exit_rules = self.exit_manager.check_exits(self.portfolio.positions)
        for rule in exit_rules:
            try:
                sell_size = rule.position.size * rule.sell_fraction
                price = round(rule.position.current_price * (1 - 0.005 * rule.urgency), 4)
                result = self.client.place_order(
                    token_id=rule.token_id,
                    side=Side.SELL,
                    price=max(0.01, price),
                    size=sell_size,
                    market_condition_id=rule.position.market_condition_id,
                    strategy=f"exit_{rule.exit_type}",
                )
                if result and result.success:
                    self.portfolio.process_fill(result)
                    if rule.sell_fraction >= 1.0:
                        self.exit_manager.remove_position(rule.token_id)
                    print(f"   🔄 EXIT [{rule.exit_type}]: {rule.reason}")
            except Exception as e:
                logger.error("exit_error", error=str(e))

        # 9. Execute new trades (respect daily limit)
        remaining = self.max_daily_trades - self._daily_trade_count
        if remaining > 0 and ranked_signals:
            top_signals = ranked_signals[:min(5, remaining)]
            results = self.executor.execute_signals(top_signals)

            for r in results:
                if r.success:
                    self._daily_trade_count += 1
                    sig = next((s for s in top_signals if s.token_id == r.order.token_id), None)
                    edge = sig.edge if sig else 0.0
                    strategy = sig.strategy if sig else "unknown"

                    # Register with exit manager
                    if r.order.side == Side.BUY:
                        market = next(
                            (m for m in top_markets if any(t.token_id == r.order.token_id for t in m.tokens)),
                            None,
                        )
                        self.exit_manager.register_entry(
                            r.order.token_id, edge, r.fill_size,
                            market.end_date if market else None,
                        )

                    print(
                        f"   {'🟢' if r.order.side == Side.BUY else '🔴'} "
                        f"{r.order.side.value} {r.fill_size:.1f} shares @ ${r.fill_price:.3f} "
                        f"| edge: {edge:.1%} | strategy: {strategy} "
                        f"| cost: ${r.net_cost:.2f}"
                    )

        # Circuit breaker: if we've lost more than 5% in the last hour, pause
        if hasattr(self, '_hourly_pnl_tracker'):
            self._hourly_pnl_tracker.append((time_mod.time(), self.portfolio.total_pnl))
            cutoff = time_mod.time() - 3600
            self._hourly_pnl_tracker = [(t, p) for t, p in self._hourly_pnl_tracker if t > cutoff]
            if len(self._hourly_pnl_tracker) >= 2:
                hourly_pnl = self._hourly_pnl_tracker[-1][1] - self._hourly_pnl_tracker[0][1]
                if hourly_pnl < -(self.budget_usd * 0.05):
                    logger.warning("circuit_breaker_triggered", hourly_loss=hourly_pnl)
                    print(f"   ⚡ CIRCUIT BREAKER: Lost ${abs(hourly_pnl):.2f} in last hour. Pausing 15min.")
                    await asyncio.sleep(900)
        else:
            self._hourly_pnl_tracker = [(time_mod.time(), self.portfolio.total_pnl)]

        # 10. Check stop losses
        if self.risk_manager.needs_liquidation():
            liq_orders = self.risk_manager.get_liquidation_orders()
            if liq_orders:
                liq_results = self.executor.execute_stop_losses(liq_orders)
                for r in liq_results:
                    if r.success:
                        print(f"   ⚠️  LIQUIDATION: {r.order.token_id}")
                self.risk_manager.trading_halted = False
                self.risk_manager.halt_reason = ""
                self.portfolio.peak_value = self.portfolio.total_value

        stops = self.risk_manager.check_stop_losses()
        if stops:
            stop_results = self.executor.execute_stop_losses(stops)
            for r in stop_results:
                if r.success:
                    print(f"   🛑 STOP LOSS: {r.order.token_id}")

        # 11. Print status
        if self._cycle_count % 3 == 0:  # Every 3rd cycle
            self._print_status()

        # 12. Accumulate price history
        for m in top_markets:
            key = f"price_history_{m.condition_id}"
            history = self._context.get(key, [])
            yes_token = next((t for t in m.tokens if t.outcome == "Yes"), None)
            if yes_token and yes_token.price > 0:
                history.append(yes_token.price)
                if len(history) > 200:
                    history = history[-200:]
                self._context[key] = history

        # 13. Persist portfolio state for crash recovery
        if not self.paper:
            try:
                self.portfolio.save_state()
            except Exception as e:
                logger.error("portfolio_state_save_failed", error=str(e))

    def _check_daily_reset(self) -> None:
        """Reset daily counters at midnight UTC."""
        now = datetime.now(timezone.utc)
        if self._last_daily_reset is None or now.date() > self._last_daily_reset.date():
            if self._last_daily_reset is not None:
                print(f"\n{'='*60}")
                print(self.get_daily_summary())
                print(f"{'='*60}\n")
            self._daily_trade_count = 0
            self.risk_manager.daily_pnl = 0.0
            self._last_daily_reset = now

    def _print_banner(self) -> None:
        """Print startup info."""
        mode = "PAPER TRADING" if self.paper else "🔴 LIVE TRADING"
        print(f"""
{'='*60}
  POLYMARKET AUTONOMOUS TRADING BOT
{'='*60}
  Mode:           {mode}
  Budget:         ${self.budget_usd:,.2f}
  Strategies:     13 (sentiment, statistical, market maker,
                  arbitrage, momentum, contrarian, time decay,
                  correlation, microstructure, volatility,
                  event catalyst, BTC daily, narrative analysis)
  Scan interval:  {self.scan_interval_minutes} minutes
  Trade interval: {self.trade_interval_minutes} minutes
  Max daily:      {self.max_daily_trades} trades
  Categories:     {self.categories or 'ALL'}
{'='*60}""")

    def _print_status(self) -> None:
        """Print current portfolio status."""
        p = self.portfolio
        pos_count = len(p.positions)
        print(
            f"\n   📊 Portfolio: ${p.total_value:.2f} "
            f"| P&L: ${p.total_pnl:+.2f} ({p.return_pct:+.1%}) "
            f"| Cash: ${p.cash:.2f} "
            f"| Positions: {pos_count} "
            f"| Trades today: {self._daily_trade_count}"
        )

    def get_daily_summary(self) -> str:
        """Human-readable daily performance summary."""
        p = self.portfolio
        return (
            f"  📈 DAILY SUMMARY ({datetime.now(timezone.utc).strftime('%Y-%m-%d')})\n"
            f"  Portfolio:    ${p.total_value:,.2f}\n"
            f"  Total P&L:    ${p.total_pnl:+,.2f} ({p.return_pct:+.1%})\n"
            f"  Cash:         ${p.cash:,.2f}\n"
            f"  Positions:    {len(p.positions)}\n"
            f"  Trades today: {self._daily_trade_count}"
        )

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        print("\n\n🔴 Shutting down...")
        self._running = False

        if not self.paper and self.client and self.portfolio:
            try:
                self.portfolio.save_state()
                self.client.cancel_all_orders()
                logger.info("shutdown_orders_cancelled")
            except Exception as e:
                logger.error("shutdown_cleanup_failed", error=str(e))

        if self.portfolio and self.portfolio.positions:
            print(f"   {len(self.portfolio.positions)} open positions remain.")

        print(self.get_daily_summary())
        print("\n   Bot stopped. Goodbye.\n")

        if self.crypto_feed:
            self.crypto_feed.close()


def main() -> None:
    """CLI entry point for the autonomous runner."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Polymarket Autonomous Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m polymarket_bot.runner --budget 100 --paper
  python -m polymarket_bot.runner --budget 500 --live --categories crypto politics
  python -m polymarket_bot.runner --budget 100 --scan-interval 30 --max-daily-trades 10
        """,
    )
    parser.add_argument("--budget", type=float, default=100.0, help="Trading budget in USD (default: 100)")
    parser.add_argument("--paper", action="store_true", default=True, help="Paper trading mode (default)")
    parser.add_argument("--live", action="store_true", help="Live trading (requires .env with API keys)")
    parser.add_argument("--scan-interval", type=int, default=15, help="Market scan interval in minutes (default: 15)")
    parser.add_argument("--trade-interval", type=int, default=5, help="Trade evaluation interval in minutes (default: 5)")
    parser.add_argument("--categories", nargs="*", help="Market categories: crypto, politics, sports")
    parser.add_argument("--max-daily-trades", type=int, default=20, help="Maximum trades per day (default: 20)")

    args = parser.parse_args()

    runner = AutonomousRunner(
        budget_usd=args.budget,
        paper=not args.live,
        scan_interval_minutes=args.scan_interval,
        trade_interval_minutes=args.trade_interval,
        max_daily_trades=args.max_daily_trades,
        categories=args.categories,
    )

    asyncio.run(runner.start())


if __name__ == "__main__":
    main()
