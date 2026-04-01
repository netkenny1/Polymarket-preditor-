"""Main entry point for the Polymarket trading bot.

Orchestrates the full trading loop:
1. Fetch market data from Polymarket
2. Gather context (sentiment from X, external odds, etc.)
3. Run all strategies to generate signals
4. Aggregate and rank signals
5. Execute trades through risk management
6. Monitor positions and enforce stop losses
7. Repeat on a configurable interval

Usage:
    # Paper trading mode (default)
    python -m polymarket_bot.main

    # Run backtest
    python -m polymarket_bot.main --backtest

    # Live trading (requires API keys)
    PAPER_TRADING=false python -m polymarket_bot.main
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from typing import Any

import structlog

from polymarket_bot.backtesting.engine import BacktestEngine
from polymarket_bot.clients.odds_sources import OddsAggregator
from polymarket_bot.clients.polymarket import PaperTradingClient, PolymarketClient
from polymarket_bot.clients.twitter import MockTwitterClient, TwitterClient
from polymarket_bot.config import BotConfig
from polymarket_bot.data.models import Market, OrderBook
from polymarket_bot.data.store import DataStore
from polymarket_bot.execution.engine import ExecutionEngine
from polymarket_bot.risk.dynamic_kelly import DynamicKellySizer
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

logger = structlog.get_logger()


class TradingBot:
    """Main trading bot orchestrator.

    Manages the complete trading lifecycle including market data,
    strategy execution, risk management, and order execution.
    """

    def __init__(self, config: BotConfig | None = None) -> None:
        self.config = config or BotConfig.from_env()
        self.running = False
        self._setup_components()

    def _setup_components(self) -> None:
        """Initialize all bot components."""
        # Clients
        if self.config.trading.paper_trading:
            self.poly_client = PaperTradingClient(self.config.polymarket)
            logger.info("paper_trading_mode")
        else:
            self.poly_client = PolymarketClient(self.config.polymarket)
            logger.info("live_trading_mode")

        if self.config.twitter.bearer_token:
            self.twitter_client = TwitterClient(self.config.twitter)
        else:
            self.twitter_client = MockTwitterClient(self.config.twitter)
            logger.warning("using_mock_twitter", msg="No Twitter token, using mock client")

        self.odds_aggregator = OddsAggregator()
        self.data_store = DataStore()

        # Portfolio and risk
        self.portfolio = Portfolio(
            initial_cash=self.config.trading.max_portfolio_exposure_usd
        )
        self.sizer = DynamicKellySizer(self.config.trading, self.config.risk)
        self.risk_manager = RiskManager(
            self.config.risk, self.config.trading, self.portfolio, self.sizer
        )

        # Execution
        self.executor = ExecutionEngine(
            self.poly_client, self.risk_manager, self.portfolio
        )

        # Regime detection
        self.regime_detector = RegimeDetector()

        # Strategies (7 total)
        self.strategies = [
            SentimentStrategy(self.twitter_client, self.config.sentiment),
            StatisticalStrategy(self.odds_aggregator, self.config.trading.min_edge_threshold),
            MarketMakerStrategy(self.config.market_maker),
            ArbitrageStrategy(self.config.arbitrage, self.odds_aggregator),
            MomentumStrategy(min_edge=self.config.trading.min_edge_threshold),
            ContrarianStrategy(min_edge=self.config.trading.min_edge_threshold),
            TimeDecayStrategy(min_edge=self.config.trading.min_edge_threshold),
        ]

        self.aggregator = SignalAggregator(
            min_composite_edge=self.config.trading.min_edge_threshold
        )

    def run_once(self) -> dict[str, Any]:
        """Run a single trading cycle.

        Returns:
            Summary dict of actions taken.
        """
        summary: dict[str, Any] = {"signals": 0, "trades": 0, "stops": 0}

        try:
            # 1. Fetch markets
            markets = self.poly_client.get_markets(limit=50, active=True)
            if not markets:
                logger.warning("no_markets_found")
                return summary

            logger.info("markets_fetched", count=len(markets))

            # 2. Fetch order books for all tokens
            order_books: dict[str, OrderBook] = {}
            for market in markets:
                for token in market.tokens:
                    try:
                        book = self.poly_client.get_order_book(token.token_id)
                        order_books[token.token_id] = book
                    except Exception:
                        pass

            # 3. Build context
            context = self._build_context(markets)

            # 4. Generate signals from all strategies
            all_signals = []
            for strategy in self.strategies:
                try:
                    signals = strategy.generate_signals(markets, order_books, context)
                    all_signals.extend(signals)
                    logger.info("strategy_signals", strategy=strategy.name, count=len(signals))
                except Exception as e:
                    logger.error("strategy_error", strategy=strategy.name, error=str(e))

            # 5. Aggregate and rank
            ranked = self.aggregator.aggregate(all_signals)
            summary["signals"] = len(ranked)

            # 6. Execute top signals
            if ranked:
                top = ranked[:10]  # Max 10 trades per cycle
                results = self.executor.execute_signals(top)
                summary["trades"] = sum(1 for r in results if r.success)

            # 7. Update prices and check stops
            prices = {}
            for tid, book in order_books.items():
                if book.mid_price is not None:
                    prices[tid] = book.mid_price
            self.portfolio.update_prices(prices)

            stops = self.risk_manager.check_stop_losses()
            if stops:
                stop_results = self.executor.execute_stop_losses(stops)
                summary["stops"] = sum(1 for r in stop_results if r.success)

            # 8. Snapshot
            snap = self.portfolio.take_snapshot()
            risk_summary = self.risk_manager.get_risk_summary()

            logger.info(
                "cycle_complete",
                portfolio_value=round(snap.total_value, 2),
                pnl=round(self.portfolio.total_pnl, 2),
                positions=snap.num_positions,
                **summary,
            )

        except Exception as e:
            logger.error("cycle_error", error=str(e))

        return summary

    def _build_context(self, markets: list[Market]) -> dict[str, Any]:
        """Build strategy context with sentiment and position data."""
        context: dict[str, Any] = {
            "positions": {tid: pos for tid, pos in self.portfolio.positions.items()},
        }

        # Fetch sentiment for markets
        for market in markets[:20]:  # Limit API calls
            try:
                sentiment = self.twitter_client.get_market_sentiment(market.question)
                context[f"sentiment_{market.condition_id}"] = sentiment
            except Exception:
                pass

        return context

    def run(self) -> None:
        """Run the bot in a continuous loop."""
        self.running = True

        # Handle graceful shutdown
        def handle_signal(signum: int, frame: Any) -> None:
            logger.info("shutdown_signal_received")
            self.running = False

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        logger.info(
            "bot_started",
            paper_trading=self.config.trading.paper_trading,
            max_exposure=self.config.trading.max_portfolio_exposure_usd,
            interval=self.config.trading.rebalance_interval_seconds,
        )

        while self.running:
            self.run_once()

            if not self.running:
                break

            # Wait for next cycle
            time.sleep(self.config.trading.rebalance_interval_seconds)

        # Shutdown
        self._shutdown()

    def _shutdown(self) -> None:
        """Clean shutdown."""
        logger.info("shutting_down")
        self.executor.cancel_all()
        self.data_store.save_trade_history(self.portfolio.trade_history)
        self.data_store.save_snapshots(self.portfolio.snapshots)
        self.poly_client.close()
        if hasattr(self.twitter_client, "close"):
            self.twitter_client.close()
        self.odds_aggregator.close()
        logger.info("shutdown_complete", final_value=round(self.portfolio.total_value, 2))


def run_backtest(initial_capital: float = 1000.0, time_steps: int = 200, num_markets: int = 10) -> None:
    """Run a backtest and print results."""
    config = BotConfig()
    engine = BacktestEngine(config=config, initial_capital=initial_capital)
    result = engine.run(
        num_markets=num_markets,
        time_steps=time_steps,
    )
    print(result.summary())


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Polymarket Trading Bot")
    parser.add_argument("--backtest", action="store_true", help="Run backtester instead of live bot")
    parser.add_argument("--capital", type=float, default=1000.0, help="Initial capital (USD)")
    parser.add_argument("--steps", type=int, default=200, help="Backtest time steps")
    parser.add_argument("--markets", type=int, default=10, help="Number of simulated markets")
    args = parser.parse_args()

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
    )

    if args.backtest:
        run_backtest(
            initial_capital=args.capital,
            time_steps=args.steps,
            num_markets=args.markets,
        )
    else:
        bot = TradingBot()
        bot.run()


if __name__ == "__main__":
    main()
