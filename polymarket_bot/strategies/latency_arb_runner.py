"""Orchestrator for the Binance→Polymarket latency arbitrage strategy.

Ties together the Binance websocket feed, strategy, and execution into
an independent async process that runs alongside the main bot.
"""

from __future__ import annotations

import asyncio

import structlog

from polymarket_bot.clients.binance_feed import BinanceWebsocketFeed
from polymarket_bot.config import LatencyArbitrageConfig
from polymarket_bot.strategies.latency_arb import LatencyArbitrageStrategy

logger = structlog.get_logger()


class LatencyArbitrageRunner:
    """Run the latency arb as an independent async process."""

    def __init__(
        self,
        config: LatencyArbitrageConfig,
        polymarket_client: object,
    ) -> None:
        self.config = config
        self.feed = BinanceWebsocketFeed(config.binance_ws_url, config.binance_stream)
        self.strategy = LatencyArbitrageStrategy(
            config=config,
            price_tracker=self.feed.tracker,
            polymarket_client=polymarket_client,
        )

    async def run(self) -> None:
        logger.info("latency_arb_runner_starting")

        await asyncio.gather(
            self.feed.connect(),
            self.strategy.run_fast_loop(),
            self._market_refresh_loop(),
        )

    async def _market_refresh_loop(self) -> None:
        """Refresh the list of BTC markets every 5 minutes."""
        pm = self.strategy.pm_client
        while True:
            try:
                if pm is not None:
                    markets = await asyncio.to_thread(pm.get_markets, limit=200)
                    self.strategy.refresh_markets(markets)
            except Exception as e:
                logger.error("market_refresh_error", error=str(e))
            await asyncio.sleep(300)

    async def stop(self) -> None:
        await self.feed.disconnect()
