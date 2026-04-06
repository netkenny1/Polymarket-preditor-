"""Binance BTCUSDT real-time price feed via websocket.

Maintains a rolling window of recent trades for move detection.
Used by the latency arbitrage strategy to detect BTC price moves
before they propagate to Polymarket.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()


@dataclass
class BinanceTick:
    price: float
    quantity: float
    timestamp_ms: int
    local_recv_time: float = field(default_factory=time.monotonic)

    @property
    def age_ms(self) -> float:
        return (time.monotonic() - self.local_recv_time) * 1000


class BinancePriceTracker:
    """Rolling window of Binance trades with move detection."""

    def __init__(self, window_seconds: float = 5.0) -> None:
        self._ticks: deque[BinanceTick] = deque()
        self._window = window_seconds
        self._last_price: float = 0.0
        self._vwap_price: float = 0.0

    @property
    def current_price(self) -> float:
        return self._last_price

    @property
    def vwap(self) -> float:
        return self._vwap_price

    def ingest(self, tick: BinanceTick) -> None:
        self._ticks.append(tick)
        self._last_price = tick.price
        self._prune()
        self._update_vwap()

    def get_move_pct(self) -> float:
        if len(self._ticks) < 2:
            return 0.0
        oldest = self._ticks[0].price
        if oldest == 0:
            return 0.0
        return (self._last_price - oldest) / oldest

    def get_velocity(self) -> float:
        """Price change per second over recent ticks."""
        if len(self._ticks) < 5:
            return 0.0
        recent = list(self._ticks)[-10:]
        dt = recent[-1].local_recv_time - recent[0].local_recv_time
        if dt <= 0:
            return 0.0
        return (recent[-1].price - recent[0].price) / dt

    def _prune(self) -> None:
        cutoff = time.monotonic() - self._window
        while self._ticks and self._ticks[0].local_recv_time < cutoff:
            self._ticks.popleft()

    def _update_vwap(self) -> None:
        total_pq = sum(t.price * t.quantity for t in self._ticks)
        total_q = sum(t.quantity for t in self._ticks)
        self._vwap_price = total_pq / total_q if total_q > 0 else self._last_price


class BinanceWebsocketFeed:
    """Async websocket client for Binance BTCUSDT aggTrade stream."""

    def __init__(self, ws_url: str, stream: str) -> None:
        self._url = f"{ws_url}/{stream}"
        self._tracker = BinancePriceTracker()
        self._running = False
        self._ws = None

    @property
    def tracker(self) -> BinancePriceTracker:
        return self._tracker

    async def connect(self) -> None:
        import websockets

        self._running = True
        while self._running:
            try:
                async with websockets.connect(self._url) as ws:
                    self._ws = ws
                    logger.info("binance_ws_connected", url=self._url)
                    async for message in ws:
                        self._handle_message(message)
            except Exception as e:
                logger.error("binance_ws_error", error=str(e))
                if self._running:
                    await asyncio.sleep(1.0)

    def _handle_message(self, raw: str) -> None:
        data = json.loads(raw)
        tick = BinanceTick(
            price=float(data["p"]),
            quantity=float(data["q"]),
            timestamp_ms=int(data["T"]),
        )
        self._tracker.ingest(tick)

    async def disconnect(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
