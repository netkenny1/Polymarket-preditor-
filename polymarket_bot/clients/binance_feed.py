"""Binance BTCUSDT market data — real-time websocket + historical klines.

Live:  wss://stream.binance.com:9443/ws/btcusdt@bookTicker  (stable mid)
REST:  https://api.binance.com/api/v3/klines                (historical 1m)

The bookTicker stream publishes the best bid/ask on every book update,
which gives a much more stable "current price" than @aggTrade. For the
BTC 5-minute strategy we only need a single latest mid price.

For backtesting we pull 1-minute OHLCV klines via the REST API, cached
to a local JSON so re-runs are fast and deterministic.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import httpx
import structlog

logger = structlog.get_logger()


@dataclass
class BinanceTick:
    """A single top-of-book snapshot."""

    bid: float
    ask: float
    timestamp_ms: int
    local_recv_time: float = field(default_factory=time.monotonic)

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def age_ms(self) -> float:
        return (time.monotonic() - self.local_recv_time) * 1000


class BinancePriceTracker:
    """Rolling window of Binance mid prices.

    Keeps the last `window_seconds` of ticks and provides the current
    mid, a rolling mid list, and simple price-move helpers. Safe to
    share between the async WS reader and the strategy thread.
    """

    def __init__(self, window_seconds: float = 600.0) -> None:
        self._ticks: deque[BinanceTick] = deque()
        self._window = window_seconds
        self._last_mid: float = 0.0

    @property
    def current_price(self) -> float:
        """Most recent mid price, or 0.0 if no ticks yet."""
        return self._last_mid

    def ingest(self, tick: BinanceTick) -> None:
        self._ticks.append(tick)
        self._last_mid = tick.mid
        self._prune()

    def get_move_pct(self, window_seconds: float | None = None) -> float:
        """Percent move from oldest tick in the window to latest."""
        if len(self._ticks) < 2:
            return 0.0
        if window_seconds is None:
            oldest = self._ticks[0]
        else:
            cutoff = time.monotonic() - window_seconds
            oldest = self._ticks[0]
            for t in self._ticks:
                if t.local_recv_time >= cutoff:
                    oldest = t
                    break
        if oldest.mid <= 0:
            return 0.0
        return (self._last_mid - oldest.mid) / oldest.mid

    def last_n_mids(self, n: int) -> list[float]:
        if n <= 0:
            return []
        return [t.mid for t in list(self._ticks)[-n:]]

    def _prune(self) -> None:
        cutoff = time.monotonic() - self._window
        while self._ticks and self._ticks[0].local_recv_time < cutoff:
            self._ticks.popleft()


class BinanceWebsocketFeed:
    """Async websocket client for Binance BTCUSDT bookTicker stream."""

    def __init__(
        self,
        ws_url: str = "wss://stream.binance.com:9443/ws",
        stream: str = "btcusdt@bookTicker",
        tracker: BinancePriceTracker | None = None,
    ) -> None:
        self._url = f"{ws_url}/{stream}"
        self._tracker = tracker or BinancePriceTracker()
        self._running = False
        self._ws = None

    @property
    def tracker(self) -> BinancePriceTracker:
        return self._tracker

    async def connect(self) -> None:
        import websockets  # local import so test envs without it still import the module

        self._running = True
        while self._running:
            try:
                async with websockets.connect(self._url) as ws:
                    self._ws = ws
                    logger.info("binance_ws_connected", url=self._url)
                    async for message in ws:
                        self._handle_message(message)
            except Exception as e:  # pragma: no cover - network retry
                logger.error("binance_ws_error", error=str(e))
                if self._running:
                    await asyncio.sleep(1.0)

    def _handle_message(self, raw: str) -> None:
        data = json.loads(raw)
        # bookTicker payload: {"u":..., "s":"BTCUSDT", "b":"...", "B":"...", "a":"...", "A":"..."}
        # aggTrade payload has "p" for price — handle both for robustness.
        try:
            if "b" in data and "a" in data:
                bid = float(data["b"])
                ask = float(data["a"])
            elif "p" in data:
                px = float(data["p"])
                bid = ask = px
            else:
                return
        except (KeyError, TypeError, ValueError):
            return
        ts_ms = int(data.get("T") or data.get("E") or int(time.time() * 1000))
        self._tracker.ingest(BinanceTick(bid=bid, ask=ask, timestamp_ms=ts_ms))

    async def disconnect(self) -> None:
        self._running = False
        if self._ws is not None:
            await self._ws.close()


# ── Historical klines (REST) ────────────────────────────────────────

@dataclass
class Kline:
    """A single Binance 1-minute candle."""

    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def close_time_ms(self) -> int:
        return self.open_time_ms + 60_000 - 1


class BinanceHistoricalClient:
    """Fetch + cache Binance 1-minute BTCUSDT klines for backtesting.

    Uses the public `/api/v3/klines` endpoint (no API key required). Caps
    at 1000 bars per call, paginates automatically, caches to disk.
    """

    REST_URL = "https://api.binance.com"
    MAX_PER_CALL = 1000

    def __init__(
        self,
        cache_dir: str | os.PathLike = ".cache/binance",
        symbol: str = "BTCUSDT",
        interval: str = "1m",
    ) -> None:
        self.symbol = symbol
        self.interval = interval
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = httpx.Client(base_url=self.REST_URL, timeout=30.0)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "BinanceHistoricalClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _cache_path(self, start_ms: int, end_ms: int) -> Path:
        key = f"{self.symbol}_{self.interval}_{start_ms}_{end_ms}"
        h = hashlib.md5(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{self.symbol}_{self.interval}_{h}.json"

    def fetch_klines(self, start_ms: int, end_ms: int) -> list[Kline]:
        """Fetch all 1-minute klines between `start_ms` and `end_ms` (inclusive)."""
        cache = self._cache_path(start_ms, end_ms)
        if cache.exists():
            try:
                raw = json.loads(cache.read_text())
                return [Kline(**k) for k in raw]
            except Exception:  # pragma: no cover - corrupt cache
                cache.unlink(missing_ok=True)

        klines: list[Kline] = []
        cursor = start_ms
        while cursor < end_ms:
            params = {
                "symbol": self.symbol,
                "interval": self.interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": self.MAX_PER_CALL,
            }
            try:
                resp = self._client.get("/api/v3/klines", params=params)
                resp.raise_for_status()
                batch = resp.json()
            except httpx.HTTPError as e:
                logger.error("binance_klines_fetch_failed", error=str(e), cursor=cursor)
                break
            if not batch:
                break
            for row in batch:
                klines.append(
                    Kline(
                        open_time_ms=int(row[0]),
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=float(row[5]),
                    )
                )
            last_open = int(batch[-1][0])
            if last_open + 60_000 >= end_ms or len(batch) < self.MAX_PER_CALL:
                break
            cursor = last_open + 60_000

        try:
            cache.write_text(json.dumps([k.__dict__ for k in klines]))
        except Exception:  # pragma: no cover
            pass
        logger.info(
            "binance_klines_fetched",
            symbol=self.symbol,
            interval=self.interval,
            count=len(klines),
            start_ms=start_ms,
            end_ms=end_ms,
        )
        return klines

    def fetch_last_n_days(self, days: int) -> list[Kline]:
        now_ms = int(time.time() * 1000)
        start = now_ms - days * 86_400_000
        return self.fetch_klines(start, now_ms)

    @staticmethod
    def closes(klines: Iterable[Kline]) -> list[float]:
        return [k.close for k in klines]
