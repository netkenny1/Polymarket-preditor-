"""Crypto price feed client for real-time BTC/ETH/SOL data.

Uses CoinGecko's free API as the primary source (no API key needed).
Includes a mock implementation for backtesting and testing.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx
import numpy as np
import structlog

logger = structlog.get_logger()

# CoinGecko symbol -> API id mapping
_SYMBOL_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "DOGE": "dogecoin",
    "XRP": "ripple",
    "MATIC": "matic-network",
    "AVAX": "avalanche-2",
    "ADA": "cardano",
}

_COINGECKO_BASE = "https://api.coingecko.com/api/v3"


@dataclass
class CryptoPrice:
    """Snapshot of a crypto asset's price data."""

    symbol: str
    price: float
    open_today: float
    high_24h: float
    low_24h: float
    change_24h_pct: float
    change_7d_pct: float
    volume_24h: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def intraday_change_pct(self) -> float:
        if self.open_today > 0:
            return (self.price - self.open_today) / self.open_today
        return 0.0

    @property
    def is_up_today(self) -> bool:
        return self.price > self.open_today


class CryptoPriceFeed:
    """Fetch real-time crypto prices from CoinGecko (free, no key needed).

    Rate limit: ~10-30 calls/min on free tier.
    Caches results for 60 seconds to stay within limits.
    """

    def __init__(self, use_mock: bool = False, cache_ttl: int = 60) -> None:
        self.use_mock = use_mock
        self._cache: dict[str, tuple[CryptoPrice, float]] = {}
        self._cache_ttl = cache_ttl
        self._price_history: dict[str, list[float]] = defaultdict(list)
        self._client = httpx.Client(timeout=10.0)
        self._last_request_time = 0.0
        self._min_request_interval = 2.0  # seconds between API calls

    def get_price(self, symbol: str = "BTC") -> CryptoPrice:
        """Get current price for a crypto asset."""
        if self.use_mock:
            return self._mock_price(symbol)

        # Check cache
        now = time.time()
        cached = self._cache.get(symbol)
        if cached and (now - cached[1]) < self._cache_ttl:
            return cached[0]

        price = self._fetch_from_coingecko(symbol)
        self._cache[symbol] = (price, now)
        self._price_history[symbol].append(price.price)

        # Keep history bounded
        if len(self._price_history[symbol]) > 1000:
            self._price_history[symbol] = self._price_history[symbol][-500:]

        return price

    def get_all_prices(self) -> dict[str, CryptoPrice]:
        """Get prices for BTC, ETH, SOL."""
        result = {}
        for symbol in ["BTC", "ETH", "SOL"]:
            try:
                result[symbol] = self.get_price(symbol)
            except Exception as e:
                logger.error("price_fetch_failed", symbol=symbol, error=str(e))
        return result

    def get_price_history(self, symbol: str) -> list[float]:
        """Get cached price history (builds up over time)."""
        return list(self._price_history.get(symbol, []))

    def get_btc_context(self) -> dict[str, Any]:
        """Build context dict for BTC daily strategy."""
        try:
            btc = self.get_price("BTC")
            return {
                "btc_price": btc.price,
                "btc_open_today": btc.open_today,
                "btc_24h_change": btc.change_24h_pct,
                "btc_7d_change": btc.change_7d_pct,
                "btc_intraday_pct": btc.intraday_change_pct,
                "btc_volume_24h": btc.volume_24h,
                "btc_high_24h": btc.high_24h,
                "btc_low_24h": btc.low_24h,
                "btc_avg_volume": btc.volume_24h,  # Will improve with history
            }
        except Exception as e:
            logger.error("btc_context_failed", error=str(e))
            return {}

    def _fetch_from_coingecko(self, symbol: str) -> CryptoPrice:
        """Fetch from CoinGecko free API."""
        coin_id = _SYMBOL_MAP.get(symbol.upper(), symbol.lower())

        # Rate limit
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)

        try:
            # Get current price + 24h data
            resp = self._client.get(
                f"{_COINGECKO_BASE}/coins/{coin_id}",
                params={
                    "localization": "false",
                    "tickers": "false",
                    "community_data": "false",
                    "developer_data": "false",
                },
            )
            self._last_request_time = time.time()
            resp.raise_for_status()
            data = resp.json()

            market_data = data.get("market_data", {})
            current_price = market_data.get("current_price", {}).get("usd", 0.0)
            high_24h = market_data.get("high_24h", {}).get("usd", current_price)
            low_24h = market_data.get("low_24h", {}).get("usd", current_price)
            change_24h = market_data.get("price_change_percentage_24h", 0.0)
            change_7d = market_data.get("price_change_percentage_7d", 0.0)
            volume_24h = market_data.get("total_volume", {}).get("usd", 0.0)

            # Approximate today's open from 24h change
            open_today = current_price / (1 + change_24h / 100) if change_24h else current_price

            return CryptoPrice(
                symbol=symbol.upper(),
                price=current_price,
                open_today=open_today,
                high_24h=high_24h,
                low_24h=low_24h,
                change_24h_pct=change_24h,
                change_7d_pct=change_7d,
                volume_24h=volume_24h,
            )

        except Exception as e:
            logger.error("coingecko_fetch_failed", symbol=symbol, error=str(e))
            # Return mock on failure
            return self._mock_price(symbol)

    def _mock_price(self, symbol: str) -> CryptoPrice:
        """Generate realistic mock price data."""
        base_prices = {"BTC": 67000, "ETH": 3200, "SOL": 140, "DOGE": 0.15}
        base = base_prices.get(symbol.upper(), 100.0)
        noise = np.random.normal(0, base * 0.01)

        return CryptoPrice(
            symbol=symbol.upper(),
            price=base + noise,
            open_today=base,
            high_24h=base * 1.02,
            low_24h=base * 0.98,
            change_24h_pct=round(noise / base * 100, 2),
            change_7d_pct=round(np.random.normal(0, 3), 2),
            volume_24h=base * 400000,
        )

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()


class MockCryptoPriceFeed(CryptoPriceFeed):
    """Mock feed for backtesting and testing."""

    def __init__(
        self,
        base_prices: dict[str, float] | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(use_mock=True)
        self._base = base_prices or {"BTC": 67000, "ETH": 3200, "SOL": 140}
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._current: dict[str, float] = dict(self._base)
        self._open: dict[str, float] = dict(self._base)

    def step(self) -> None:
        """Advance mock prices by one step (random walk)."""
        self._step += 1
        for sym in self._current:
            # Mean-reverting random walk with slight upward drift
            base = self._base[sym]
            current = self._current[sym]
            drift = 0.0001 * base
            revert = 0.01 * (base - current)
            noise = self._rng.normal(0, base * 0.005)
            self._current[sym] = max(base * 0.5, current + drift + revert + noise)

        # Reset open every 24 steps ("new day")
        if self._step % 24 == 0:
            self._open = dict(self._current)

    def get_price(self, symbol: str = "BTC") -> CryptoPrice:
        sym = symbol.upper()
        price = self._current.get(sym, 100.0)
        open_price = self._open.get(sym, price)

        return CryptoPrice(
            symbol=sym,
            price=price,
            open_today=open_price,
            high_24h=price * 1.01,
            low_24h=price * 0.99,
            change_24h_pct=((price - open_price) / open_price * 100) if open_price else 0,
            change_7d_pct=self._rng.normal(0, 3),
            volume_24h=self._base.get(sym, 100) * 400000,
        )
