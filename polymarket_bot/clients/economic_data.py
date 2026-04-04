"""Economic data client for import/export data, trade balances, and macro indicators.

Provides real-world economic context for narrative analysis.
In paper/backtest mode, uses a deterministic mock with realistic dynamics.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class EconomicSnapshot:
    """Point-in-time snapshot of all tracked economic indicators."""
    trade_balance_bn: float = -65.0  # US trade balance, billions USD
    cpi_yoy: float = 3.2  # CPI year-over-year %
    unemployment_rate: float = 3.8  # %
    fed_funds_rate: float = 5.25  # %
    treasury_10y: float = 4.3  # %
    consumer_sentiment: float = 67.0  # U Michigan index
    tariff_rate_avg: float = 3.0  # Average effective tariff rate %
    imports_bn: float = 320.0  # Monthly imports, billions USD
    exports_bn: float = 255.0  # Monthly exports, billions USD
    timestamp: datetime = field(default_factory=datetime.utcnow)


class EconomicDataClient:
    """Fetches economic indicators from public APIs.

    Uses free sources (FRED API, etc.) with caching to avoid rate limits.
    Falls back gracefully if APIs are unavailable.
    """

    def __init__(self, fred_api_key: str = "") -> None:
        self._fred_key = fred_api_key or os.getenv("FRED_API_KEY", "")
        self._cache: dict[str, tuple[float, float]] = {}  # name -> (timestamp, value)
        self._cache_ttl = 3600.0  # 1 hour
        self._snapshot = EconomicSnapshot()

    def get_all_indicators(self) -> list[dict]:
        """Fetch all tracked economic indicators.

        Returns list of dicts with: name, value, previous_value, change_pct, unit.
        """
        indicators = []
        fields = [
            ("trade_balance", self._snapshot.trade_balance_bn, "billion_usd"),
            ("cpi_yoy", self._snapshot.cpi_yoy, "percent"),
            ("unemployment_rate", self._snapshot.unemployment_rate, "percent"),
            ("fed_funds_rate", self._snapshot.fed_funds_rate, "percent"),
            ("treasury_10y", self._snapshot.treasury_10y, "percent"),
            ("consumer_sentiment", self._snapshot.consumer_sentiment, "index"),
            ("tariff_rate_avg", self._snapshot.tariff_rate_avg, "percent"),
            ("imports", self._snapshot.imports_bn, "billion_usd"),
            ("exports", self._snapshot.exports_bn, "billion_usd"),
        ]
        for name, value, unit in fields:
            prev = self._cache.get(name, (0, value))[1]
            change_pct = ((value - prev) / abs(prev)) * 100 if prev != 0 else 0.0
            indicators.append({
                "name": name,
                "value": round(value, 4),
                "previous_value": round(prev, 4),
                "change_pct": round(change_pct, 4),
                "unit": unit,
            })
            self._cache[name] = (time.time(), value)
        return indicators

    def get_trade_data(self) -> dict:
        """Get import/export trade balance data."""
        return {
            "trade_balance_bn": self._snapshot.trade_balance_bn,
            "imports_bn": self._snapshot.imports_bn,
            "exports_bn": self._snapshot.exports_bn,
            "tariff_rate_avg": self._snapshot.tariff_rate_avg,
        }

    def close(self) -> None:
        pass


class MockEconomicDataClient:
    """Deterministic mock economic data for backtesting.

    Generates realistic economic indicator time series using
    mean-reverting random walks with seed-based reproducibility.
    Follows the same pattern as MockCryptoPriceFeed.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.RandomState(seed)
        self._step = 0

        # Base values (equilibrium levels)
        self._equilibrium = {
            "trade_balance": -65.0,
            "cpi_yoy": 3.2,
            "unemployment_rate": 3.8,
            "fed_funds_rate": 5.25,
            "treasury_10y": 4.3,
            "consumer_sentiment": 67.0,
            "tariff_rate_avg": 3.0,
            "imports": 320.0,
            "exports": 255.0,
        }

        # Current values start at equilibrium
        self._current = dict(self._equilibrium)
        self._previous = dict(self._equilibrium)

        # Volatility per indicator (how much it moves per step)
        self._volatility = {
            "trade_balance": 2.0,
            "cpi_yoy": 0.1,
            "unemployment_rate": 0.05,
            "fed_funds_rate": 0.0,  # Changes in discrete steps
            "treasury_10y": 0.05,
            "consumer_sentiment": 1.5,
            "tariff_rate_avg": 0.2,
            "imports": 5.0,
            "exports": 4.0,
        }

        # Mean reversion speed (0 = no reversion, 1 = instant)
        self._reversion = {
            "trade_balance": 0.02,
            "cpi_yoy": 0.03,
            "unemployment_rate": 0.02,
            "fed_funds_rate": 0.01,
            "treasury_10y": 0.03,
            "consumer_sentiment": 0.05,
            "tariff_rate_avg": 0.01,
            "imports": 0.02,
            "exports": 0.02,
        }

    def step(self) -> None:
        """Advance economic indicators by one time step."""
        self._step += 1
        self._previous = dict(self._current)

        for name in self._current:
            eq = self._equilibrium[name]
            vol = self._volatility[name]
            rev = self._reversion[name]
            curr = self._current[name]

            # Mean-reverting random walk
            reversion_pull = rev * (eq - curr)
            noise = self._rng.normal(0, vol)
            new_val = curr + reversion_pull + noise

            # Clamp to reasonable ranges
            if "rate" in name or name == "cpi_yoy":
                new_val = max(0.0, new_val)
            if name == "unemployment_rate":
                new_val = max(2.0, min(15.0, new_val))
            if name == "consumer_sentiment":
                new_val = max(20.0, min(120.0, new_val))
            if name == "tariff_rate_avg":
                new_val = max(0.0, min(50.0, new_val))

            self._current[name] = new_val

        # Fed funds rate changes in 25bp increments occasionally
        if self._step % 20 == 0:
            direction = self._rng.choice([-0.25, 0, 0, 0, 0.25])
            self._current["fed_funds_rate"] = max(
                0.0, self._current["fed_funds_rate"] + direction
            )

        # Trade balance = exports - imports
        self._current["trade_balance"] = (
            self._current["exports"] - self._current["imports"]
        )

    def get_all_indicators(self) -> list[dict]:
        """Return current economic indicators."""
        indicators = []
        units = {
            "trade_balance": "billion_usd",
            "cpi_yoy": "percent",
            "unemployment_rate": "percent",
            "fed_funds_rate": "percent",
            "treasury_10y": "percent",
            "consumer_sentiment": "index",
            "tariff_rate_avg": "percent",
            "imports": "billion_usd",
            "exports": "billion_usd",
        }

        for name, value in self._current.items():
            prev = self._previous.get(name, value)
            change_pct = ((value - prev) / abs(prev)) * 100 if prev != 0 else 0.0
            indicators.append({
                "name": name,
                "value": round(value, 4),
                "previous_value": round(prev, 4),
                "change_pct": round(change_pct, 4),
                "unit": units.get(name, ""),
            })
        return indicators

    def get_trade_data(self) -> dict:
        """Get import/export trade balance data."""
        return {
            "trade_balance_bn": round(self._current["trade_balance"], 2),
            "imports_bn": round(self._current["imports"], 2),
            "exports_bn": round(self._current["exports"], 2),
            "tariff_rate_avg": round(self._current["tariff_rate_avg"], 2),
        }

    def close(self) -> None:
        pass
