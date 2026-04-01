"""Utility functions for the trading bot."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timedelta
from typing import Any

import structlog

logger = structlog.get_logger()


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide safely, returning default if denominator is zero."""
    if abs(denominator) < 1e-10:
        return default
    return numerator / denominator


def exponential_decay(value: float, half_life_minutes: float, elapsed_minutes: float) -> float:
    """Apply exponential decay to a value."""
    if half_life_minutes <= 0:
        return 0.0
    decay_factor = 0.5 ** (elapsed_minutes / half_life_minutes)
    return value * decay_factor


def compute_vwap(prices: list[float], volumes: list[float]) -> float:
    """Compute volume-weighted average price."""
    if not prices or not volumes or len(prices) != len(volumes):
        return 0.0
    total_volume = sum(volumes)
    if total_volume == 0:
        return 0.0
    return sum(p * v for p, v in zip(prices, volumes)) / total_volume


def round_price(price: float, tick_size: float = 0.01) -> float:
    """Round price to nearest tick."""
    return round(round(price / tick_size) * tick_size, 4)


def generate_order_id(prefix: str = "bot") -> str:
    """Generate a unique order ID."""
    timestamp = str(time.time_ns())
    hash_val = hashlib.md5(timestamp.encode()).hexdigest()[:8]
    return f"{prefix}_{hash_val}"


def is_market_open(end_date: datetime | None) -> bool:
    """Check if a market is still open for trading."""
    if end_date is None:
        return True
    return datetime.utcnow() < end_date


def time_to_expiry_hours(end_date: datetime | None) -> float:
    """Calculate hours until market expiry."""
    if end_date is None:
        return float("inf")
    delta = end_date - datetime.utcnow()
    return max(0.0, delta.total_seconds() / 3600)


def categorize_market(question: str, tags: list[str]) -> str:
    """Heuristically categorize a market based on question text and tags."""
    text = question.lower() + " ".join(t.lower() for t in tags)

    crypto_keywords = ["bitcoin", "btc", "ethereum", "eth", "crypto", "token", "defi", "solana", "price"]
    politics_keywords = ["president", "election", "senate", "congress", "vote", "trump", "biden", "democrat", "republican", "governor", "parliament"]
    sports_keywords = ["nba", "nfl", "mlb", "nhl", "soccer", "football", "basketball", "baseball", "championship", "super bowl", "world cup", "win"]

    scores = {
        "crypto": sum(1 for kw in crypto_keywords if kw in text),
        "politics": sum(1 for kw in politics_keywords if kw in text),
        "sports": sum(1 for kw in sports_keywords if kw in text),
    }

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "other"
    return best


def format_usd(amount: float) -> str:
    """Format a dollar amount."""
    return f"${amount:,.2f}"


def calculate_sharpe_ratio(
    returns: list[float], risk_free_rate: float = 0.0, annualization: float = 365.0
) -> float:
    """Calculate annualized Sharpe ratio from a list of returns."""
    if len(returns) < 2:
        return 0.0
    import numpy as np

    arr = np.array(returns)
    excess = arr - risk_free_rate / annualization
    mean_excess = np.mean(excess)
    std = np.std(excess, ddof=1)
    if std < 1e-10:
        return 0.0
    return float(mean_excess / std * np.sqrt(annualization))
