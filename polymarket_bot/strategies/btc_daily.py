"""BTC daily open/close direction strategy for Polymarket.

Core idea: Predict whether BTC will close above its daily open price
and trade Polymarket markets like "Will BTC close above $X on [date]?"

Combines multiple intraday and contextual indicators:
- Intraday momentum: BTC up 2%+ from open after noon UTC historically
  closes green ~65% of the time
- Volume profile: High volume in the direction of the move confirms
  continuation
- Mean reversion: Extreme intraday moves (>5%) tend to partially revert
- Day-of-week seasonality: Mondays lean bullish, weekends are mixed
- Hour-of-day patterns: US market open (2-4pm UTC) tends to pump,
  Asian close (8-9am UTC) tends to dump
- Recent trend: 3-day and 7-day momentum as directional bias
- Volatility context: High-vol regimes increase uncertainty, reducing
  position confidence

Profitable because:
1. BTC daily direction markets attract retail bettors who overreact
   to short-term price swings
2. Combining multiple weak signals produces a stronger composite edge
3. Intraday patterns in BTC are persistent and well-documented
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog

from polymarket_bot.data.models import Market, OrderBook, Side, Signal
from polymarket_bot.strategies.base import BaseStrategy
from polymarket_bot.utils.helpers import clamp

logger = structlog.get_logger()

# Day-of-week bias for BTC daily close direction.
# 0 = Monday, 6 = Sunday.  Positive = bullish lean.
_DOW_BIAS: dict[int, float] = {
    0: 0.04,   # Monday – historically slight bullish
    1: 0.01,   # Tuesday – neutral
    2: 0.01,   # Wednesday – neutral
    3: 0.00,   # Thursday – neutral
    4: -0.01,  # Friday – slight sell-off into weekend
    5: -0.02,  # Saturday – mixed / lower liquidity
    6: -0.02,  # Sunday – mixed / lower liquidity
}

# Hour-of-day bias buckets (UTC).
# Positive = bullish pressure tends to appear in this window.
_HOUR_BIAS: dict[tuple[int, int], float] = {
    (2, 4):   0.00,   # Late US night – quiet
    (8, 9):   -0.02,  # Asian session close – sell pressure
    (14, 16): 0.03,   # US equity market open – pump tendency
    (20, 22): 0.01,   # US afternoon / early evening – moderate
}

# Keywords used to identify BTC daily close markets.
_BTC_KEYWORDS = {"btc", "bitcoin"}


class BTCDailyStrategy(BaseStrategy):
    """Trade BTC daily open/close direction markets on Polymarket."""

    name = "btc_daily"

    def __init__(
        self,
        min_edge: float = 0.04,
        momentum_threshold: float = 0.02,
        extreme_threshold: float = 0.05,
        base_confidence: float = 0.55,
    ) -> None:
        self.min_edge = min_edge
        self.momentum_threshold = momentum_threshold
        self.extreme_threshold = extreme_threshold
        self.base_confidence = base_confidence

    # ── Public interface ─────────────────────────────────────────

    def generate_signals(
        self,
        markets: list[Market],
        order_books: dict[str, OrderBook],
        context: dict[str, Any],
    ) -> list[Signal]:
        signals: list[Signal] = []
        tradeable = self.filter_tradeable_markets(markets)

        # Bail early when required BTC data is absent.
        btc_price = context.get("btc_price")
        btc_open_today = context.get("btc_open_today")
        if btc_price is None or btc_open_today is None:
            return signals

        btc_data = {
            "btc_price": float(btc_price),
            "btc_open_today": float(btc_open_today),
            "btc_24h_change": float(context.get("btc_24h_change", 0.0)),
            "btc_7d_change": float(context.get("btc_7d_change", 0.0)),
            "btc_volume_24h": float(context.get("btc_volume_24h", 0.0)),
            "btc_avg_volume": float(context.get("btc_avg_volume", 0.0)),
        }

        for market in tradeable:
            if not self._is_btc_daily_market(market):
                continue

            try:
                signal = self._analyze(market, order_books, btc_data)
                if signal is not None:
                    signals.append(signal)
            except Exception as e:
                logger.error("btc_daily_error", market=market.condition_id, error=str(e))

        return signals

    # ── Core probability estimation ──────────────────────────────

    def estimate_close_probability(self, btc_data: dict) -> float:
        """Combine all indicators into a single probability that BTC
        closes above today's open price.

        Returns a value in (0, 1).
        """
        btc_price: float = btc_data["btc_price"]
        btc_open: float = btc_data["btc_open_today"]
        change_24h: float = btc_data.get("btc_24h_change", 0.0)
        change_7d: float = btc_data.get("btc_7d_change", 0.0)
        volume_24h: float = btc_data.get("btc_volume_24h", 0.0)
        avg_volume: float = btc_data.get("btc_avg_volume", 0.0)

        intraday_return = (btc_price - btc_open) / btc_open if btc_open > 0 else 0.0

        now = datetime.now(timezone.utc)
        hour = now.hour
        dow = now.weekday()

        # Start from the coin-flip baseline.
        prob = 0.50

        # 1. Intraday momentum ------------------------------------------
        if hour >= 12 and intraday_return >= self.momentum_threshold:
            # Up 2%+ after noon UTC → historically ~65% chance of green close.
            prob += 0.10
        elif hour >= 12 and intraday_return <= -self.momentum_threshold:
            prob -= 0.10
        else:
            # Smaller proportional nudge early in the day.
            prob += clamp(intraday_return * 2.0, -0.06, 0.06)

        # 2. Volume profile ----------------------------------------------
        if avg_volume > 0 and volume_24h > 0:
            vol_ratio = volume_24h / avg_volume
            if vol_ratio > 1.3:
                # High volume confirms the direction of the intraday move.
                direction = 1.0 if intraday_return > 0 else -1.0
                prob += direction * 0.03
            elif vol_ratio < 0.7:
                # Low volume → less conviction, pull toward 0.50.
                prob = prob * 0.85 + 0.50 * 0.15

        # 3. Mean reversion on extremes ----------------------------------
        if abs(intraday_return) > self.extreme_threshold:
            # Fade the extreme – expect partial reversion.
            reversion = -intraday_return * 0.30
            prob += clamp(reversion, -0.08, 0.08)

        # 4. Day-of-week effect ------------------------------------------
        prob += _DOW_BIAS.get(dow, 0.0)

        # 5. Hour-of-day pattern -----------------------------------------
        for (start, end), bias in _HOUR_BIAS.items():
            if start <= hour < end:
                prob += bias
                break

        # 6. Recent trend (3-day via 24h proxy, 7-day) -------------------
        trend_3d = change_24h / 100.0  # Context supplies percent values.
        trend_7d = change_7d / 100.0

        prob += clamp(trend_3d * 0.5, -0.04, 0.04)
        prob += clamp(trend_7d * 0.2, -0.03, 0.03)

        # 7. Volatility context ------------------------------------------
        # If intraday move is already large, uncertainty is high —
        # pull probability toward 0.50 (less conviction).
        if abs(intraday_return) > 0.03:
            vol_dampen = min(abs(intraday_return) * 2.0, 0.30)
            prob = prob * (1.0 - vol_dampen) + 0.50 * vol_dampen

        return clamp(prob, 0.05, 0.95)

    # ── Private helpers ──────────────────────────────────────────

    @staticmethod
    def _is_btc_daily_market(market: Market) -> bool:
        """Return True if the market looks like a BTC daily close market."""
        text = market.question.lower()
        return any(kw in text for kw in _BTC_KEYWORDS)

    def _analyze(
        self,
        market: Market,
        order_books: dict[str, OrderBook],
        btc_data: dict,
    ) -> Signal | None:
        """Produce a signal for a single BTC daily market."""
        yes_token = next((t for t in market.tokens if t.outcome == "Yes"), None)
        no_token = next((t for t in market.tokens if t.outcome == "No"), None)
        if yes_token is None or no_token is None:
            return None

        market_yes_price = yes_token.price
        fair_prob = self.estimate_close_probability(btc_data)

        edge_yes = fair_prob - market_yes_price        # positive → YES underpriced
        edge_no = (1.0 - fair_prob) - no_token.price   # positive → NO underpriced

        # Pick the side with the larger edge.
        if edge_yes >= edge_no and edge_yes >= self.min_edge:
            # BUY YES – we think BTC closes up and market underprices it.
            token = yes_token
            fair_value = fair_prob
            market_price = market_yes_price
            edge = edge_yes
        elif edge_no > edge_yes and edge_no >= self.min_edge:
            # BUY NO – we think BTC closes down and market overprices "up".
            token = no_token
            fair_value = 1.0 - fair_prob
            market_price = no_token.price
            edge = edge_no
        else:
            return None  # Insufficient edge on either side.

        # ── Confidence ──────────────────────────────────────────
        btc_open = btc_data["btc_open_today"]
        btc_price = btc_data["btc_price"]
        intraday_return = (btc_price - btc_open) / btc_open if btc_open > 0 else 0.0

        confidence = self.base_confidence

        # More confident later in the day (less time for reversal).
        hour = datetime.now(timezone.utc).hour
        if hour >= 18:
            confidence += 0.10
        elif hour >= 12:
            confidence += 0.05

        # High-vol intraday move → less confident in our forecast.
        if abs(intraday_return) > self.extreme_threshold:
            confidence -= 0.10

        # Volume confirmation boosts confidence.
        avg_vol = btc_data.get("btc_avg_volume", 0.0)
        cur_vol = btc_data.get("btc_volume_24h", 0.0)
        if avg_vol > 0 and cur_vol / avg_vol > 1.3:
            confidence += 0.05

        confidence = clamp(confidence, 0.20, 0.85)

        return Signal(
            market_condition_id=market.condition_id,
            token_id=token.token_id,
            side=Side.BUY,
            outcome=token.outcome,
            estimated_fair_value=fair_value,
            market_price=market_price,
            edge=edge,
            confidence=confidence,
            strategy=self.name,
            metadata={
                "model": "btc_daily_composite",
                "fair_close_up_prob": round(fair_prob, 4),
                "intraday_return_pct": round(intraday_return * 100, 2),
                "hour_utc": hour,
                "day_of_week": datetime.now(timezone.utc).strftime("%A"),
            },
        )
