"""Position sizing using Kelly Criterion and risk constraints.

The Kelly Criterion determines the mathematically optimal bet size
that maximizes long-term growth rate. We use fractional Kelly
(typically 1/4 Kelly) for safety - full Kelly is too aggressive
and can lead to ruin with estimation errors.

Formula: f* = (p * b - q) / b
Where:
    f* = fraction of bankroll to bet
    p = probability of winning (our estimated fair value)
    b = odds received (payout / risk)
    q = 1 - p (probability of losing)
"""

from __future__ import annotations

import structlog

from polymarket_bot.config import TradingConfig, RiskConfig
from polymarket_bot.data.models import Signal, Side
from polymarket_bot.utils.helpers import clamp

logger = structlog.get_logger()


class PositionSizer:
    """Calculate optimal position sizes using Kelly Criterion.

    Uses fractional Kelly (configurable fraction, default 0.25) to
    balance growth rate against drawdown risk. Applies hard limits
    on position size as a safety net.
    """

    def __init__(self, trading_config: TradingConfig, risk_config: RiskConfig) -> None:
        self.trading = trading_config
        self.risk = risk_config

    def calculate_kelly_fraction(self, signal: Signal) -> float:
        """Calculate the Kelly fraction for a signal.

        Returns the fraction of bankroll to bet (0 to 1).
        """
        p = signal.estimated_fair_value  # Our estimated probability of winning
        q = 1.0 - p

        if signal.side == Side.BUY:
            # Buying at market_price, pays $1 if correct
            # Risk = market_price, Reward = 1 - market_price
            b = (1.0 - signal.market_price) / signal.market_price if signal.market_price > 0 else 0
        else:
            # Selling at market_price, pays $1 if the outcome doesn't happen
            # For selling YES: risk is (1 - market_price), reward is market_price
            b = signal.market_price / (1.0 - signal.market_price) if signal.market_price < 1 else 0

        if b <= 0:
            return 0.0

        # Kelly formula: f* = (p*b - q) / b
        kelly = (p * b - q) / b

        if kelly <= 0:
            return 0.0  # Negative edge - don't bet

        # Apply fractional Kelly
        fractional = kelly * self.trading.kelly_fraction

        return clamp(fractional, 0.0, 0.25)  # Never bet more than 25% of bankroll

    def calculate_position_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_exposure: float,
    ) -> float:
        """Calculate the dollar amount to trade.

        Applies Kelly sizing with multiple safety constraints:
        1. Kelly fraction of bankroll
        2. Max single position limit
        3. Max portfolio exposure limit
        4. Per-market position limit
        5. Confidence scaling
        """
        if portfolio_value <= 0:
            return 0.0

        # 1. Kelly-optimal size
        kelly_frac = self.calculate_kelly_fraction(signal)
        kelly_size = kelly_frac * portfolio_value

        if kelly_size <= 0:
            return 0.0

        # 2. Scale by confidence (moderate scaling — Kelly already accounts for edge)
        confidence_bonus = 1.0 + (signal.confidence - 0.5) * 0.4
        confidence_scaled = kelly_size * clamp(confidence_bonus, 0.5, 1.2)

        # 3. Apply hard limits
        max_position = self.trading.max_single_position_usd
        remaining_capacity = self.trading.max_portfolio_exposure_usd - current_exposure
        per_market_limit = portfolio_value * self.risk.position_limit_per_market_pct

        size = min(
            confidence_scaled,
            max_position,
            remaining_capacity,
            per_market_limit,
        )

        # 4. Minimum viable trade (scale with portfolio size)
        min_trade = max(0.10, portfolio_value * 0.005)  # 0.5% of capital or $0.10
        if size < min_trade:
            return 0.0

        logger.debug(
            "position_sized",
            market=signal.market_condition_id,
            kelly_frac=round(kelly_frac, 4),
            kelly_size=round(kelly_size, 2),
            final_size=round(size, 2),
            edge=round(signal.edge, 4),
            confidence=round(signal.confidence, 2),
        )

        return round(size, 2)

    def calculate_size_in_shares(self, dollar_amount: float, price: float) -> float:
        """Convert dollar amount to number of shares at a given price."""
        if price <= 0 or price >= 1:
            return 0.0
        return round(dollar_amount / price, 2)
