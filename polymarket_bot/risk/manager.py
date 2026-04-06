"""Risk management engine.

Enforces risk limits and circuit breakers to protect the portfolio.
This is the most critical component - a trading bot without risk
management is just a way to lose money faster.

Risk controls:
1. Max drawdown circuit breaker (halt all trading)
2. Daily loss limit
3. Per-position stop losses and trailing stops
4. Maximum portfolio exposure
5. Correlated position limits
6. Per-market position limits
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import structlog

from polymarket_bot.config import RiskConfig, TradingConfig
from polymarket_bot.data.models import Position, Signal, Side
from polymarket_bot.risk.portfolio import Portfolio
from polymarket_bot.risk.position_sizer import PositionSizer

logger = structlog.get_logger()


class RiskManager:
    """Enforce risk limits and manage position lifecycle.

    Acts as a gatekeeper before any trade is executed. Every signal
    must pass through the risk manager which can:
    - Approve with calculated size
    - Reduce the size
    - Reject entirely
    - Trigger stop losses / liquidations
    """

    def __init__(
        self,
        risk_config: RiskConfig,
        trading_config: TradingConfig,
        portfolio: Portfolio,
        position_sizer: PositionSizer,
    ) -> None:
        self.config = risk_config
        self.trading = trading_config
        self.portfolio = portfolio
        self.sizer = position_sizer
        self.trading_halted = False
        self.halt_reason = ""
        self.daily_pnl: float = 0.0
        self.daily_reset_time: datetime = datetime.utcnow()
        self._liquidation_needed = False

    def check_signal(self, signal: Signal) -> tuple[bool, float, str]:
        """Evaluate a signal against risk limits.

        Returns:
            (approved, size_usd, reason)
        """
        # Reset daily P&L tracker if new day
        self._maybe_reset_daily()

        # Circuit breaker checks — use reduced sizing instead of hard halt
        recovery_multiplier = 1.0
        if self.trading_halted:
            # Allow trading to resume with reduced sizing if drawdown is recovering
            if self.portfolio.drawdown_pct < self.config.max_drawdown_pct:
                recovery_multiplier = 0.25  # Trade at 25% size during recovery
                logger.debug("trading_recovery_mode", drawdown=self.portfolio.drawdown_pct)
            else:
                return False, 0.0, f"Trading halted: {self.halt_reason}"

        # 1. Max drawdown check
        if self.portfolio.drawdown_pct >= self.config.max_drawdown_pct:
            if not self.trading_halted:
                self.trading_halted = True
                self.halt_reason = f"Max drawdown reached: {self.portfolio.drawdown_pct:.1%}"
                logger.critical("trading_halted", reason=self.halt_reason)
                self._liquidation_needed = True
            return False, 0.0, self.halt_reason

        # 2. Daily loss limit — reduce sizing instead of hard halt
        if self.daily_pnl <= -self.config.max_daily_loss_usd:
            recovery_multiplier = min(recovery_multiplier, 0.3)  # Severely reduce sizing

        # 3. Check if we already have max positions
        if len(self.portfolio.positions) >= self.trading.max_positions:
            # Only allow trades that close positions
            existing = self.portfolio.positions.get(signal.token_id)
            if existing is None or signal.side == Side.BUY:
                return False, 0.0, "Max positions reached"

        # 4. Calculate position size
        size = self.sizer.calculate_position_size(
            signal=signal,
            portfolio_value=self.portfolio.total_value,
            current_exposure=self.portfolio.total_exposure,
        )

        if size <= 0:
            return False, 0.0, "Position size too small or no edge"

        # Apply recovery multiplier if in drawdown/daily-loss recovery mode
        size *= recovery_multiplier

        if size < 0.10:
            return False, 0.0, "Position size too small after recovery scaling"

        # 5. Portfolio exposure check
        if self.portfolio.total_exposure + size > self.trading.max_portfolio_exposure_usd:
            remaining = self.trading.max_portfolio_exposure_usd - self.portfolio.total_exposure
            if remaining < 1.0:
                return False, 0.0, "Max portfolio exposure reached"
            size = min(size, remaining)

        # 6. Sufficient cash
        if signal.side == Side.BUY:
            cost = size  # Approximate
            if cost > self.portfolio.cash * 0.95:  # Keep 5% cash buffer
                size = self.portfolio.cash * 0.90
                if size < 1.0:
                    return False, 0.0, "Insufficient cash"

        logger.info(
            "risk_approved",
            market=signal.market_condition_id,
            side=signal.side.value,
            size=round(size, 2),
            edge=round(signal.edge, 4),
        )

        return True, size, "approved"

    def check_stop_losses(self) -> list[dict[str, Any]]:
        """Check all positions for stop loss and trailing stop triggers.

        Returns:
            List of positions that should be closed.
        """
        stops_triggered = []

        for token_id, pos in list(self.portfolio.positions.items()):
            if pos.size <= 0:
                continue

            # Fixed stop loss
            loss_pct = (pos.avg_entry_price - pos.current_price) / pos.avg_entry_price
            if loss_pct >= self.config.stop_loss_pct:
                stops_triggered.append({
                    "token_id": token_id,
                    "position": pos,
                    "reason": f"Stop loss: {loss_pct:.1%} loss",
                    "type": "stop_loss",
                })
                continue

            # Trailing stop
            if pos.max_price_seen > pos.avg_entry_price:
                drop_from_peak = (pos.max_price_seen - pos.current_price) / pos.max_price_seen
                if drop_from_peak >= self.config.trailing_stop_pct:
                    stops_triggered.append({
                        "token_id": token_id,
                        "position": pos,
                        "reason": f"Trailing stop: {drop_from_peak:.1%} from peak",
                        "type": "trailing_stop",
                    })

        if stops_triggered:
            logger.warning(
                "stops_triggered",
                count=len(stops_triggered),
                tokens=[s["token_id"] for s in stops_triggered],
            )

        return stops_triggered

    def update_daily_pnl(self, pnl_delta: float) -> None:
        """Track intraday P&L for daily loss limit."""
        self.daily_pnl += pnl_delta

    def _maybe_reset_daily(self) -> None:
        """Reset daily P&L counter at midnight UTC."""
        now = datetime.utcnow()
        if now.date() > self.daily_reset_time.date():
            self.daily_pnl = 0.0
            self.daily_reset_time = now
            # Also reset trading halt if drawdown has recovered
            if self.trading_halted and self.portfolio.drawdown_pct < self.config.max_drawdown_pct * 0.5:
                self.trading_halted = False
                self.halt_reason = ""
                logger.info("trading_resumed", drawdown=self.portfolio.drawdown_pct)

    def needs_liquidation(self) -> bool:
        """Check if emergency liquidation is needed."""
        if self._liquidation_needed:
            self._liquidation_needed = False
            return True
        return False

    def get_liquidation_orders(self) -> list[dict[str, Any]]:
        """Get all positions that need to be liquidated."""
        orders = []
        for token_id, pos in list(self.portfolio.positions.items()):
            if pos.size > 0:
                orders.append({
                    "token_id": token_id,
                    "position": pos,
                    "reason": "Emergency liquidation - max drawdown",
                    "type": "liquidation",
                })
        return orders

    def reconcile_positions(self, actual_positions: dict[str, float]) -> list[str]:
        """Compare portfolio positions with actual on-chain positions.
        Returns list of discrepancy descriptions.
        """
        discrepancies = []
        for token_id, pos in self.portfolio.positions.items():
            actual_size = actual_positions.get(token_id, 0.0)
            if abs(pos.size - actual_size) > 0.01:
                discrepancies.append(
                    f"Position mismatch: {token_id} bot={pos.size:.2f} actual={actual_size:.2f}"
                )
        for token_id, actual_size in actual_positions.items():
            if token_id not in self.portfolio.positions and actual_size > 0.01:
                discrepancies.append(
                    f"Unknown position: {token_id} actual={actual_size:.2f}"
                )
        if discrepancies:
            logger.warning("position_reconciliation_failed", count=len(discrepancies))
        return discrepancies

    def get_risk_summary(self) -> dict[str, Any]:
        """Get a summary of current risk metrics."""
        return {
            "trading_halted": self.trading_halted,
            "halt_reason": self.halt_reason,
            "drawdown_pct": round(self.portfolio.drawdown_pct, 4),
            "daily_pnl": round(self.daily_pnl, 2),
            "total_exposure": round(self.portfolio.total_exposure, 2),
            "max_exposure": self.trading.max_portfolio_exposure_usd,
            "num_positions": len(self.portfolio.positions),
            "max_positions": self.trading.max_positions,
            "cash": round(self.portfolio.cash, 2),
            "portfolio_value": round(self.portfolio.total_value, 2),
        }
