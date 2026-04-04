"""Dynamic Kelly scaling based on recent performance.

Standard Kelly assumes perfect edge estimation. In reality, our edge
estimates are noisy. Dynamic Kelly adjusts the fraction based on:

1. Recent win rate vs expected win rate (are we calibrated?)
2. Recent P&L trajectory (winning streak = keep size, losing = reduce)
3. Portfolio heat (total risk as % of capital)
4. Regime-adaptive sizing (reduce in volatile regimes)

This is how professional quantitative traders manage bet sizing -
they don't use a fixed fraction but adapt to market conditions
and their own recent track record.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import structlog

from polymarket_bot.data.models import Signal
from polymarket_bot.risk.position_sizer import PositionSizer
from polymarket_bot.config import TradingConfig, RiskConfig
from polymarket_bot.utils.helpers import clamp

logger = structlog.get_logger()


class DynamicKellySizer(PositionSizer):
    """Adaptive position sizing that scales Kelly fraction dynamically.

    Extends the base PositionSizer with performance-aware scaling.
    """

    def __init__(
        self,
        trading_config: TradingConfig,
        risk_config: RiskConfig,
        lookback_trades: int = 50,
        min_kelly_multiplier: float = 0.3,
        max_kelly_multiplier: float = 1.5,
    ) -> None:
        super().__init__(trading_config, risk_config)
        self.lookback = lookback_trades
        self.min_multiplier = min_kelly_multiplier
        self.max_multiplier = max_kelly_multiplier

        # Track recent trade outcomes
        self._recent_pnls: deque[float] = deque(maxlen=lookback_trades)
        self._recent_edges: deque[float] = deque(maxlen=lookback_trades)
        self._recent_wins: deque[bool] = deque(maxlen=lookback_trades)
        self._current_multiplier: float = 1.0

    def record_outcome(self, predicted_edge: float, actual_pnl: float) -> None:
        """Record a trade outcome for calibration tracking."""
        self._recent_pnls.append(actual_pnl)
        self._recent_edges.append(predicted_edge)
        self._recent_wins.append(actual_pnl > 0)
        self._update_multiplier()

    def _update_multiplier(self) -> None:
        """Recalculate the Kelly multiplier based on recent performance."""
        if len(self._recent_pnls) < 5:
            self._current_multiplier = 1.0
            return

        pnls = np.array(self._recent_pnls)
        wins = np.array(self._recent_wins)

        # ── Factor 1: Win rate calibration ───────────────────────
        # If we're winning more than expected, we can size up
        actual_wr = np.mean(wins)
        expected_wr = 0.55  # Rough expected win rate for edge strategies
        calibration = actual_wr / expected_wr if expected_wr > 0 else 1.0

        # ── Factor 2: P&L trajectory ─────────────────────────────
        # Exponentially weighted recent P&L
        if len(pnls) >= 10:
            weights = np.exp(np.linspace(-1, 0, len(pnls)))
            weights /= weights.sum()
            weighted_pnl = np.sum(pnls * weights)
            # Positive trajectory = increase, negative = decrease
            trajectory = 1.0 + clamp(weighted_pnl / 100, -0.3, 0.3)
        else:
            trajectory = 1.0

        # ── Factor 3: Consecutive losses ─────────────────────────
        # After N consecutive losses, scale down aggressively
        consec_losses = 0
        for win in reversed(self._recent_wins):
            if not win:
                consec_losses += 1
            else:
                break
        loss_penalty = max(0.5, 1.0 - consec_losses * 0.1)

        # ── Factor 4: Edge accuracy ──────────────────────────────
        # How well did our predicted edges match reality?
        if len(self._recent_edges) >= 10:
            # If our predicted edges are consistently positive but P&L is negative,
            # our edge estimation is off -> reduce sizing
            avg_predicted = np.mean(np.abs(list(self._recent_edges)[-10:]))
            avg_realized = np.mean(list(self._recent_pnls)[-10:])
            if avg_predicted > 0:
                edge_accuracy = clamp(avg_realized / avg_predicted, 0.3, 1.5)
            else:
                edge_accuracy = 1.0
        else:
            edge_accuracy = 1.0

        # ── Combined multiplier ──────────────────────────────────
        multiplier = calibration * trajectory * loss_penalty * edge_accuracy
        self._current_multiplier = clamp(multiplier, self.min_multiplier, self.max_multiplier)

        logger.debug(
            "kelly_multiplier_updated",
            multiplier=round(self._current_multiplier, 3),
            calibration=round(calibration, 3),
            trajectory=round(trajectory, 3),
            loss_penalty=round(loss_penalty, 3),
            edge_accuracy=round(edge_accuracy, 3),
        )

    def calculate_position_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_exposure: float,
    ) -> float:
        """Calculate position size with dynamic Kelly scaling."""
        base_size = super().calculate_position_size(signal, portfolio_value, current_exposure)

        if base_size <= 0:
            return 0.0

        # Apply dynamic multiplier
        adjusted = base_size * self._current_multiplier

        # Re-apply hard limits
        max_position = self.trading.max_single_position_usd
        per_market_limit = portfolio_value * self.risk.position_limit_per_market_pct
        remaining = self.trading.max_portfolio_exposure_usd - current_exposure

        adjusted = min(adjusted, max_position, per_market_limit, remaining)

        if adjusted < 1.0:
            return 0.0

        return round(adjusted, 2)

    @property
    def current_multiplier(self) -> float:
        return self._current_multiplier

    def get_stats(self) -> dict[str, Any]:
        """Get current dynamic sizing statistics."""
        pnls = list(self._recent_pnls)
        wins = list(self._recent_wins)
        return {
            "kelly_multiplier": round(self._current_multiplier, 3),
            "recent_trades": len(pnls),
            "win_rate": round(sum(wins) / len(wins), 3) if wins else 0,
            "avg_pnl": round(sum(pnls) / len(pnls), 2) if pnls else 0,
            "total_recent_pnl": round(sum(pnls), 2),
        }
