"""Portfolio analytics and performance reporting.

Provides advanced risk metrics and performance attribution beyond the
basic stats computed by the backtesting engine. Designed to work with
portfolio value series, trade histories, and return streams produced
by Portfolio and BacktestEngine.

Metrics include:
- Sortino ratio (downside-risk-adjusted return)
- Calmar ratio (return vs max drawdown)
- Value at Risk and Conditional VaR (expected shortfall)
- Consecutive win/loss streaks
- Drawdown recovery time
- Strategy-level P&L attribution
- Hour-of-day performance breakdown
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from polymarket_bot.data.models import TradeResult, Side
from polymarket_bot.utils.helpers import calculate_sharpe_ratio


@dataclass
class StrategyStats:
    """Performance statistics for a single strategy."""

    strategy: str
    num_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    sharpe: float = 0.0


@dataclass
class AnalyticsReport:
    """Comprehensive portfolio analytics report.

    Aggregates all risk and performance metrics into a single object
    that can be logged, serialized, or displayed.
    """

    # Return metrics
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Risk metrics
    max_drawdown_pct: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0

    # Streak metrics
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Recovery
    recovery_time_steps: int = 0

    # Attribution
    strategy_attribution: dict[str, StrategyStats] = field(default_factory=dict)
    hourly_performance: dict[int, float] = field(default_factory=dict)

    # Trade summary
    num_trades: int = 0
    win_rate: float = 0.0

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            "",
            "=" * 60,
            "  PORTFOLIO ANALYTICS REPORT",
            "=" * 60,
            f"  Total Return:           {self.total_return_pct:+.2%}",
            f"  Sharpe Ratio:           {self.sharpe_ratio:.3f}",
            f"  Sortino Ratio:          {self.sortino_ratio:.3f}",
            f"  Calmar Ratio:           {self.calmar_ratio:.3f}",
            f"  Max Drawdown:           {self.max_drawdown_pct:.2%}",
            f"  VaR (95%):              {self.var_95:.4f}",
            f"  CVaR (95%):             {self.cvar_95:.4f}",
            f"  Max Consecutive Wins:   {self.max_consecutive_wins}",
            f"  Max Consecutive Losses: {self.max_consecutive_losses}",
            f"  Recovery Time (steps):  {self.recovery_time_steps}",
            f"  Total Trades:           {self.num_trades}",
            f"  Win Rate:               {self.win_rate:.1%}",
        ]

        if self.strategy_attribution:
            lines.append("")
            lines.append("  Strategy Attribution:")
            lines.append(
                f"  {'Strategy':<20} {'Trades':>7} {'PnL':>10} "
                f"{'Win%':>7} {'AvgPnL':>10} {'Sharpe':>7}"
            )
            lines.append("  " + "-" * 63)
            for name, stats in sorted(
                self.strategy_attribution.items(),
                key=lambda x: x[1].total_pnl,
                reverse=True,
            ):
                lines.append(
                    f"  {name:<20} {stats.num_trades:>7} "
                    f"${stats.total_pnl:>9.2f} "
                    f"{stats.win_rate:>6.1%} "
                    f"${stats.avg_pnl:>9.4f} "
                    f"{stats.sharpe:>7.2f}"
                )

        if self.hourly_performance:
            lines.append("")
            lines.append("  Hourly Performance (top 5 hours):")
            sorted_hours = sorted(
                self.hourly_performance.items(), key=lambda x: x[1], reverse=True
            )
            for hour, pnl in sorted_hours[:5]:
                lines.append(f"    Hour {hour:02d}:00  ${pnl:>+10.4f}")

        lines.append("=" * 60)
        return "\n".join(lines)


class PortfolioAnalytics:
    """Compute advanced portfolio analytics and performance metrics.

    All methods are stateless and operate on provided data series,
    making them suitable for both live monitoring and backtest analysis.
    """

    @staticmethod
    def compute_sortino_ratio(
        returns: Sequence[float], risk_free: float = 0.0
    ) -> float:
        """Compute the Sortino ratio using downside deviation.

        Unlike the Sharpe ratio which penalizes all volatility equally,
        the Sortino ratio only penalizes downside volatility, making it
        more appropriate for strategies with asymmetric return profiles.

        Args:
            returns: Sequence of period returns.
            risk_free: Risk-free rate per period. Defaults to 0.0.

        Returns:
            Sortino ratio, or 0.0 if insufficient data.
        """
        if len(returns) < 2:
            return 0.0

        arr = np.array(returns, dtype=np.float64)
        excess = arr - risk_free
        mean_excess = float(np.mean(excess))

        # Downside deviation: std of returns below the target (risk_free)
        downside = np.minimum(excess, 0.0)
        downside_std = float(np.sqrt(np.mean(downside ** 2)))

        if downside_std < 1e-10:
            return 0.0

        return mean_excess / downside_std

    @staticmethod
    def compute_calmar_ratio(total_return: float, max_drawdown: float) -> float:
        """Compute the Calmar ratio (return / max drawdown).

        Measures how well the strategy compensates for its worst
        peak-to-trough decline.

        Args:
            total_return: Total return as a decimal (e.g. 0.15 for 15%).
            max_drawdown: Maximum drawdown as a positive decimal (e.g. 0.10 for 10%).

        Returns:
            Calmar ratio, or 0.0 if max drawdown is negligible.
        """
        if max_drawdown < 1e-10:
            return 0.0

        return total_return / max_drawdown

    @staticmethod
    def compute_var(
        returns: Sequence[float], confidence: float = 0.95
    ) -> float:
        """Compute Value at Risk using the historical percentile method.

        VaR answers: "What is the worst expected loss at a given
        confidence level?" Returned as a positive number representing
        the loss magnitude.

        Args:
            returns: Sequence of period returns.
            confidence: Confidence level (e.g. 0.95 for 95%). Defaults to 0.95.

        Returns:
            VaR as a positive float (loss magnitude), or 0.0 if insufficient data.
        """
        if len(returns) < 2:
            return 0.0

        arr = np.array(returns, dtype=np.float64)
        percentile = (1 - confidence) * 100
        var = float(np.percentile(arr, percentile))

        # Return as positive loss magnitude
        return -var if var < 0 else 0.0

    @staticmethod
    def compute_cvar(
        returns: Sequence[float], confidence: float = 0.95
    ) -> float:
        """Compute Conditional Value at Risk (Expected Shortfall).

        CVaR is the expected loss given that the loss exceeds VaR.
        It captures tail risk better than VaR alone.

        Args:
            returns: Sequence of period returns.
            confidence: Confidence level (e.g. 0.95 for 95%). Defaults to 0.95.

        Returns:
            CVaR as a positive float (expected tail loss), or 0.0 if insufficient data.
        """
        if len(returns) < 2:
            return 0.0

        arr = np.array(returns, dtype=np.float64)
        percentile = (1 - confidence) * 100
        var_threshold = float(np.percentile(arr, percentile))

        # Average of all returns at or below the VaR threshold
        tail_returns = arr[arr <= var_threshold]
        if len(tail_returns) == 0:
            return 0.0

        cvar = float(np.mean(tail_returns))
        return -cvar if cvar < 0 else 0.0

    @staticmethod
    def compute_max_consecutive_losses(pnl_series: Sequence[float]) -> int:
        """Compute the longest streak of consecutive losses.

        Args:
            pnl_series: Sequence of per-period P&L values.

        Returns:
            Length of the longest losing streak.
        """
        if not pnl_series:
            return 0

        max_streak = 0
        current_streak = 0

        for pnl in pnl_series:
            if pnl < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    @staticmethod
    def compute_max_consecutive_wins(pnl_series: Sequence[float]) -> int:
        """Compute the longest streak of consecutive wins.

        Args:
            pnl_series: Sequence of per-period P&L values.

        Returns:
            Length of the longest winning streak.
        """
        if not pnl_series:
            return 0

        max_streak = 0
        current_streak = 0

        for pnl in pnl_series:
            if pnl > 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    @staticmethod
    def compute_recovery_time(portfolio_values: Sequence[float]) -> int:
        """Compute the number of steps from the deepest drawdown trough to recovery.

        Recovery is defined as the portfolio value returning to the peak
        that preceded the maximum drawdown. If the portfolio never recovers,
        returns the number of steps from the trough to the end of the series.

        Args:
            portfolio_values: Sequence of portfolio values over time.

        Returns:
            Number of steps from trough to recovery (or end of series).
        """
        if len(portfolio_values) < 2:
            return 0

        values = np.array(portfolio_values, dtype=np.float64)

        # Find the maximum drawdown trough
        peak = values[0]
        max_dd = 0.0
        trough_idx = 0
        peak_before_trough = peak

        for i in range(1, len(values)):
            if values[i] > peak:
                peak = values[i]
            dd = (peak - values[i]) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
                trough_idx = i
                peak_before_trough = peak

        if max_dd < 1e-10:
            return 0

        # Count steps from trough until portfolio recovers to the pre-drawdown peak
        for i in range(trough_idx + 1, len(values)):
            if values[i] >= peak_before_trough:
                return i - trough_idx

        # Never fully recovered
        return len(values) - 1 - trough_idx

    @staticmethod
    def compute_strategy_attribution(
        trades: Sequence[TradeResult],
    ) -> dict[str, StrategyStats]:
        """Compute P&L attribution broken down by strategy name.

        Groups trades by their strategy label and computes per-strategy
        statistics including trade count, total P&L, win rate, average
        P&L, and Sharpe ratio.

        Args:
            trades: Sequence of executed TradeResult objects.

        Returns:
            Dictionary mapping strategy name to StrategyStats.
        """
        if not trades:
            return {}

        # Group trade P&L by strategy
        strategy_pnls: dict[str, list[float]] = defaultdict(list)

        for trade in trades:
            if not trade.success:
                continue

            strategy = trade.order.strategy or "unknown"

            # Approximate per-trade P&L from fill data
            if trade.order.side == Side.BUY:
                # For buys, P&L is unrealized; use negative cost as placeholder
                pnl = -trade.net_cost
            else:
                # For sells, P&L is the revenue minus fees
                pnl = trade.fill_price * trade.fill_size - trade.fees

            strategy_pnls[strategy].append(pnl)

        result: dict[str, StrategyStats] = {}
        for strategy, pnls in strategy_pnls.items():
            pnl_arr = np.array(pnls, dtype=np.float64)
            num = len(pnls)
            total = float(np.sum(pnl_arr))
            wins = int(np.sum(pnl_arr > 0))
            win_rate = wins / num if num > 0 else 0.0
            avg = float(np.mean(pnl_arr)) if num > 0 else 0.0
            sharpe = calculate_sharpe_ratio(pnls) if num >= 2 else 0.0

            result[strategy] = StrategyStats(
                strategy=strategy,
                num_trades=num,
                total_pnl=total,
                win_rate=win_rate,
                avg_pnl=avg,
                sharpe=sharpe,
            )

        return result

    @staticmethod
    def compute_hourly_performance(
        trades: Sequence[TradeResult],
    ) -> dict[int, float]:
        """Compute aggregate P&L by hour of day.

        Useful for identifying time-of-day patterns in profitability.

        Args:
            trades: Sequence of executed TradeResult objects.

        Returns:
            Dictionary mapping hour (0-23) to total P&L for that hour.
        """
        hourly_pnl: dict[int, float] = defaultdict(float)

        for trade in trades:
            if not trade.success:
                continue

            hour = trade.timestamp.hour

            if trade.order.side == Side.BUY:
                pnl = -trade.net_cost
            else:
                pnl = trade.fill_price * trade.fill_size - trade.fees

            hourly_pnl[hour] += pnl

        return dict(hourly_pnl)

    def full_report(
        self,
        portfolio_values: Sequence[float],
        trades: Sequence[TradeResult],
        returns: Sequence[float],
    ) -> AnalyticsReport:
        """Generate a comprehensive analytics report.

        Combines all individual metrics into a single AnalyticsReport
        dataclass for easy consumption.

        Args:
            portfolio_values: Time series of portfolio values.
            trades: List of all executed trades.
            returns: Time series of period returns.

        Returns:
            AnalyticsReport with all metrics populated.
        """
        returns_list = list(returns)

        # Compute step-level P&L for streak analysis
        step_pnls = []
        for i in range(1, len(portfolio_values)):
            change = portfolio_values[i] - portfolio_values[i - 1]
            if abs(change) > 1e-6:
                step_pnls.append(change)

        # Max drawdown from portfolio values
        peak = portfolio_values[0] if portfolio_values else 0.0
        max_dd = 0.0
        for val in portfolio_values:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        # Total return
        if len(portfolio_values) >= 2 and portfolio_values[0] > 0:
            total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        else:
            total_return = 0.0

        # Win rate from step P&Ls
        if step_pnls:
            win_rate = sum(1 for p in step_pnls if p > 0) / len(step_pnls)
        else:
            win_rate = 0.0

        return AnalyticsReport(
            total_return_pct=total_return,
            sharpe_ratio=calculate_sharpe_ratio(returns_list) if len(returns_list) >= 2 else 0.0,
            sortino_ratio=self.compute_sortino_ratio(returns_list),
            calmar_ratio=self.compute_calmar_ratio(total_return, max_dd),
            max_drawdown_pct=max_dd,
            var_95=self.compute_var(returns_list, confidence=0.95),
            cvar_95=self.compute_cvar(returns_list, confidence=0.95),
            max_consecutive_wins=self.compute_max_consecutive_wins(step_pnls),
            max_consecutive_losses=self.compute_max_consecutive_losses(step_pnls),
            recovery_time_steps=self.compute_recovery_time(portfolio_values),
            strategy_attribution=self.compute_strategy_attribution(trades),
            hourly_performance=self.compute_hourly_performance(trades),
            num_trades=len([t for t in trades if t.success]),
            win_rate=win_rate,
        )
