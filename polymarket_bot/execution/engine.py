"""Order execution engine.

Handles the lifecycle of converting signals into executed trades:
1. Signal -> Risk check -> Size calculation
2. Price optimization (limit vs market)
3. Order submission
4. Fill tracking and position update
5. Post-trade logging

Uses limit orders by default (better fills on Polymarket's CLOB).
"""

from __future__ import annotations

from typing import Any

import structlog

from polymarket_bot.clients.polymarket import PolymarketClient
from polymarket_bot.data.models import Signal, Side, TradeResult
from polymarket_bot.risk.manager import RiskManager
from polymarket_bot.risk.portfolio import Portfolio
from polymarket_bot.risk.position_sizer import PositionSizer
from polymarket_bot.utils.helpers import round_price

logger = structlog.get_logger()


class ExecutionEngine:
    """Execute trades from signals through the full pipeline.

    Pipeline: Signal -> Risk Check -> Size -> Price -> Order -> Fill -> Update
    """

    def __init__(
        self,
        client: PolymarketClient,
        risk_manager: RiskManager,
        portfolio: Portfolio,
    ) -> None:
        self.client = client
        self.risk = risk_manager
        self.portfolio = portfolio
        self._pending_orders: dict[str, Any] = {}

    def execute_signals(self, signals: list[Signal]) -> list[TradeResult]:
        """Execute a batch of signals through the full pipeline.

        Processes signals in priority order (highest edge * confidence first).
        Stops if risk limits are hit.
        """
        results: list[TradeResult] = []

        # Sort by expected value
        sorted_signals = sorted(
            signals,
            key=lambda s: s.edge * s.confidence,
            reverse=True,
        )

        for signal in sorted_signals:
            result = self.execute_signal(signal)
            if result is not None:
                results.append(result)

        return results

    def execute_signal(self, signal: Signal) -> TradeResult | None:
        """Execute a single signal."""
        # 1. Risk check
        approved, size_usd, reason = self.risk.check_signal(signal)
        if not approved:
            logger.info(
                "signal_rejected",
                market=signal.market_condition_id,
                reason=reason,
            )
            return None

        # 2. Calculate shares from dollar amount
        price = self._optimize_price(signal)
        shares = round(size_usd / price, 2) if price > 0 else 0

        if shares <= 0:
            return None

        # 3. Place order
        logger.info(
            "executing_trade",
            market=signal.market_condition_id,
            side=signal.side.value,
            outcome=signal.outcome,
            price=price,
            shares=shares,
            size_usd=round(size_usd, 2),
            edge=round(signal.edge, 4),
            strategy=signal.strategy,
        )

        result = self.client.place_order(
            token_id=signal.token_id,
            side=signal.side,
            price=price,
            size=shares,
            market_condition_id=signal.market_condition_id,
            strategy=signal.strategy,
        )

        # 4. Update portfolio on fill
        if result.success:
            self.portfolio.process_fill(result)
            # Only count realized P&L toward daily loss (not buy costs which are investments)
            if signal.side == Side.SELL:
                # Estimate realized P&L from the sell
                pos = self.portfolio.positions.get(signal.token_id)
                if pos:
                    realized = (result.fill_price - pos.avg_entry_price) * result.fill_size
                    self.risk.update_daily_pnl(realized)
            logger.info(
                "trade_executed",
                order_id=result.order.order_id,
                fill_price=result.fill_price,
                fill_size=result.fill_size,
                net_cost=round(result.net_cost, 2),
            )
        else:
            logger.error(
                "trade_failed",
                market=signal.market_condition_id,
                error=result.error,
            )

        return result

    def execute_stop_losses(self, stops: list[dict[str, Any]]) -> list[TradeResult]:
        """Execute stop loss orders for triggered positions."""
        results = []

        for stop in stops:
            pos = stop["position"]
            logger.warning(
                "executing_stop_loss",
                token_id=stop["token_id"],
                reason=stop["reason"],
                position_size=pos.size,
            )

            # Sell the entire position at market
            result = self.client.place_order(
                token_id=stop["token_id"],
                side=Side.SELL,
                price=round_price(pos.current_price * 0.98),  # Slight discount for fill
                size=pos.size,
                market_condition_id=pos.market_condition_id,
                strategy="stop_loss",
            )

            if result.success:
                self.portfolio.process_fill(result)
                pnl = (result.fill_price - pos.avg_entry_price) * result.fill_size
                self.risk.update_daily_pnl(pnl)
                logger.info("stop_loss_filled", pnl=round(pnl, 2))

            results.append(result)

        return results

    def _optimize_price(self, signal: Signal) -> float:
        """Determine optimal limit order price.

        Places limit orders slightly better than the signal's market price
        to improve fill quality while ensuring execution.
        """
        if signal.side == Side.BUY:
            # Place bid slightly below market for better fill
            # But not too far to ensure execution
            price = signal.market_price - 0.005
        else:
            # Place ask slightly above market
            price = signal.market_price + 0.005

        price = round_price(price)
        # Ensure valid price range
        return max(0.01, min(0.99, price))

    def cancel_all(self) -> bool:
        """Cancel all open orders - emergency shutdown."""
        logger.warning("cancelling_all_orders")
        return self.client.cancel_all_orders()
