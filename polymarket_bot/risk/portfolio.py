"""Portfolio tracking and P&L calculation."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Optional

import structlog

from polymarket_bot.data.models import (
    Order,
    Position,
    PortfolioSnapshot,
    Side,
    TradeResult,
)

logger = structlog.get_logger()


class Portfolio:
    """Track positions, cash, and P&L.

    Maintains a real-time view of all positions, handles trade fills,
    and computes portfolio-level metrics.
    """

    def __init__(self, initial_cash: float = 1000.0) -> None:
        self.cash: float = initial_cash
        self.initial_cash: float = initial_cash
        self.positions: dict[str, Position] = {}  # keyed by token_id
        self.trade_history: list[TradeResult] = []
        self.snapshots: list[PortfolioSnapshot] = []
        self.peak_value: float = initial_cash
        self.realized_pnl: float = 0.0

    @property
    def total_value(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions.values())

    @property
    def total_exposure(self) -> float:
        return sum(abs(p.market_value) for p in self.positions.values())

    @property
    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl

    @property
    def return_pct(self) -> float:
        if self.initial_cash == 0:
            return 0.0
        return (self.total_value - self.initial_cash) / self.initial_cash

    @property
    def drawdown_pct(self) -> float:
        if self.peak_value == 0:
            return 0.0
        return (self.peak_value - self.total_value) / self.peak_value

    def process_fill(self, result: TradeResult) -> None:
        """Process a trade fill and update positions."""
        if not result.success:
            return

        order = result.order
        token_id = order.token_id

        if order.side == Side.BUY:
            self._process_buy(token_id, result)
        else:
            self._process_sell(token_id, result)

        self.trade_history.append(result)

    def _process_buy(self, token_id: str, result: TradeResult) -> None:
        """Process a buy fill."""
        cost = result.fill_price * result.fill_size + result.fees
        self.cash -= cost

        if token_id in self.positions:
            pos = self.positions[token_id]
            total_size = pos.size + result.fill_size
            if total_size > 0:
                pos.avg_entry_price = (
                    pos.avg_entry_price * pos.size + result.fill_price * result.fill_size
                ) / total_size
            pos.size = total_size
        else:
            self.positions[token_id] = Position(
                market_condition_id=result.order.market_condition_id,
                token_id=token_id,
                outcome=result.order.strategy,
                size=result.fill_size,
                avg_entry_price=result.fill_price,
                current_price=result.fill_price,
                strategy=result.order.strategy,
            )

    def _process_sell(self, token_id: str, result: TradeResult) -> None:
        """Process a sell fill."""
        revenue = result.fill_price * result.fill_size - result.fees
        self.cash += revenue

        if token_id in self.positions:
            pos = self.positions[token_id]
            pnl = (result.fill_price - pos.avg_entry_price) * result.fill_size - result.fees
            self.realized_pnl += pnl
            pos.realized_pnl += pnl
            pos.size -= result.fill_size

            # Remove position if fully closed
            if pos.size <= 0.001:
                del self.positions[token_id]
        else:
            # Short sale (or selling tokens we received)
            self.positions[token_id] = Position(
                market_condition_id=result.order.market_condition_id,
                token_id=token_id,
                outcome=result.order.strategy,
                size=-result.fill_size,
                avg_entry_price=result.fill_price,
                current_price=result.fill_price,
                strategy=result.order.strategy,
            )

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update position prices and portfolio metrics."""
        for token_id, price in prices.items():
            if token_id in self.positions:
                self.positions[token_id].update_price(price)

        # Update peak for drawdown tracking
        current = self.total_value
        if current > self.peak_value:
            self.peak_value = current

    def take_snapshot(self) -> PortfolioSnapshot:
        """Take a point-in-time snapshot of the portfolio."""
        snap = PortfolioSnapshot(
            timestamp=datetime.utcnow(),
            total_value=self.total_value,
            cash=self.cash,
            positions_value=self.total_value - self.cash,
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl=self.realized_pnl,
            num_positions=len(self.positions),
            drawdown_pct=self.drawdown_pct,
        )
        self.snapshots.append(snap)
        return snap

    def get_positions_by_market(self) -> dict[str, list[Position]]:
        """Group positions by market condition ID."""
        groups: dict[str, list[Position]] = defaultdict(list)
        for pos in self.positions.values():
            groups[pos.market_condition_id].append(pos)
        return groups

    def get_exposure_by_strategy(self) -> dict[str, float]:
        """Calculate total exposure per strategy."""
        exposure: dict[str, float] = defaultdict(float)
        for pos in self.positions.values():
            exposure[pos.strategy] += abs(pos.market_value)
        return dict(exposure)

    def save_state(self, filepath: str = "portfolio_state.json") -> None:
        """Save portfolio state to disk for crash recovery."""
        import json
        state = {
            "cash": self.cash,
            "initial_cash": self.initial_cash,
            "peak_value": self.peak_value,
            "realized_pnl": self.realized_pnl,
            "positions": {
                tid: {
                    "market_condition_id": pos.market_condition_id,
                    "token_id": pos.token_id,
                    "outcome": pos.outcome,
                    "size": pos.size,
                    "avg_entry_price": pos.avg_entry_price,
                    "current_price": pos.current_price,
                    "strategy": pos.strategy,
                    "realized_pnl": pos.realized_pnl,
                }
                for tid, pos in self.positions.items()
            },
            "saved_at": datetime.utcnow().isoformat(),
        }
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load_state(cls, filepath: str = "portfolio_state.json") -> "Portfolio":
        """Load portfolio state from disk."""
        import json
        with open(filepath) as f:
            state = json.load(f)

        portfolio = cls(initial_cash=state["initial_cash"])
        portfolio.cash = state["cash"]
        portfolio.peak_value = state["peak_value"]
        portfolio.realized_pnl = state["realized_pnl"]

        for tid, pos_data in state.get("positions", {}).items():
            portfolio.positions[tid] = Position(
                market_condition_id=pos_data["market_condition_id"],
                token_id=pos_data["token_id"],
                outcome=pos_data["outcome"],
                size=pos_data["size"],
                avg_entry_price=pos_data["avg_entry_price"],
                current_price=pos_data["current_price"],
                strategy=pos_data.get("strategy", ""),
                realized_pnl=pos_data.get("realized_pnl", 0.0),
            )

        return portfolio
