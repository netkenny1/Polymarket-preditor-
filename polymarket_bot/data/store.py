"""Data persistence and caching for the trading bot."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from polymarket_bot.data.models import PortfolioSnapshot, TradeResult

logger = structlog.get_logger()


class DataStore:
    """Simple JSON-file-based data persistence.

    Stores trade history, portfolio snapshots, and market data cache
    for backtesting and analysis.
    """

    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save_trade_history(self, trades: list[TradeResult], filename: str = "trades.json") -> None:
        path = self.data_dir / filename
        records = []
        for t in trades:
            records.append({
                "order_id": t.order.order_id,
                "market": t.order.market_condition_id,
                "token_id": t.order.token_id,
                "side": t.order.side.value,
                "price": t.fill_price,
                "size": t.fill_size,
                "fees": t.fees,
                "strategy": t.order.strategy,
                "success": t.success,
                "timestamp": t.timestamp.isoformat(),
            })
        path.write_text(json.dumps(records, indent=2))
        logger.info("saved_trades", count=len(records), path=str(path))

    def save_snapshots(self, snapshots: list[PortfolioSnapshot], filename: str = "snapshots.json") -> None:
        path = self.data_dir / filename
        records = []
        for s in snapshots:
            records.append({
                "timestamp": s.timestamp.isoformat(),
                "total_value": s.total_value,
                "cash": s.cash,
                "positions_value": s.positions_value,
                "unrealized_pnl": s.unrealized_pnl,
                "realized_pnl": s.realized_pnl,
                "num_positions": s.num_positions,
                "drawdown_pct": s.drawdown_pct,
            })
        path.write_text(json.dumps(records, indent=2))

    def load_json(self, filename: str) -> Any:
        path = self.data_dir / filename
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def save_json(self, data: Any, filename: str) -> None:
        path = self.data_dir / filename
        path.write_text(json.dumps(data, indent=2, default=str))
