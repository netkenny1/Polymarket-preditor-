"""Data models for the BTC 5-minute trading bot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class Outcome(str, Enum):
    UP = "Up"
    DOWN = "Down"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass
class Token:
    """A tradeable outcome token (Up or Down) on Polymarket."""

    token_id: str
    outcome: str  # "Up" / "Down"
    price: float  # 0.0 to 1.0


@dataclass
class Market:
    """A Polymarket BTC 5-minute up/down market."""

    condition_id: str
    slug: str
    question: str = ""
    tokens: list[Token] = field(default_factory=list)
    # Window bounds in unix seconds. `start_ts` is also the strike (BTC at t=start).
    start_ts: int = 0
    end_ts: int = 0
    strike_price: float = 0.0  # BTC price at window start (resolution reference)
    active: bool = True
    liquidity: float = 0.0
    volume_24h: float = 0.0

    @property
    def up_token(self) -> Optional[Token]:
        return next((t for t in self.tokens if t.outcome.lower() == "up" or t.outcome.lower() == "yes"), None)

    @property
    def down_token(self) -> Optional[Token]:
        return next((t for t in self.tokens if t.outcome.lower() == "down" or t.outcome.lower() == "no"), None)

    @property
    def up_price(self) -> float:
        t = self.up_token
        return t.price if t else 0.5

    @property
    def down_price(self) -> float:
        t = self.down_token
        return t.price if t else 0.5

    @property
    def spread(self) -> float:
        return abs(1.0 - self.up_price - self.down_price)

    def seconds_remaining(self, now_ts: int) -> int:
        return max(0, self.end_ts - now_ts)


@dataclass
class OrderBookLevel:
    price: float
    size: float


@dataclass
class OrderBook:
    """Full order book for a market outcome token."""

    token_id: str
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def bid_depth(self) -> float:
        return sum(level.size for level in self.bids)

    @property
    def ask_depth(self) -> float:
        return sum(level.size for level in self.asks)


@dataclass
class Order:
    order_id: str
    market_condition_id: str
    token_id: str
    side: Side
    price: float
    size: float
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    strategy: str = ""


@dataclass
class Position:
    market_condition_id: str
    token_id: str
    outcome: str
    size: float
    avg_entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    max_price_seen: float = 0.0
    strategy: str = ""

    @property
    def market_value(self) -> float:
        return self.size * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.size * self.avg_entry_price

    def update_price(self, price: float) -> None:
        self.current_price = price
        self.unrealized_pnl = self.size * (price - self.avg_entry_price)
        if price > self.max_price_seen:
            self.max_price_seen = price


@dataclass
class Signal:
    """A trading signal emitted by the strategy."""

    market_condition_id: str
    token_id: str
    side: Side
    outcome: str
    estimated_fair_value: float  # model's fair probability
    market_price: float  # current market price (YES ask / NO ask)
    edge: float  # |fair - market|
    confidence: float  # 0 to 1
    strategy: str
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def expected_value(self) -> float:
        if self.side == Side.BUY:
            return (self.estimated_fair_value / self.market_price) - 1.0 if self.market_price > 0 else 0.0
        return (self.market_price / self.estimated_fair_value) - 1.0 if self.estimated_fair_value > 0 else 0.0


@dataclass
class TradeResult:
    order: Order
    success: bool
    fill_price: float = 0.0
    fill_size: float = 0.0
    fees: float = 0.0
    error: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def net_cost(self) -> float:
        return self.fill_price * self.fill_size + self.fees


@dataclass
class PortfolioSnapshot:
    timestamp: datetime
    total_value: float
    cash: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    num_positions: int
    drawdown_pct: float = 0.0
