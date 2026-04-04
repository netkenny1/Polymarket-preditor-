"""Data models for the Polymarket trading bot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class Outcome(str, Enum):
    YES = "Yes"
    NO = "No"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class MarketCategory(str, Enum):
    CRYPTO = "crypto"
    POLITICS = "politics"
    SPORTS = "sports"
    POP_CULTURE = "pop_culture"
    SCIENCE = "science"
    OTHER = "other"


@dataclass
class Token:
    """A tradeable outcome token on Polymarket."""

    token_id: str
    outcome: str  # "Yes" or "No"
    price: float  # 0.0 to 1.0
    winner: Optional[bool] = None


@dataclass
class Market:
    """A Polymarket prediction market."""

    condition_id: str
    question: str
    slug: str
    tokens: list[Token] = field(default_factory=list)
    category: MarketCategory = MarketCategory.OTHER
    end_date: Optional[datetime] = None
    volume_24h: float = 0.0
    liquidity: float = 0.0
    active: bool = True
    description: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def yes_price(self) -> float:
        for t in self.tokens:
            if t.outcome == "Yes":
                return t.price
        return 0.5

    @property
    def no_price(self) -> float:
        for t in self.tokens:
            if t.outcome == "No":
                return t.price
        return 0.5

    @property
    def spread(self) -> float:
        return abs(1.0 - self.yes_price - self.no_price)

    @property
    def implied_probability(self) -> float:
        return self.yes_price


@dataclass
class OrderBookLevel:
    """A single price level in the order book."""

    price: float
    size: float


@dataclass
class OrderBook:
    """Full order book for a market outcome."""

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
    """A trade order."""

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
    """A held position in a market."""

    market_condition_id: str
    token_id: str
    outcome: str
    size: float
    avg_entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    max_price_seen: float = 0.0  # For trailing stop
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
    """A trading signal from a strategy."""

    market_condition_id: str
    token_id: str
    side: Side
    outcome: str
    estimated_fair_value: float  # Our estimate of true probability
    market_price: float  # Current market price
    edge: float  # fair_value - market_price (for BUY) or inverse
    confidence: float  # 0 to 1
    strategy: str
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def expected_value(self) -> float:
        """Expected profit per dollar risked."""
        if self.side == Side.BUY:
            # Buy at market_price, expect to win fair_value
            return (self.estimated_fair_value / self.market_price) - 1.0
        else:
            # Sell at market_price, expect it's worth fair_value
            return (self.market_price / self.estimated_fair_value) - 1.0


@dataclass
class TradeResult:
    """Result of an executed trade."""

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
    """Point-in-time snapshot of the portfolio."""

    timestamp: datetime
    total_value: float
    cash: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    num_positions: int
    drawdown_pct: float = 0.0


@dataclass
class SentimentData:
    """Aggregated sentiment for a market/topic."""

    query: str
    tweet_count: int = 0
    avg_sentiment: float = 0.0  # -1 to 1
    sentiment_std: float = 0.0
    volume_ratio: float = 1.0  # Current vs baseline volume
    bullish_pct: float = 0.5
    bearish_pct: float = 0.5
    sample_tweets: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class NarrativeCategory(str, Enum):
    """Categories for narrative themes."""
    TRADE_WAR = "trade_war"
    MONETARY_POLICY = "monetary_policy"
    GEOPOLITICAL = "geopolitical"
    CRYPTO_REGULATION = "crypto_regulation"
    FISCAL_POLICY = "fiscal_policy"
    ELECTION = "election"
    MARKET_CRISIS = "market_crisis"
    OTHER = "other"


@dataclass
class EconomicIndicator:
    """A single economic data point."""
    name: str
    value: float
    previous_value: float
    change_pct: float
    unit: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class NarrativeEvent:
    """A discrete event that feeds into a narrative."""
    event_id: str
    source: str  # "trump_tweet", "economic_data", "news", "market_move"
    content: str
    timestamp: datetime
    category: NarrativeCategory
    sentiment: float  # -1 to 1
    magnitude: float  # 0 to 1
    keywords: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class HistoricalPhase:
    """One phase within a historical pattern."""
    phase_name: str
    description: str
    duration_days: int
    market_impact: dict = field(default_factory=dict)  # {category: direction}
    keywords: list[str] = field(default_factory=list)
    sequence_index: int = 0


@dataclass
class HistoricalPattern:
    """A codified historical precedent for predictive history."""
    pattern_id: str
    name: str
    category: NarrativeCategory
    description: str
    trigger_keywords: list[str] = field(default_factory=list)
    timeline_days: int = 0
    market_impact: dict = field(default_factory=dict)  # {category: direction}
    outcome_direction: float = 0.0  # -1 to 1
    outcome_magnitude: float = 0.0  # 0 to 1
    phases: list[HistoricalPhase] = field(default_factory=list)
    similarity_threshold: float = 0.4
    source_period: str = ""


@dataclass
class Narrative:
    """A coherent story built from multiple events with predictive power."""
    narrative_id: str
    title: str
    category: NarrativeCategory
    thesis: str
    events: list[NarrativeEvent] = field(default_factory=list)
    affected_market_ids: list[str] = field(default_factory=list)
    predicted_direction: float = 0.0  # -1 to 1
    confidence: float = 0.0
    strength: float = 0.0  # 0 to 1
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    historical_pattern_id: Optional[str] = None
    is_active: bool = True

    @property
    def age_hours(self) -> float:
        delta = datetime.now(timezone.utc) - self.created_at
        return delta.total_seconds() / 3600.0

    @property
    def event_count(self) -> int:
        return len(self.events)


@dataclass
class SimulationScenario:
    """One parallel scenario in the council-of-agents system."""
    scenario_id: str
    name: str
    narrative_id: str
    assumptions: dict = field(default_factory=dict)
    predicted_direction: float = 0.0
    predicted_magnitude: float = 0.0
    weight: float = 1.0
    accuracy_history: list[float] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def avg_accuracy(self) -> float:
        if not self.accuracy_history:
            return 0.5
        return sum(self.accuracy_history) / len(self.accuracy_history)
