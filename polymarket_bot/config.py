"""Configuration management for the Polymarket trading bot."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class PolymarketConfig:
    """Polymarket CLOB API configuration."""

    api_key: str = ""
    api_secret: str = ""
    api_passphrase: str = ""
    private_key: str = ""
    base_url: str = "https://clob.polymarket.com"
    gamma_url: str = "https://gamma-api.polymarket.com"
    chain_id: int = 137  # Polygon mainnet


@dataclass(frozen=True)
class TwitterConfig:
    """X/Twitter API configuration."""

    bearer_token: str = ""
    max_tweets_per_query: int = 100
    sentiment_lookback_minutes: int = 60


@dataclass(frozen=True)
class TradingConfig:
    """Core trading parameters."""

    paper_trading: bool = True
    max_portfolio_exposure_usd: float = 50.0
    max_single_position_usd: float = 15.0
    min_edge_threshold: float = 0.02  # 2% minimum edge to trade
    kelly_fraction: float = 0.25  # Quarter-Kelly for $100 capital
    max_positions: int = 20
    min_liquidity_usd: float = 500.0  # Skip illiquid markets
    max_spread: float = 0.10  # Skip markets with >10% spread
    rebalance_interval_seconds: int = 300  # 5 minutes
    stale_price_seconds: int = 120  # Consider price stale after 2 min


@dataclass(frozen=True)
class MarketMakerConfig:
    """Market making strategy parameters."""

    spread: float = 0.04  # 4 cent spread (2 cents each side)
    order_size_usd: float = 25.0
    max_inventory: float = 200.0  # Max position in a single market
    inventory_skew_factor: float = 0.5  # Skew quotes based on inventory
    refresh_interval_seconds: int = 30
    min_book_depth_usd: float = 100.0  # Min depth to market make


@dataclass(frozen=True)
class SentimentConfig:
    """Sentiment strategy parameters."""

    volume_spike_threshold: float = 3.0  # 3x normal volume = spike
    sentiment_threshold: float = 0.3  # Strong sentiment signal
    min_tweets: int = 10  # Need at least 10 tweets for signal
    decay_half_life_minutes: float = 30.0  # Signal decays over time
    keywords_per_market: int = 5


@dataclass(frozen=True)
class ArbitrageConfig:
    """Arbitrage strategy parameters."""

    min_arb_edge: float = 0.02  # 2% minimum arb profit
    complement_tolerance: float = 0.03  # Complement sum tolerance
    max_execution_delay_seconds: float = 5.0


@dataclass(frozen=True)
class RiskConfig:
    """Risk management parameters."""

    max_drawdown_pct: float = 0.20  # 20% max drawdown, halt trading
    max_daily_loss_usd: float = 30.0
    max_correlated_exposure_pct: float = 0.40  # 40% in correlated markets
    position_limit_per_market_pct: float = 0.08  # 8% of portfolio per market
    stop_loss_pct: float = 0.25  # 25% stop loss per position
    trailing_stop_pct: float = 0.20  # 20% trailing stop


@dataclass(frozen=True)
class NarrativeConfig:
    """Narrative analysis strategy parameters."""
    max_active_narratives: int = 10
    max_scenarios_per_narrative: int = 5
    narrative_decay_hours: float = 48.0
    min_events_for_narrative: int = 3
    pattern_similarity_threshold: float = 0.4
    scenario_evaluation_interval_steps: int = 5
    min_narrative_confidence: float = 0.3
    max_signal_confidence: float = 0.75
    council_min_agreement: float = 0.6
    council_size: int = 5


@dataclass
class BotConfig:
    """Master configuration aggregating all sub-configs."""

    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    twitter: TwitterConfig = field(default_factory=TwitterConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    market_maker: MarketMakerConfig = field(default_factory=MarketMakerConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    arbitrage: ArbitrageConfig = field(default_factory=ArbitrageConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    narrative: NarrativeConfig = field(default_factory=NarrativeConfig)

    @classmethod
    def from_env(cls) -> BotConfig:
        """Load configuration from environment variables."""
        return cls(
            polymarket=PolymarketConfig(
                api_key=os.getenv("POLYMARKET_API_KEY", ""),
                api_secret=os.getenv("POLYMARKET_API_SECRET", ""),
                api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE", ""),
                private_key=os.getenv("POLYMARKET_PRIVATE_KEY", ""),
            ),
            twitter=TwitterConfig(
                bearer_token=os.getenv("TWITTER_BEARER_TOKEN", ""),
            ),
            trading=TradingConfig(
                paper_trading=os.getenv("PAPER_TRADING", "true").lower() == "true",
                max_portfolio_exposure_usd=float(
                    os.getenv("MAX_PORTFOLIO_EXPOSURE_USD", "1000")
                ),
                max_single_position_usd=float(
                    os.getenv("MAX_SINGLE_POSITION_USD", "100")
                ),
                min_edge_threshold=float(os.getenv("MIN_EDGE_THRESHOLD", "0.05")),
                kelly_fraction=float(os.getenv("KELLY_FRACTION", "0.25")),
            ),
        )
