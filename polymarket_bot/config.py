"""Configuration for the BTC 5-minute Polymarket bot.

Lean config: one strategy, one market, one price feed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class PolymarketConfig:
    """Polymarket CLOB API credentials and endpoints."""

    api_key: str = ""
    api_secret: str = ""
    api_passphrase: str = ""
    private_key: str = ""
    funder_address: str = ""  # Polymarket proxy / Safe address
    base_url: str = "https://clob.polymarket.com"
    gamma_url: str = "https://gamma-api.polymarket.com"
    ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    data_api_url: str = "https://data-api.polymarket.com"
    chain_id: int = 137  # Polygon


@dataclass(frozen=True)
class BinanceConfig:
    """Binance BTCUSDT real-time + historical config."""

    ws_url: str = "wss://stream.binance.com:9443/ws"
    # bookTicker gives stable best-bid/ask mid; aggTrade is noisier.
    ws_stream: str = "btcusdt@bookTicker"
    rest_url: str = "https://api.binance.com"
    archive_url: str = "https://data.binance.vision"
    symbol: str = "BTCUSDT"


@dataclass(frozen=True)
class StrategyConfig:
    """BTC 5-minute up/down strategy parameters."""

    # Market slug pattern on Polymarket. `{ts}` is the window-start unix seconds.
    slug_template: str = "btc-updown-5m-{ts}"
    window_seconds: int = 300

    # Volatility estimation
    vol_window_minutes: int = 60  # EWMA lookback on 1-min BTC returns
    vol_ewma_lambda: float = 0.94  # RiskMetrics default
    min_annual_vol: float = 0.20  # floor
    max_annual_vol: float = 2.50  # ceiling (safety against outliers)

    # Edge requirements
    min_edge: float = 0.03  # 3 cents fair-vs-market gap before trading
    min_seconds_to_expiry: int = 20  # don't open new positions in final 20s
    max_seconds_to_expiry: int = 270  # avoid the first 30s (no informative move yet)

    # Fees: Polymarket dynamic crypto taker fee ~ 0.018 * 4p(1-p); maker = 0.
    maker_fee: float = 0.0
    taker_fee_peak: float = 0.018  # at p=0.5
    # We refuse to cross the spread unless edge > taker_fee + buffer.
    prefer_maker: bool = True
    post_only: bool = True

    # Position sizing
    kelly_fraction: float = 0.25  # quarter Kelly
    max_position_usd: float = 25.0
    min_position_usd: float = 2.0  # below this, skip — too small to clear fees
    confidence_shrinkage: float = 1.0  # blend fair toward market when uncertain

    # Model-error / sanity clamps
    prob_clamp_min: float = 0.01
    prob_clamp_max: float = 0.99


@dataclass(frozen=True)
class RiskConfig:
    """Risk limits and circuit breakers."""

    max_portfolio_exposure_usd: float = 100.0
    max_single_position_usd: float = 25.0
    max_positions: int = 5
    max_daily_loss_usd: float = 30.0
    max_drawdown_pct: float = 0.25
    position_limit_per_market_pct: float = 0.35
    stop_loss_pct: float = 0.50  # 5-min markets rarely deserve stops, but belt+braces
    trailing_stop_pct: float = 0.40
    min_liquidity_usd: float = 100.0


@dataclass(frozen=True)
class TradingConfig:
    """Core trading toggles."""

    paper_trading: bool = True
    initial_capital_usd: float = 100.0
    log_level: str = "INFO"
    # Legacy fields kept for compatibility with risk manager + position sizer
    kelly_fraction: float = 0.25
    min_edge_threshold: float = 0.03
    max_portfolio_exposure_usd: float = 100.0
    max_single_position_usd: float = 25.0
    max_positions: int = 5


@dataclass
class BotConfig:
    """Master configuration for the BTC 5-min bot."""

    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)

    @classmethod
    def from_env(cls) -> BotConfig:
        """Load configuration from environment variables."""
        paper = os.getenv("PAPER_TRADING", "true").lower() == "true"
        capital = float(os.getenv("INITIAL_CAPITAL_USD", "100"))
        kelly = float(os.getenv("KELLY_FRACTION", "0.25"))
        min_edge = float(os.getenv("MIN_EDGE", "0.03"))
        max_pos = float(os.getenv("MAX_POSITION_USD", "25"))
        max_exp = float(os.getenv("MAX_PORTFOLIO_EXPOSURE_USD", "100"))

        return cls(
            polymarket=PolymarketConfig(
                api_key=os.getenv("POLYMARKET_API_KEY", ""),
                api_secret=os.getenv("POLYMARKET_API_SECRET", ""),
                api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE", ""),
                private_key=os.getenv("POLYMARKET_PRIVATE_KEY", ""),
                funder_address=os.getenv("POLYMARKET_FUNDER", ""),
            ),
            strategy=StrategyConfig(
                kelly_fraction=kelly,
                min_edge=min_edge,
                max_position_usd=max_pos,
            ),
            risk=RiskConfig(
                max_portfolio_exposure_usd=max_exp,
                max_single_position_usd=max_pos,
            ),
            trading=TradingConfig(
                paper_trading=paper,
                initial_capital_usd=capital,
                kelly_fraction=kelly,
                min_edge_threshold=min_edge,
                max_portfolio_exposure_usd=max_exp,
                max_single_position_usd=max_pos,
                log_level=os.getenv("LOG_LEVEL", "INFO"),
            ),
        )
