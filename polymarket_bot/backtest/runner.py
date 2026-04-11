"""Backtest the BTC 5-min strategy against real Binance historical data.

How it works:

  1. Pull N days of 1-minute BTCUSDT klines from Binance (cached to disk).
  2. Partition the series into 5-minute windows aligned to the UTC grid.
  3. For each window, replay each minute as a simulated "current time":
       - strike = close of the minute BEFORE the window started (S0)
       - spot   = close of the current minute (St)
       - seconds_left = (window_end - current_minute_end)
       - sigma  = EWMA of 1-min log returns over trailing `vol_window_minutes`
  4. Model the **market price** that Polymarket would show, using a LAGGED
     Black-Scholes computed from the spot `lag_seconds` ago (simulating
     Polymarket market-maker reaction lag — the documented structural edge).
  5. Run `BTC5MinStrategy.generate_signals`; if a signal fires, simulate
     a maker fill at the market price (maker fee = 0).
  6. When the window closes (T = 0), settle each open position:
       - win  → receive 1 USDC per share
       - lose → receive 0
  7. Track PnL, win rate, drawdown, and per-trade diagnostics.

Everything is deterministic given (start_ts, end_ts, lag_seconds). Pulls
real data so the "test against real BTC data" claim is honest — no
synthetic price paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import structlog

from polymarket_bot.clients.binance_feed import BinanceHistoricalClient, Kline
from polymarket_bot.config import StrategyConfig
from polymarket_bot.data.models import (
    Market,
    OrderBook,
    OrderBookLevel,
    Side,
    Token,
)
from polymarket_bot.pricing.black_scholes import (
    digital_call_prob,
    ewma_vol,
    years_from_seconds,
)
from polymarket_bot.pricing.kelly import kelly_fraction, shrink_toward_market
from polymarket_bot.strategies.btc_5min import BTC5MinContext, BTC5MinStrategy

logger = structlog.get_logger()


@dataclass
class BacktestConfig:
    """Inputs to a single backtest run."""

    days: int = 7
    initial_capital: float = 100.0
    lag_seconds: int = 30  # market-maker lag we assume Polymarket has
    spread_cents: float = 0.02  # bid-ask spread on YES/NO tokens
    min_seconds_to_expiry: int = 20  # don't open trades in final seconds
    max_seconds_to_expiry: int = 240  # skip first 60s of each window
    vol_window_minutes: int = 60
    ewma_lambda: float = 0.94
    min_edge: float = 0.03
    kelly_fraction: float = 0.25
    max_position_usd: float = 25.0
    verbose: bool = False


@dataclass
class BacktestTrade:
    """One executed trade in a backtest run."""

    window_start_ts: int
    window_end_ts: int
    side_outcome: str  # "Up" or "Down"
    strike: float
    entry_spot: float
    entry_seconds_left: int
    entry_price: float
    fair_at_entry: float
    sigma_at_entry: float
    size_usd: float
    shares: float
    resolved: Optional[bool] = None  # True if won
    exit_spot: float = 0.0
    pnl: float = 0.0

    def as_dict(self) -> dict:
        return {
            "window_start": datetime.fromtimestamp(self.window_start_ts, tz=timezone.utc).isoformat(),
            "side": self.side_outcome,
            "strike": round(self.strike, 2),
            "entry_spot": round(self.entry_spot, 2),
            "t_left": self.entry_seconds_left,
            "entry_price": round(self.entry_price, 4),
            "fair": round(self.fair_at_entry, 4),
            "sigma": round(self.sigma_at_entry, 3),
            "size_usd": round(self.size_usd, 2),
            "shares": round(self.shares, 2),
            "won": self.resolved,
            "exit_spot": round(self.exit_spot, 2),
            "pnl": round(self.pnl, 4),
        }


@dataclass
class BacktestResult:
    """Aggregate backtest statistics."""

    config: BacktestConfig
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    initial_capital: float = 0.0
    final_capital: float = 0.0
    total_windows: int = 0
    windows_traded: int = 0

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.resolved)

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades if t.resolved is False)

    @property
    def win_rate(self) -> float:
        n = self.total_trades
        return self.wins / n if n else 0.0

    @property
    def total_pnl(self) -> float:
        return self.final_capital - self.initial_capital

    @property
    def return_pct(self) -> float:
        return self.total_pnl / self.initial_capital if self.initial_capital else 0.0

    @property
    def max_drawdown_pct(self) -> float:
        if not self.equity_curve:
            return 0.0
        peak = self.equity_curve[0]
        dd = 0.0
        for v in self.equity_curve:
            if v > peak:
                peak = v
            if peak > 0:
                dd = max(dd, (peak - v) / peak)
        return dd

    @property
    def avg_trade_pnl(self) -> float:
        n = self.total_trades
        return self.total_pnl / n if n else 0.0

    def summary(self) -> str:
        return (
            f"\n{'=' * 60}\n"
            f"  BTC 5-MIN BACKTEST RESULTS\n"
            f"{'=' * 60}\n"
            f"  Days:                 {self.config.days}\n"
            f"  Lag (s):              {self.config.lag_seconds}\n"
            f"  Initial Capital:      ${self.initial_capital:,.2f}\n"
            f"  Final Capital:        ${self.final_capital:,.2f}\n"
            f"  Total PnL:            ${self.total_pnl:+,.2f}\n"
            f"  Return:               {self.return_pct:+.2%}\n"
            f"  Max Drawdown:         {self.max_drawdown_pct:.2%}\n"
            f"  5-min Windows:        {self.total_windows}\n"
            f"  Windows Traded:       {self.windows_traded}\n"
            f"  Total Trades:         {self.total_trades}\n"
            f"  Wins / Losses:        {self.wins} / {self.losses}\n"
            f"  Win Rate:             {self.win_rate:.1%}\n"
            f"  Avg Trade PnL:        ${self.avg_trade_pnl:+.4f}\n"
            f"{'=' * 60}\n"
        )


# ── Core simulation ───────────────────────────────────────────────────

def _market_price_with_lag(
    strike: float,
    spot_now: float,
    spot_lagged: float,
    seconds_left: int,
    sigma_annual: float,
    spread: float,
) -> tuple[float, float]:
    """Simulate (yes_ask, no_ask) using a lagged BS quote.

    The "market maker" reprices from `spot_lagged` (their stale view),
    which is what gives our fresh-spot strategy an edge. We add a bid-ask
    spread symmetrically around the lagged fair price.
    """
    T = years_from_seconds(max(1, seconds_left))
    fair_up_lagged = digital_call_prob(S=spot_lagged, K=strike, T=T, sigma=sigma_annual)
    fair_up_lagged = max(0.01, min(0.99, fair_up_lagged))
    fair_down_lagged = 1.0 - fair_up_lagged

    half = spread / 2.0
    yes_ask = max(0.01, min(0.99, fair_up_lagged + half))
    no_ask = max(0.01, min(0.99, fair_down_lagged + half))
    return yes_ask, no_ask


def _synthesize_market(
    strike: float,
    yes_ask: float,
    no_ask: float,
    window_start_ts: int,
    window_end_ts: int,
) -> tuple[Market, dict[str, OrderBook]]:
    up_tok = Token(token_id=f"up_{window_start_ts}", outcome="Up", price=yes_ask)
    down_tok = Token(token_id=f"down_{window_start_ts}", outcome="Down", price=no_ask)
    market = Market(
        condition_id=f"cond_{window_start_ts}",
        slug=f"btc-updown-5m-{window_start_ts}",
        question="Will BTC be up or down in the next 5 minutes?",
        tokens=[up_tok, down_tok],
        start_ts=window_start_ts,
        end_ts=window_end_ts,
        strike_price=strike,
        active=True,
        liquidity=500.0,
    )
    books = {
        up_tok.token_id: OrderBook(
            token_id=up_tok.token_id,
            bids=[OrderBookLevel(price=max(0.01, yes_ask - 0.01), size=500.0)],
            asks=[OrderBookLevel(price=yes_ask, size=500.0)],
        ),
        down_tok.token_id: OrderBook(
            token_id=down_tok.token_id,
            bids=[OrderBookLevel(price=max(0.01, no_ask - 0.01), size=500.0)],
            asks=[OrderBookLevel(price=no_ask, size=500.0)],
        ),
    }
    return market, books


def run_backtest(config: BacktestConfig, klines: list[Kline] | None = None) -> BacktestResult:
    """Run the BTC 5-min strategy against Binance historical klines.

    Args:
        config: Backtest parameters.
        klines: Pre-fetched klines. If None, fetches `config.days` of 1-min
            BTCUSDT klines from Binance (cached to .cache/binance).
    """
    if klines is None:
        with BinanceHistoricalClient() as client:
            klines = client.fetch_last_n_days(config.days)

    if len(klines) < config.vol_window_minutes + 10:
        raise RuntimeError(
            f"Not enough klines for backtest: got {len(klines)}, "
            f"need at least {config.vol_window_minutes + 10}"
        )

    strat_config = StrategyConfig(
        vol_window_minutes=config.vol_window_minutes,
        vol_ewma_lambda=config.ewma_lambda,
        min_edge=config.min_edge,
        min_seconds_to_expiry=config.min_seconds_to_expiry,
        max_seconds_to_expiry=config.max_seconds_to_expiry,
        kelly_fraction=config.kelly_fraction,
        max_position_usd=config.max_position_usd,
        prefer_maker=True,
        post_only=True,
    )
    strategy = BTC5MinStrategy(strat_config)

    result = BacktestResult(
        config=config,
        initial_capital=config.initial_capital,
        final_capital=config.initial_capital,
    )
    cash = config.initial_capital
    result.equity_curve.append(cash)

    # Index klines by minute-open timestamp (ms) for O(1) lookups.
    k_by_ts = {k.open_time_ms: k for k in klines}
    sorted_ts = sorted(k_by_ts.keys())
    closes_by_ts: dict[int, float] = {ts: k_by_ts[ts].close for ts in sorted_ts}

    # Partition into 5-minute windows aligned to UTC (300s boundaries).
    # For each window, `window_start_ts` is the unix second at which the
    # window opens; the strike is the close of the minute immediately
    # before the window started.
    first_ts_s = sorted_ts[0] // 1000
    last_ts_s = sorted_ts[-1] // 1000

    # Align to the next 5-min boundary.
    window_start = ((first_ts_s + 299) // 300) * 300
    vol_window_bars = config.vol_window_minutes

    windows_with_signal = 0
    trades_executed = 0

    while window_start + 300 <= last_ts_s:
        window_end = window_start + 300
        strike_ms = (window_start - 60) * 1000  # close of previous minute
        if strike_ms not in k_by_ts:
            window_start += 300
            continue
        strike = k_by_ts[strike_ms].close

        # Resolution spot = close of the last minute of the window.
        resolve_ms = (window_end - 60) * 1000
        if resolve_ms not in k_by_ts:
            window_start += 300
            continue
        resolve_spot = k_by_ts[resolve_ms].close

        result.total_windows += 1
        window_had_signal = False

        # Replay each minute inside the window as a decision tick.
        # Tick at the END of minute t (i.e. we see that minute's close).
        open_trade: BacktestTrade | None = None
        for tick_minute in range(5):
            tick_ts = window_start + tick_minute * 60 + 60  # end of this minute
            tick_ms = (tick_ts - 60) * 1000
            if tick_ms not in k_by_ts:
                continue

            seconds_left = window_end - tick_ts
            spot = k_by_ts[tick_ms].close

            # Trailing EWMA vol from the preceding `vol_window_bars` closes.
            closes_history: list[float] = []
            for i in range(vol_window_bars, 0, -1):
                past_ms = tick_ms - i * 60_000
                c = closes_by_ts.get(past_ms)
                if c is not None:
                    closes_history.append(c)
            if len(closes_history) < 10:
                continue

            sigma = ewma_vol(closes_history, lam=config.ewma_lambda)
            if sigma <= 0:
                continue

            # Market-maker's lagged view.
            lag_ms = ((tick_ts - config.lag_seconds) - 60) * 1000
            lag_ms = (lag_ms // 60_000) * 60_000  # snap to the nearest minute
            spot_lagged = closes_by_ts.get(lag_ms, spot)

            yes_ask, no_ask = _market_price_with_lag(
                strike=strike,
                spot_now=spot,
                spot_lagged=spot_lagged,
                seconds_left=seconds_left,
                sigma_annual=sigma,
                spread=config.spread_cents,
            )
            market, books = _synthesize_market(strike, yes_ask, no_ask, window_start, window_end)

            ctx = BTC5MinContext(
                btc_spot=spot,
                btc_closes_1m=closes_history,
                now_ts=tick_ts,
                bankroll_usd=cash,
            )
            signals = strategy.generate_signals([market], books, ctx)
            if not signals:
                continue

            sig = signals[0]
            if open_trade is not None:
                # Already have a position in this window — skip further entries.
                continue

            # Quarter-Kelly dollar stake (strategy has already shrunk + computed).
            kelly_q = shrink_toward_market(sig.estimated_fair_value, sig.market_price, sig.confidence)
            kelly_f = kelly_fraction(
                q=kelly_q,
                p=sig.market_price,
                fraction=config.kelly_fraction,
            )
            size_usd = min(
                kelly_f * cash,
                config.max_position_usd,
                cash * 0.95,
            )
            if size_usd < 1.0:
                continue
            shares = size_usd / sig.market_price
            cost = shares * sig.market_price
            cash -= cost  # maker fee = 0

            open_trade = BacktestTrade(
                window_start_ts=window_start,
                window_end_ts=window_end,
                side_outcome=sig.outcome,
                strike=strike,
                entry_spot=spot,
                entry_seconds_left=seconds_left,
                entry_price=sig.market_price,
                fair_at_entry=sig.estimated_fair_value,
                sigma_at_entry=sigma,
                size_usd=cost,
                shares=shares,
            )
            trades_executed += 1
            window_had_signal = True
            if config.verbose:
                logger.info(
                    "backtest_trade_opened",
                    window=window_start,
                    side=sig.outcome,
                    strike=round(strike, 2),
                    spot=round(spot, 2),
                    fair=round(sig.estimated_fair_value, 4),
                    price=round(sig.market_price, 4),
                    shares=round(shares, 2),
                )

        if open_trade is not None:
            won_up = resolve_spot >= strike
            won = (open_trade.side_outcome.lower() in ("up", "yes")) == won_up
            payout = open_trade.shares * (1.0 if won else 0.0)
            cash += payout
            open_trade.resolved = won
            open_trade.exit_spot = resolve_spot
            open_trade.pnl = payout - open_trade.size_usd
            result.trades.append(open_trade)

        result.equity_curve.append(cash)
        if window_had_signal:
            windows_with_signal += 1
        window_start += 300

    result.windows_traded = windows_with_signal
    result.final_capital = cash
    return result
