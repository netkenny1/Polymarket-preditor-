"""BTC 5-minute up/down strategy: Black-Scholes fair + Kelly sizing.

This is the only strategy. It targets Polymarket's `btc-updown-5m-{ts}`
market series (a fresh binary every 300s asking "will BTC close up or
down in the next 5 minutes?").

Edge source: the market's implied probability (= the Up token ask price)
occasionally diverges from the Black-Scholes fair probability computed
from the real-time Binance BTC spot, the window-start strike price, and
the trailing realized volatility. When that gap exceeds fees + a small
buffer, we buy the undervalued side and size the bet with fractional
Kelly.

Mechanics:
  1. Pull window start strike S0 (BTC price when the window opened).
  2. Read current BTC mid `St` from Binance bookTicker.
  3. Read trailing realized volatility `sigma` from 60min of 1-min closes.
  4. Compute BS digital-call prob `q = P(S_end > S0)` with T = seconds_left/YEAR.
  5. Compare to market:
       - q > yes_ask + min_edge  → buy YES (the Up token)
       - q < 1 - no_ask - min_edge → buy NO  (the Down token)
  6. Size with quarter-Kelly on bankroll, capped at max_position_usd.
  7. Post maker-only (POST_ONLY) — taker fees eat any edge here.

This is both "the BTC arbitrage strategy" (because the edge comes from
cross-venue price disagreement with a CEX) and "the Black-Scholes options
strategy" (because the fair price is a digital option formula). They are
the same thing at this timescale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import structlog

from polymarket_bot.config import StrategyConfig
from polymarket_bot.data.models import Market, OrderBook, Side, Signal, Token
from polymarket_bot.pricing.black_scholes import (
    digital_call_prob,
    ewma_vol,
    years_from_seconds,
)
from polymarket_bot.pricing.kelly import (
    kelly_fraction,
    polymarket_taker_fee,
    shrink_toward_market,
)
from polymarket_bot.strategies.base import BaseStrategy

logger = structlog.get_logger()


@dataclass
class BTC5MinContext:
    """Per-tick context passed into `generate_signals`."""

    btc_spot: float  # current Binance mid
    btc_closes_1m: Sequence[float]  # trailing 1-minute closes for vol
    now_ts: int  # current unix seconds
    bankroll_usd: float = 100.0


@dataclass
class FairPriceBreakdown:
    """Diagnostic breakdown for a single fair-value computation."""

    spot: float
    strike: float
    seconds_left: int
    sigma_annual: float
    fair_up: float
    fair_down: float
    d2: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "spot": round(self.spot, 2),
            "strike": round(self.strike, 2),
            "seconds_left": self.seconds_left,
            "sigma_annual": round(self.sigma_annual, 4),
            "fair_up": round(self.fair_up, 4),
            "fair_down": round(self.fair_down, 4),
            "d2": round(self.d2, 4),
        }


class BTC5MinStrategy(BaseStrategy):
    """Black-Scholes + Kelly strategy for the BTC 5-min up/down market."""

    name = "btc_5min"

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    # ── Pure fair-price computation (no I/O, easy to unit test) ──

    def compute_fair(
        self,
        spot: float,
        strike: float,
        seconds_left: int,
        sigma_annual: float,
    ) -> FairPriceBreakdown:
        """Black-Scholes fair probabilities for UP and DOWN outcomes."""
        sigma_annual = self._clamp_vol(sigma_annual)
        T = years_from_seconds(max(0, seconds_left))

        fair_up = digital_call_prob(S=spot, K=strike, T=T, sigma=sigma_annual)
        fair_up = max(self.config.prob_clamp_min, min(self.config.prob_clamp_max, fair_up))
        fair_down = 1.0 - fair_up

        # Diagnostic d2 (same formula as inside digital_call_prob)
        if T > 0 and sigma_annual > 0 and spot > 0 and strike > 0:
            s_sqrt_t = sigma_annual * math.sqrt(T)
            d2 = (math.log(spot / strike) - 0.5 * sigma_annual * sigma_annual * T) / s_sqrt_t
        else:
            d2 = 0.0

        return FairPriceBreakdown(
            spot=spot,
            strike=strike,
            seconds_left=seconds_left,
            sigma_annual=sigma_annual,
            fair_up=fair_up,
            fair_down=fair_down,
            d2=d2,
        )

    def _clamp_vol(self, sigma: float) -> float:
        return max(self.config.min_annual_vol, min(self.config.max_annual_vol, sigma))

    # ── Core signal generation ────────────────────────────────────

    def generate_signals(
        self,
        markets: list[Market],
        order_books: dict[str, OrderBook],
        context: dict[str, Any] | BTC5MinContext,
    ) -> list[Signal]:
        """Produce zero or more signals for the active BTC 5-min window(s)."""
        ctx = self._coerce_context(context)
        if ctx is None:
            return []

        if ctx.btc_spot <= 0:
            logger.debug("btc_5min_skip_no_spot")
            return []

        sigma = ewma_vol(
            ctx.btc_closes_1m,
            lam=self.config.vol_ewma_lambda,
        )
        if sigma <= 0:
            logger.debug("btc_5min_skip_no_vol")
            return []

        signals: list[Signal] = []
        for market in markets:
            if not market.active:
                continue
            sig = self._evaluate_market(market, order_books, ctx, sigma)
            if sig is not None:
                signals.append(sig)

        return signals

    def _evaluate_market(
        self,
        market: Market,
        order_books: dict[str, OrderBook],
        ctx: BTC5MinContext,
        sigma_annual: float,
    ) -> Signal | None:
        seconds_left = market.seconds_remaining(ctx.now_ts)
        if seconds_left < self.config.min_seconds_to_expiry:
            return None
        if seconds_left > self.config.max_seconds_to_expiry:
            # Window just opened — spot has barely moved, fair ≈ 0.5 ≈ market.
            return None

        strike = market.strike_price or ctx.btc_spot
        fair = self.compute_fair(
            spot=ctx.btc_spot,
            strike=strike,
            seconds_left=seconds_left,
            sigma_annual=sigma_annual,
        )

        up_tok, down_tok = market.up_token, market.down_token
        if up_tok is None or down_tok is None:
            return None

        up_book = order_books.get(up_tok.token_id)
        down_book = order_books.get(down_tok.token_id)
        up_ask = up_book.best_ask if up_book else up_tok.price
        down_ask = down_book.best_ask if down_book else down_tok.price
        if up_ask is None or down_ask is None:
            return None
        if not (0.01 < up_ask < 0.99) or not (0.01 < down_ask < 0.99):
            return None

        up_edge = fair.fair_up - up_ask
        down_edge = fair.fair_down - down_ask

        # Subtract the taker fee from the apparent edge unless we will post maker-only.
        fee_bump = 0.0 if self.config.prefer_maker else polymarket_taker_fee(
            up_ask, peak_fee=self.config.taker_fee_peak
        )

        if up_edge - fee_bump >= self.config.min_edge and up_edge >= down_edge:
            return self._make_signal(
                market=market,
                token=up_tok,
                fair_value=fair.fair_up,
                market_price=up_ask,
                edge=up_edge - fee_bump,
                fair=fair,
                seconds_left=seconds_left,
            )

        if down_edge - fee_bump >= self.config.min_edge and down_edge > up_edge:
            return self._make_signal(
                market=market,
                token=down_tok,
                fair_value=fair.fair_down,
                market_price=down_ask,
                edge=down_edge - fee_bump,
                fair=fair,
                seconds_left=seconds_left,
            )

        return None

    def _make_signal(
        self,
        market: Market,
        token: Token,
        fair_value: float,
        market_price: float,
        edge: float,
        fair: FairPriceBreakdown,
        seconds_left: int,
    ) -> Signal:
        # Confidence scales with (a) how much time is left (more time = more
        # chance to be wrong) and (b) how big the edge is relative to spread.
        time_score = min(1.0, seconds_left / 180.0)  # full conf at ~3min left
        edge_score = min(1.0, edge / 0.05)  # full conf at 5c edge
        confidence = 0.4 + 0.3 * time_score + 0.3 * edge_score
        confidence = max(0.1, min(0.95, confidence))

        kelly_q = shrink_toward_market(fair_value, market_price, confidence)
        kelly_f = kelly_fraction(
            q=kelly_q,
            p=market_price,
            fraction=self.config.kelly_fraction,
        )

        return Signal(
            market_condition_id=market.condition_id,
            token_id=token.token_id,
            side=Side.BUY,
            outcome=token.outcome,
            estimated_fair_value=fair_value,
            market_price=market_price,
            edge=edge,
            confidence=confidence,
            strategy=self.name,
            metadata={
                **fair.as_dict(),
                "kelly_fraction": round(kelly_f, 4),
                "shrunk_fair": round(kelly_q, 4),
                "post_only": self.config.post_only,
            },
        )

    @staticmethod
    def _coerce_context(context: dict[str, Any] | BTC5MinContext | None) -> BTC5MinContext | None:
        if context is None:
            return None
        if isinstance(context, BTC5MinContext):
            return context
        try:
            return BTC5MinContext(
                btc_spot=float(context.get("btc_spot", 0.0)),
                btc_closes_1m=list(context.get("btc_closes_1m", [])),
                now_ts=int(context.get("now_ts", 0)),
                bankroll_usd=float(context.get("bankroll_usd", 100.0)),
            )
        except (TypeError, ValueError):
            return None
