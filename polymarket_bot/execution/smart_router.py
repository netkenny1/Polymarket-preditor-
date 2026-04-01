"""Smart Order Router for optimal execution on Polymarket's CLOB.

Instead of placing a single limit order at a fixed offset from market price,
the smart router analyzes the order book to:
  1. Compute the true cost of execution via VWAP
  2. Estimate market impact based on order size relative to book depth
  3. Set limit prices that account for impact
  4. Split large orders into smaller slices to reduce market footprint

Usage:
    router = SmartOrderRouter()
    slices = router.route_order(signal, order_book, total_size_usd=250.0)
    for s in slices:
        await asyncio.sleep(s.delay_ms / 1000)
        client.place_order(..., price=s.price, size=s.size, ...)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import structlog

from polymarket_bot.data.models import OrderBook, OrderBookLevel, Side, Signal
from polymarket_bot.utils.helpers import round_price

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_IMPACT_COEFF = 0.1
"""Scaling coefficient for the square-root market impact model."""

SPLIT_THRESHOLD_PCT = 0.30
"""Split the order when its size exceeds 30% of the top-3-level average."""

MAX_SLICES = 5
"""Maximum number of child slices per parent order."""

SLICE_DELAY_MS = 500
"""Base delay in milliseconds between consecutive slices."""


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class OrderSlice:
    """A single child order produced by the router.

    Attributes:
        price:    Limit price for this slice.
        size:     Size in *shares* (not USD).
        delay_ms: Milliseconds to wait before sending this slice
                  (relative to the first slice, which has delay_ms=0).
    """

    price: float
    size: float
    delay_ms: int


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class SmartOrderRouter:
    """Analyze an order book and produce execution slices for a signal.

    The router is stateless -- all market context is passed per call so it
    can be shared across markets and strategies.

    Parameters:
        impact_coeff:       Coefficient for the sqrt impact model (default 0.1).
        split_threshold_pct: Fraction of avg top-level size above which we split.
        max_slices:         Cap on the number of child slices.
        slice_delay_ms:     Base inter-slice delay in milliseconds.
    """

    def __init__(
        self,
        impact_coeff: float = DEFAULT_IMPACT_COEFF,
        split_threshold_pct: float = SPLIT_THRESHOLD_PCT,
        max_slices: int = MAX_SLICES,
        slice_delay_ms: int = SLICE_DELAY_MS,
    ) -> None:
        self.impact_coeff = impact_coeff
        self.split_threshold_pct = split_threshold_pct
        self.max_slices = max_slices
        self.slice_delay_ms = slice_delay_ms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route_order(
        self,
        signal: Signal,
        order_book: OrderBook,
        total_size_usd: float,
    ) -> list[OrderSlice]:
        """Produce an execution plan for *total_size_usd* worth of shares.

        Steps:
          1. Pick the relevant side of the book.
          2. Compute VWAP and total book depth.
          3. Estimate market impact.
          4. Derive the optimal limit price.
          5. Decide whether to split and, if so, how many slices.
          6. Return a list of ``OrderSlice`` objects, largest first.

        Returns an empty list when the book is too thin to trade safely
        (e.g. no levels on the relevant side).
        """
        side = signal.side
        levels = order_book.asks if side == Side.BUY else order_book.bids

        if not levels:
            logger.warning(
                "smart_router_empty_book",
                token_id=signal.token_id,
                side=side.value,
            )
            return []

        # Book depth in USD (price * size summed across all levels).
        book_depth_usd = self._book_depth_usd(levels)
        if book_depth_usd <= 0:
            return []

        # Average size of the top-3 levels (in shares).
        avg_top_level_size = self._avg_top_level_size(levels, n=3)

        # VWAP for the full order (informational / logging).
        vwap = self._compute_vwap(order_book, side, total_size_usd)

        # Market impact estimate.
        impact = self._estimate_market_impact(total_size_usd, book_depth_usd)

        # Optimal limit price incorporating impact.
        limit_price = self._optimal_limit_price(signal, order_book, impact)

        # Convert USD amount to share count at our limit price.
        total_shares = total_size_usd / limit_price if limit_price > 0 else 0.0
        if total_shares <= 0:
            return []

        # Decide whether to split.
        if self._should_split(total_shares, avg_top_level_size):
            num_slices = self._choose_num_slices(total_shares, avg_top_level_size)
            share_slices = self._split_order(total_shares, num_slices)
        else:
            share_slices = [total_shares]

        # Build OrderSlice list.
        slices: list[OrderSlice] = []
        for idx, slice_size in enumerate(share_slices):
            slices.append(
                OrderSlice(
                    price=limit_price,
                    size=round(slice_size, 2),
                    delay_ms=self.slice_delay_ms * idx,
                )
            )

        logger.info(
            "smart_router_plan",
            token_id=signal.token_id,
            side=side.value,
            total_usd=round(total_size_usd, 2),
            vwap=round(vwap, 4),
            impact=round(impact, 6),
            limit_price=limit_price,
            num_slices=len(slices),
            book_depth_usd=round(book_depth_usd, 2),
        )

        return slices

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    def _compute_vwap(
        self,
        book: OrderBook,
        side: Side,
        size_usd: float,
    ) -> float:
        """Walk the order book to compute the volume-weighted average price
        required to fill *size_usd* worth of shares.

        For a BUY we walk the asks (ascending); for a SELL we walk the bids
        (descending -- they are already sorted best-first in the model).

        If the book is too thin to fill the full amount, the VWAP is computed
        over whatever liquidity is available.
        """
        levels = book.asks if side == Side.BUY else book.bids

        remaining_usd = size_usd
        total_cost = 0.0
        total_shares = 0.0

        for level in levels:
            level_value = level.price * level.size  # USD available at this level
            if level_value <= 0:
                continue

            if level_value >= remaining_usd:
                # This level can absorb the rest of the order.
                shares_here = remaining_usd / level.price
                total_cost += shares_here * level.price
                total_shares += shares_here
                remaining_usd = 0.0
                break
            else:
                # Consume the entire level and move on.
                total_cost += level_value
                total_shares += level.size
                remaining_usd -= level_value

        if total_shares == 0:
            # Fallback: mid price or best available.
            mid = book.mid_price
            return mid if mid is not None else 0.0

        return total_cost / total_shares

    def _estimate_market_impact(
        self,
        size_usd: float,
        book_depth_usd: float,
    ) -> float:
        """Estimate price impact using a square-root model.

        impact = impact_coeff * sqrt(order_value / total_book_depth)

        This is a standard approximation from equity microstructure research
        adapted for prediction-market order books.
        """
        if book_depth_usd <= 0 or size_usd <= 0:
            return 0.0
        return self.impact_coeff * math.sqrt(size_usd / book_depth_usd)

    def _optimal_limit_price(
        self,
        signal: Signal,
        book: OrderBook,
        impact: float,
    ) -> float:
        """Compute the limit price for the order, accounting for impact.

        For BUY orders we start from the best ask and *add* the expected
        impact to ensure our limit is aggressive enough to fill, but we
        cap it at the signal's estimated fair value so we never overpay.

        For SELL orders we start from the best bid and *subtract* impact,
        flooring at a minimum viable price.
        """
        if signal.side == Side.BUY:
            reference = book.best_ask if book.best_ask is not None else signal.market_price
            # Widen our limit by the expected impact so the order still fills
            # even after our own footprint moves the book.
            raw_price = reference + impact
            # Never pay more than our fair-value estimate.
            capped = min(raw_price, signal.estimated_fair_value)
            price = round_price(capped)
        else:
            reference = book.best_bid if book.best_bid is not None else signal.market_price
            raw_price = reference - impact
            # Never sell below a minimum floor.
            price = round_price(max(raw_price, 0.01))

        # Clamp into Polymarket's valid price range.
        return max(0.01, min(0.99, price))

    def _should_split(self, size_shares: float, avg_level_size: float) -> bool:
        """Return True if the order is large enough relative to the book
        that splitting will reduce market impact.

        Threshold: order size > 30% of the average top-3-level size.
        """
        if avg_level_size <= 0:
            return False
        return size_shares > self.split_threshold_pct * avg_level_size

    def _split_order(self, total_size: float, num_slices: int) -> list[float]:
        """Split *total_size* into *num_slices* decreasing tranches.

        The largest slice is sent first to capture the best available
        prices before they are consumed.  Slice weights follow a simple
        linearly-decreasing pattern:

            weight_i = (num_slices - i)  /  sum(1..num_slices)

        Example with 3 slices: weights = [3/6, 2/6, 1/6] = [50%, 33%, 17%].
        """
        num_slices = max(1, min(num_slices, self.max_slices))

        weight_sum = num_slices * (num_slices + 1) / 2
        slices: list[float] = []
        allocated = 0.0

        for i in range(num_slices):
            weight = (num_slices - i) / weight_sum
            if i == num_slices - 1:
                # Put any remaining rounding residual in the last slice.
                slice_size = total_size - allocated
            else:
                slice_size = total_size * weight
                allocated += slice_size
            slices.append(round(slice_size, 2))

        return slices

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _book_depth_usd(levels: list[OrderBookLevel]) -> float:
        """Total USD value across all levels (price * size)."""
        return sum(lvl.price * lvl.size for lvl in levels)

    @staticmethod
    def _avg_top_level_size(levels: list[OrderBookLevel], n: int = 3) -> float:
        """Average share size of the top *n* levels."""
        top = levels[:n]
        if not top:
            return 0.0
        return sum(lvl.size for lvl in top) / len(top)

    def _choose_num_slices(
        self,
        total_shares: float,
        avg_level_size: float,
    ) -> int:
        """Heuristic: one slice per 30%-of-average-level chunk, capped."""
        if avg_level_size <= 0:
            return 1
        chunk = self.split_threshold_pct * avg_level_size
        raw = math.ceil(total_shares / chunk)
        return max(2, min(raw, self.max_slices))
