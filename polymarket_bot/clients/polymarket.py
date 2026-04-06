"""Polymarket CLOB API client for market data and order management."""

from __future__ import annotations

import random
import time
from datetime import datetime
from typing import Any, Optional

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from polymarket_bot.config import PolymarketConfig
from polymarket_bot.data.models import (
    Market,
    MarketCategory,
    Order,
    OrderBook,
    OrderBookLevel,
    OrderStatus,
    Position,
    Side,
    Token,
    TradeResult,
)
from polymarket_bot.utils.helpers import categorize_market

logger = structlog.get_logger()


class PolymarketClient:
    """Client for the Polymarket CLOB and Gamma APIs.

    Handles fetching market data, order books, placing orders, and
    managing positions through the Polymarket REST APIs.
    """

    FILL_POLL_ATTEMPTS = 3
    FILL_POLL_INTERVAL_S = 2.0

    def __init__(self, config: PolymarketConfig) -> None:
        self.config = config
        self._clob_client = httpx.Client(
            base_url=config.base_url,
            timeout=30.0,
            headers=self._build_headers(),
        )
        self._gamma_client = httpx.Client(
            base_url=config.gamma_url,
            timeout=30.0,
        )
        self._pending_orders: dict[str, Order] = {}

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["POLY_API_KEY"] = self.config.api_key
            headers["POLY_API_SECRET"] = self.config.api_secret
            headers["POLY_PASSPHRASE"] = self.config.api_passphrase
        return headers

    def close(self) -> None:
        if self._pending_orders:
            logger.warning(
                "closing_with_pending_orders",
                count=len(self._pending_orders),
                order_ids=list(self._pending_orders.keys()),
            )
            try:
                self.cancel_all_orders()
            except Exception:
                logger.exception("failed_to_cancel_orders_on_close")
        self._pending_orders.clear()
        self._clob_client.close()
        self._gamma_client.close()

    # ── Market Data ──────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        active: bool = True,
        closed: bool = False,
    ) -> list[Market]:
        """Fetch available markets from the Gamma API."""
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "active": active,
            "closed": closed,
        }
        resp = self._gamma_client.get("/markets", params=params)
        resp.raise_for_status()
        raw_markets = resp.json()
        return [self._parse_market(m) for m in raw_markets if m.get("enableOrderBook")]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_market(self, condition_id: str) -> Market:
        """Fetch a single market by condition ID."""
        resp = self._gamma_client.get(f"/markets/{condition_id}")
        resp.raise_for_status()
        return self._parse_market(resp.json())

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_order_book(self, token_id: str) -> OrderBook:
        """Fetch the order book for a specific token."""
        resp = self._clob_client.get("/book", params={"token_id": token_id})
        resp.raise_for_status()
        data = resp.json()

        bids = [
            OrderBookLevel(price=float(b["price"]), size=float(b["size"]))
            for b in data.get("bids", [])
        ]
        asks = [
            OrderBookLevel(price=float(a["price"]), size=float(a["size"]))
            for a in data.get("asks", [])
        ]
        # Sort: bids descending, asks ascending
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)

        return OrderBook(token_id=token_id, bids=bids, asks=asks)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_midpoint(self, token_id: str) -> Optional[float]:
        """Get the midpoint price for a token."""
        resp = self._clob_client.get("/midpoint", params={"token_id": token_id})
        resp.raise_for_status()
        data = resp.json()
        mid = data.get("mid")
        return float(mid) if mid is not None else None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_price(self, token_id: str, side: Side) -> Optional[float]:
        """Get the best available price for a side."""
        resp = self._clob_client.get(
            "/price", params={"token_id": token_id, "side": side.value}
        )
        resp.raise_for_status()
        data = resp.json()
        price = data.get("price")
        return float(price) if price is not None else None

    def get_prices_batch(self, token_ids: list[str]) -> dict[str, float]:
        """Get midpoint prices for multiple tokens concurrently."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        prices = {}
        with ThreadPoolExecutor(max_workers=10) as pool:
            future_to_tid = {pool.submit(self.get_midpoint, tid): tid for tid in token_ids}
            for future in as_completed(future_to_tid):
                tid = future_to_tid[future]
                try:
                    mid = future.result(timeout=10)
                    if mid is not None:
                        prices[tid] = mid
                except Exception:
                    pass
        return prices

    def get_order_books_batch(self, token_ids: list[str]) -> dict[str, OrderBook]:
        """Fetch order books for multiple tokens concurrently."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        books = {}
        with ThreadPoolExecutor(max_workers=10) as pool:
            future_to_tid = {pool.submit(self.get_order_book, tid): tid for tid in token_ids}
            for future in as_completed(future_to_tid):
                tid = future_to_tid[future]
                try:
                    book = future.result(timeout=10)
                    if book is not None:
                        books[tid] = book
                except Exception:
                    pass
        return books

    # ── Order Management ─────────────────────────────────────────

    def place_order(
        self,
        token_id: str,
        side: Side,
        price: float,
        size: float,
        market_condition_id: str = "",
        strategy: str = "",
    ) -> TradeResult:
        """Place a GTC limit order on the CLOB and poll for fills.

        Submits the order, then polls up to ``FILL_POLL_ATTEMPTS`` times to
        detect immediate or fast fills.  If the order is still open after
        polling it is stored in ``_pending_orders`` for later reconciliation
        via ``check_pending_orders()``.
        """
        from polymarket_bot.utils.helpers import generate_order_id

        order = Order(
            order_id=generate_order_id(),
            market_condition_id=market_condition_id,
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            strategy=strategy,
        )

        logger.info(
            "placing_order",
            order_id=order.order_id,
            token_id=token_id,
            side=side.value,
            price=price,
            size=size,
        )

        try:
            payload = {
                "tokenID": token_id,
                "side": side.value,
                "price": str(price),
                "size": str(size),
                "type": "GTC",
            }
            resp = self._clob_client.post("/order", json=payload)
            resp.raise_for_status()
            data = resp.json()

            order.order_id = data.get("orderID", order.order_id)
            order.status = OrderStatus.OPEN

        except httpx.HTTPError as e:
            logger.error("order_submission_failed", error=str(e))
            order.status = OrderStatus.FAILED
            return TradeResult(order=order, success=False, error=str(e))

        fill_data = self._check_order_status(order.order_id)

        if fill_data is not None:
            fill_price = fill_data["fill_price"]
            fill_size = fill_data["fill_size"]
            is_full = fill_size >= size

            order.filled_size = fill_size
            order.status = OrderStatus.FILLED if is_full else OrderStatus.PARTIAL

            if not is_full:
                self._pending_orders[order.order_id] = order

            logger.info(
                "order_filled",
                order_id=order.order_id,
                status=order.status.value,
                fill_price=fill_price,
                fill_size=fill_size,
            )
            return TradeResult(
                order=order,
                success=True,
                fill_price=fill_price,
                fill_size=fill_size,
            )

        self._pending_orders[order.order_id] = order
        logger.info(
            "order_pending",
            order_id=order.order_id,
            msg="order still open after polling",
        )
        return TradeResult(
            order=order,
            success=False,
            error="order_pending",
        )

    # ── Fill & Order Status Polling ──────────────────────────────

    def _check_order_status(
        self, order_id: str
    ) -> Optional[dict[str, float]]:
        """Poll the CLOB for fill data on *order_id*.

        Returns ``{"fill_price": …, "fill_size": …}`` when fills are
        detected, or ``None`` if the order is still live after all attempts.
        """
        for attempt in range(1, self.FILL_POLL_ATTEMPTS + 1):
            time.sleep(self.FILL_POLL_INTERVAL_S)
            try:
                resp = self._clob_client.get(f"/order/{order_id}")
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError as e:
                logger.warning(
                    "order_status_poll_failed",
                    order_id=order_id,
                    attempt=attempt,
                    error=str(e),
                )
                continue

            status = data.get("status", "").lower()

            if status == "matched":
                fills = data.get("fills", [])
                if fills:
                    total_size = sum(float(f["size"]) for f in fills)
                    total_cost = sum(
                        float(f["price"]) * float(f["size"]) for f in fills
                    )
                    avg_price = total_cost / total_size if total_size else 0.0
                    return {"fill_price": avg_price, "fill_size": total_size}
                return {
                    "fill_price": float(data.get("price", 0)),
                    "fill_size": float(data.get("size", 0)),
                }

            if status == "cancelled":
                return None

            logger.debug(
                "order_still_open",
                order_id=order_id,
                attempt=attempt,
                status=status,
            )

        return None

    def get_open_orders(self) -> list[Order]:
        """Fetch the user's currently open orders from the CLOB."""
        try:
            resp = self._clob_client.get("/orders", params={"state": "live"})
            resp.raise_for_status()
            raw_orders = resp.json()
        except httpx.HTTPError as e:
            logger.error("get_open_orders_failed", error=str(e))
            return []

        orders: list[Order] = []
        for raw in raw_orders:
            orders.append(
                Order(
                    order_id=raw.get("orderID", raw.get("id", "")),
                    market_condition_id=raw.get("market", ""),
                    token_id=raw.get("tokenID", raw.get("asset_id", "")),
                    side=Side(raw["side"]) if raw.get("side") in ("BUY", "SELL") else Side.BUY,
                    price=float(raw.get("price", 0)),
                    size=float(raw.get("original_size", raw.get("size", 0))),
                    filled_size=float(raw.get("size_matched", 0)),
                    status=OrderStatus.OPEN,
                )
            )
        return orders

    def check_pending_orders(self) -> list[TradeResult]:
        """Re-check all orders stored in ``_pending_orders``.

        Returns a list of ``TradeResult`` objects for orders that have
        received new fills since the last check.  Fully filled or cancelled
        orders are removed from the pending set.
        """
        newly_filled: list[TradeResult] = []
        resolved_ids: list[str] = []

        for order_id, order in self._pending_orders.items():
            try:
                resp = self._clob_client.get(f"/order/{order_id}")
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError as e:
                logger.warning(
                    "pending_order_check_failed",
                    order_id=order_id,
                    error=str(e),
                )
                continue

            status = data.get("status", "").lower()

            if status == "cancelled":
                order.status = OrderStatus.CANCELLED
                resolved_ids.append(order_id)
                continue

            fills = data.get("fills", [])
            if not fills:
                continue

            total_size = sum(float(f["size"]) for f in fills)
            total_cost = sum(
                float(f["price"]) * float(f["size"]) for f in fills
            )
            avg_price = total_cost / total_size if total_size else 0.0

            new_fill = total_size - order.filled_size
            if new_fill <= 0:
                continue

            order.filled_size = total_size
            is_full = total_size >= order.size

            if is_full:
                order.status = OrderStatus.FILLED
                resolved_ids.append(order_id)
            else:
                order.status = OrderStatus.PARTIAL

            if status == "matched":
                order.status = OrderStatus.FILLED
                if order_id not in resolved_ids:
                    resolved_ids.append(order_id)

            logger.info(
                "pending_order_fill_detected",
                order_id=order_id,
                new_fill=new_fill,
                total_filled=total_size,
                status=order.status.value,
            )

            newly_filled.append(
                TradeResult(
                    order=order,
                    success=True,
                    fill_price=avg_price,
                    fill_size=new_fill,
                )
            )

        for oid in resolved_ids:
            del self._pending_orders[oid]

        return newly_filled

    def get_positions(self) -> list[Position]:
        """Fetch actual positions from the CLOB / on-chain state.

        Used to reconcile the bot's internal portfolio against reality.
        """
        try:
            resp = self._clob_client.get("/positions")
            resp.raise_for_status()
            raw_positions = resp.json()
        except httpx.HTTPError as e:
            logger.error("get_positions_failed", error=str(e))
            return []

        positions: list[Position] = []
        for raw in raw_positions:
            size = float(raw.get("size", 0))
            if size == 0:
                continue
            positions.append(
                Position(
                    market_condition_id=raw.get("market", raw.get("condition_id", "")),
                    token_id=raw.get("asset_id", raw.get("tokenID", "")),
                    outcome=raw.get("outcome", ""),
                    size=size,
                    avg_entry_price=float(raw.get("avg_price", 0)),
                    current_price=float(raw.get("cur_price", raw.get("price", 0))),
                )
            )
        return positions

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        try:
            resp = self._clob_client.delete(f"/order/{order_id}")
            resp.raise_for_status()
            self._pending_orders.pop(order_id, None)
            return True
        except httpx.HTTPError as e:
            logger.error("cancel_failed", order_id=order_id, error=str(e))
            return False

    def cancel_all_orders(self) -> bool:
        """Cancel all open orders."""
        try:
            resp = self._clob_client.delete("/orders")
            resp.raise_for_status()
            self._pending_orders.clear()
            return True
        except httpx.HTTPError as e:
            logger.error("cancel_all_failed", error=str(e))
            return False

    # ── Parsing ──────────────────────────────────────────────────

    def _parse_market(self, raw: dict[str, Any]) -> Market:
        """Parse raw API response into a Market model."""
        tokens = []
        for t in raw.get("tokens", []):
            tokens.append(
                Token(
                    token_id=t.get("token_id", ""),
                    outcome=t.get("outcome", ""),
                    price=float(t.get("price", 0.5)),
                )
            )

        end_date = None
        if raw.get("end_date_iso"):
            try:
                end_date = datetime.fromisoformat(
                    raw["end_date_iso"].replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        question = raw.get("question", "")
        tags = raw.get("tags", []) or []
        cat = categorize_market(question, tags)

        return Market(
            condition_id=raw.get("condition_id", ""),
            question=question,
            slug=raw.get("slug", raw.get("market_slug", "")),
            tokens=tokens,
            category=MarketCategory(cat) if cat in MarketCategory.__members__.values() else MarketCategory.OTHER,
            end_date=end_date,
            volume_24h=float(raw.get("volume_num_24hr", 0) or 0),
            liquidity=float(raw.get("liquidity_num", 0) or 0),
            active=raw.get("active", True),
            description=raw.get("description", ""),
            tags=tags,
        )


class PaperTradingClient(PolymarketClient):
    """Simulated client for paper trading / backtesting.

    Overrides order execution to simulate fills locally with realistic
    slippage, partial fills, and market impact modeling.
    """

    def __init__(self, config: PolymarketConfig) -> None:
        super().__init__(config)
        self.simulated_orders: list[Order] = []
        self.simulated_fills: list[TradeResult] = []
        self._simulated_prices: dict[str, float] = {}
        self._simulated_volumes: dict[str, float] = {}  # For market impact
        self._order_books: dict[str, list[tuple[float, float]]] = {}  # Simplified books
        self._fill_attempts: int = 0
        self._fill_successes: int = 0

    @property
    def fill_rate(self) -> float:
        return self._fill_successes / max(1, self._fill_attempts)

    def set_simulated_prices(self, prices: dict[str, float]) -> None:
        """Set simulated prices for backtesting."""
        self._simulated_prices = prices

    def set_simulated_volumes(self, volumes: dict[str, float]) -> None:
        """Set simulated market volumes for impact modeling."""
        self._simulated_volumes = volumes

    def get_midpoint(self, token_id: str) -> Optional[float]:
        """Return simulated price if available, else try real API."""
        if token_id in self._simulated_prices:
            return self._simulated_prices[token_id]
        try:
            return super().get_midpoint(token_id)
        except Exception:
            return self._simulated_prices.get(token_id, 0.5)

    def place_order(
        self,
        token_id: str,
        side: Side,
        price: float,
        size: float,
        market_condition_id: str = "",
        strategy: str = "",
    ) -> TradeResult:
        """Simulate order execution with realistic slippage + adverse selection.

        Models:
        1. Fill price based on MID PRICE (not order price) to prevent
           instant paper profit from limit orders below mid
        2. Spread crossing cost + market impact + adverse selection
        3. Partial fills for large orders
        4. 2% taker fees
        """
        from polymarket_bot.utils.helpers import generate_order_id

        self._fill_attempts += 1

        mid_price = self._simulated_prices.get(token_id, 0.5)

        # ── Slippage model (applied to MID, not order price) ─────
        market_vol = self._simulated_volumes.get(token_id, 5000.0)
        order_value = size * price
        impact_ratio = order_value / max(market_vol, 100.0)

        spread_cost = 0.008
        market_impact = impact_ratio * 0.08
        adverse_selection = 0.005
        total_slippage = spread_cost + market_impact + adverse_selection

        if side == Side.BUY:
            fill_price = mid_price * (1 + total_slippage)
        else:
            fill_price = mid_price * (1 - total_slippage)

        fill_price = max(0.01, min(0.99, fill_price))

        # ── Fill probability ─────────────────────────────────────
        mid_denom = max(mid_price, 0.01)
        if side == Side.BUY:
            price_distance = (price - mid_price) / mid_denom
        else:
            price_distance = (mid_price - price) / mid_denom

        # Passive orders (below mid for buys) are less likely to fill
        fill_prob = 0.55
        if price_distance < 0:
            fill_prob -= abs(price_distance) * 4.0
        if price_distance > 0:
            fill_prob += price_distance * 0.8
        fill_prob *= max(0.4, 1.0 - impact_ratio * 0.6)
        fill_prob = max(0.10, min(0.85, fill_prob))

        if random.random() > fill_prob:
            order = Order(
                order_id=generate_order_id("paper"),
                market_condition_id=market_condition_id,
                token_id=token_id,
                side=side,
                price=price,
                size=size,
                status=OrderStatus.CANCELLED,
                filled_size=0.0,
                strategy=strategy,
            )
            self.simulated_orders.append(order)
            return TradeResult(
                order=order,
                success=False,
                error="no_fill_simulated",
            )

        # ── Partial fill model ───────────────────────────────────
        if impact_ratio > 0.08:
            fill_ratio = max(0.25, 1.0 - (impact_ratio - 0.08) * 3.5)
            fill_size = size * fill_ratio
        else:
            fill_size = size

        order = Order(
            order_id=generate_order_id("paper"),
            market_condition_id=market_condition_id,
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            status=OrderStatus.FILLED if fill_size == size else OrderStatus.PARTIAL,
            filled_size=fill_size,
            strategy=strategy,
        )
        self.simulated_orders.append(order)

        fees = fill_size * fill_price * 0.02

        result = TradeResult(
            order=order,
            success=True,
            fill_price=fill_price,
            fill_size=fill_size,
            fees=fees,
        )
        self.simulated_fills.append(result)
        self._fill_successes += 1

        logger.info(
            "paper_trade_executed",
            order_id=order.order_id,
            side=side.value,
            price=round(fill_price, 4),
            size=round(fill_size, 2),
            slippage=round(total_slippage, 4),
        )
        return result

    def cancel_order(self, order_id: str) -> bool:
        for o in self.simulated_orders:
            if o.order_id == order_id and o.status == OrderStatus.OPEN:
                o.status = OrderStatus.CANCELLED
                return True
        return False

    def cancel_all_orders(self) -> bool:
        for o in self.simulated_orders:
            if o.status == OrderStatus.OPEN:
                o.status = OrderStatus.CANCELLED
        return True
