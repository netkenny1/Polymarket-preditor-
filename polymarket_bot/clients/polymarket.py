"""Polymarket CLOB API client for market data and order management."""

from __future__ import annotations

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

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["POLY_API_KEY"] = self.config.api_key
            headers["POLY_API_SECRET"] = self.config.api_secret
            headers["POLY_PASSPHRASE"] = self.config.api_passphrase
        return headers

    def close(self) -> None:
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
        """Get midpoint prices for multiple tokens."""
        prices = {}
        for tid in token_ids:
            mid = self.get_midpoint(tid)
            if mid is not None:
                prices[tid] = mid
        return prices

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
        """Place a limit order on the CLOB.

        In paper trading mode, simulates the order fill.
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
                "type": "GTC",  # Good till cancelled
            }
            resp = self._clob_client.post("/order", json=payload)
            resp.raise_for_status()
            data = resp.json()

            order.order_id = data.get("orderID", order.order_id)
            order.status = OrderStatus.OPEN

            return TradeResult(
                order=order,
                success=True,
                fill_price=price,
                fill_size=size,
            )
        except httpx.HTTPError as e:
            logger.error("order_failed", error=str(e))
            order.status = OrderStatus.FAILED
            return TradeResult(order=order, success=False, error=str(e))

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        try:
            resp = self._clob_client.delete(f"/order/{order_id}")
            resp.raise_for_status()
            return True
        except httpx.HTTPError as e:
            logger.error("cancel_failed", order_id=order_id, error=str(e))
            return False

    def cancel_all_orders(self) -> bool:
        """Cancel all open orders."""
        try:
            resp = self._clob_client.delete("/orders")
            resp.raise_for_status()
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
        """Simulate order execution with realistic slippage model.

        Slippage model accounts for:
        1. Base spread cost (~0.5% per side)
        2. Market impact: larger orders move the price more
        3. Partial fills: very large orders may not fully fill
        """
        from polymarket_bot.utils.helpers import generate_order_id

        # ── Slippage model ───────────────────────────────────────
        # Base slippage: half the typical spread
        base_slippage = 0.005

        # Market impact: increases with order size relative to market volume
        market_vol = self._simulated_volumes.get(token_id, 5000.0)
        order_value = size * price
        impact_ratio = order_value / max(market_vol, 100.0)
        market_impact = impact_ratio * 0.02  # 2% impact per 100% of daily volume

        total_slippage = base_slippage + market_impact

        if side == Side.BUY:
            fill_price = price * (1 + total_slippage)
        else:
            fill_price = price * (1 - total_slippage)

        # Clamp fill price to valid range
        fill_price = max(0.01, min(0.99, fill_price))

        # ── Partial fill model ───────────────────────────────────
        # Very large orders relative to market may partially fill
        if impact_ratio > 0.3:
            fill_ratio = max(0.5, 1.0 - (impact_ratio - 0.3))
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

        # Polymarket charges ~1% taker fee on fill
        fees = fill_size * fill_price * 0.002

        result = TradeResult(
            order=order,
            success=True,
            fill_price=fill_price,
            fill_size=fill_size,
            fees=fees,
        )
        self.simulated_fills.append(result)

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
