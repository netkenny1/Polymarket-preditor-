"""Live / paper-trading bot for the BTC 5-minute strategy.

High-level loop:

    1. Connect the Binance bookTicker WS and let it populate a rolling
       price tracker in the background.
    2. Periodically (every 500ms) look up the current `btc-updown-5m-{ts}`
       market via Polymarket Gamma API. Cache the condition_id and token
       ids until the window rolls over at the next 300-second boundary.
    3. Fetch the current order book for the UP / DOWN tokens.
    4. Pull trailing 1-minute BTC closes from the Binance historical REST
       endpoint (cached every ~30s to avoid rate limits).
    5. Build a `BTC5MinContext` and call `BTC5MinStrategy.generate_signals`.
    6. For each signal: run risk checks, compute a maker-only limit price
       (inside the book), and submit a POST_ONLY order via the CLOB
       client. Track the position; at window expiry it resolves on-chain
       via Chainlink and the winning tokens redeem to USDC.

The `LiveBot` class is deliberately small — it glues the pieces together
and delegates all interesting logic to the pricing, strategy, and risk
modules. One place to debug.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx
import structlog

from polymarket_bot.clients.binance_feed import (
    BinanceHistoricalClient,
    BinancePriceTracker,
    BinanceWebsocketFeed,
)
from polymarket_bot.clients.polymarket import PaperTradingClient, PolymarketClient
from polymarket_bot.config import BotConfig
from polymarket_bot.data.models import (
    Market,
    OrderBook,
    OrderBookLevel,
    Side,
    Token,
)
from polymarket_bot.risk.manager import RiskManager
from polymarket_bot.risk.portfolio import Portfolio
from polymarket_bot.risk.position_sizer import PositionSizer
from polymarket_bot.strategies.btc_5min import BTC5MinContext, BTC5MinStrategy

logger = structlog.get_logger()


@dataclass
class _MarketCache:
    """In-memory cache of the current 5-min market so we don't hammer Gamma."""

    window_start_ts: int
    market: Market


class LiveBot:
    """Glue the live pieces together and trade the BTC 5-minute market."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.tracker = BinancePriceTracker(window_seconds=600.0)
        self.binance_ws = BinanceWebsocketFeed(
            ws_url=config.binance.ws_url,
            stream=config.binance.ws_stream,
            tracker=self.tracker,
        )
        self.hist = BinanceHistoricalClient()

        # Polymarket client — paper by default, live if PAPER_TRADING=false.
        if config.trading.paper_trading:
            self.client: PolymarketClient = PaperTradingClient(config.polymarket)
        else:
            self.client = PolymarketClient(config.polymarket)

        self.portfolio = Portfolio(initial_cash=config.trading.initial_capital_usd)
        self.sizer = PositionSizer(config.trading, config.risk)
        self.risk = RiskManager(config.risk, config.trading, self.portfolio, self.sizer)
        self.strategy = BTC5MinStrategy(config.strategy)

        self._market_cache: Optional[_MarketCache] = None
        self._closes_cache: list[float] = []
        self._closes_cache_ts: float = 0.0
        self._open_position_window: Optional[int] = None
        self._running = False
        self._gamma = httpx.Client(base_url=config.polymarket.gamma_url, timeout=15.0)

    # ── Lifecycle ─────────────────────────────────────────────────

    async def run(self) -> None:
        """Main async loop. Runs until Ctrl-C or `stop()`."""
        self._running = True
        logger.info(
            "bot_starting",
            paper=self.config.trading.paper_trading,
            capital=self.portfolio.cash,
        )

        ws_task = asyncio.create_task(self.binance_ws.connect())
        try:
            # Warm up: wait until the Binance tracker has a price.
            warm_start = time.monotonic()
            while self.tracker.current_price <= 0 and time.monotonic() - warm_start < 10:
                await asyncio.sleep(0.2)
            if self.tracker.current_price <= 0:
                logger.error("bot_no_binance_price_after_warmup")
                return

            while self._running:
                try:
                    await self._tick()
                except Exception as e:  # pragma: no cover
                    logger.exception("tick_error", error=str(e))
                await asyncio.sleep(1.0)
        finally:
            self._running = False
            ws_task.cancel()
            try:
                await ws_task
            except asyncio.CancelledError:
                pass
            self.client.close()
            self.hist.close()
            self._gamma.close()

    def stop(self) -> None:
        self._running = False

    # ── Per-tick pipeline ─────────────────────────────────────────

    async def _tick(self) -> None:
        now_ts = int(time.time())
        window_start = (now_ts // self.config.strategy.window_seconds) * self.config.strategy.window_seconds

        # Rollover: if the window changed, clear per-window state.
        if self._market_cache and self._market_cache.window_start_ts != window_start:
            self._market_cache = None
            self._open_position_window = None

        market = await self._get_or_fetch_market(window_start)
        if market is None:
            return
        if not market.active:
            return

        books = await self._fetch_books(market)
        if not books:
            return

        closes = self._get_trailing_closes()
        if len(closes) < 10:
            return

        ctx = BTC5MinContext(
            btc_spot=self.tracker.current_price,
            btc_closes_1m=closes,
            now_ts=now_ts,
            bankroll_usd=self.portfolio.total_value,
        )
        signals = self.strategy.generate_signals([market], books, ctx)
        if not signals:
            return

        # Only open one position per window.
        if self._open_position_window == window_start:
            return

        for signal in signals:
            approved, size_usd, reason = self.risk.check_signal(signal)
            if not approved:
                logger.info("signal_rejected", reason=reason, edge=round(signal.edge, 4))
                continue

            limit_price = self._maker_price(signal, books)
            if limit_price is None:
                continue
            shares = round(size_usd / limit_price, 2)
            if shares <= 0:
                continue

            logger.info(
                "placing_order",
                window=window_start,
                side=signal.side.value,
                outcome=signal.outcome,
                fair=round(signal.estimated_fair_value, 4),
                market_price=round(signal.market_price, 4),
                limit=limit_price,
                shares=shares,
                edge=round(signal.edge, 4),
                post_only=self.config.strategy.post_only,
            )
            result = self.client.place_order(
                token_id=signal.token_id,
                side=signal.side,
                price=limit_price,
                size=shares,
                market_condition_id=signal.market_condition_id,
                strategy=signal.strategy,
            )
            if result.success:
                self.portfolio.process_fill(result)
                self._open_position_window = window_start
            break

    def _maker_price(self, signal, books: dict[str, OrderBook]) -> Optional[float]:
        """Compute a maker-only limit price (sit one tick inside the book)."""
        book = books.get(signal.token_id)
        if book is None or book.best_bid is None or book.best_ask is None:
            return None
        tick = 0.01
        if signal.side == Side.BUY:
            # Join the bid — one tick above best bid, below best ask so
            # the order rests in the book instead of crossing.
            price = min(book.best_bid + tick, book.best_ask - tick)
        else:
            price = max(book.best_ask - tick, book.best_bid + tick)
        return max(0.01, min(0.99, round(price, 2)))

    # ── Market / data fetchers ────────────────────────────────────

    async def _get_or_fetch_market(self, window_start_ts: int) -> Optional[Market]:
        if self._market_cache and self._market_cache.window_start_ts == window_start_ts:
            return self._market_cache.market

        slug = self.config.strategy.slug_template.format(ts=window_start_ts)
        try:
            resp = await asyncio.to_thread(
                self._gamma.get,
                "/markets",
                params={"slug": slug},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:  # pragma: no cover - network
            logger.warning("gamma_fetch_failed", slug=slug, error=str(e))
            return None

        if not data:
            return None
        raw = data[0] if isinstance(data, list) else data

        market = self._parse_market(raw, window_start_ts)
        if market is None:
            return None
        self._market_cache = _MarketCache(window_start_ts=window_start_ts, market=market)
        return market

    def _parse_market(self, raw: dict, window_start_ts: int) -> Optional[Market]:
        # Extract token ids (Gamma returns clobTokenIds as a JSON string or list).
        clob_raw = raw.get("clobTokenIds") or raw.get("clob_token_ids") or []
        if isinstance(clob_raw, str):
            import json
            try:
                clob_raw = json.loads(clob_raw)
            except Exception:
                clob_raw = []
        if not clob_raw or len(clob_raw) < 2:
            return None

        # Outcome order: Polymarket convention is index 0 = "Up", 1 = "Down".
        tokens = [
            Token(token_id=str(clob_raw[0]), outcome="Up", price=0.5),
            Token(token_id=str(clob_raw[1]), outcome="Down", price=0.5),
        ]

        # Strike: Polymarket publishes the window-start BTC reference price
        # in the market metadata. Falls back to current Binance mid.
        strike = float(
            raw.get("startPrice")
            or raw.get("reference_price")
            or self.tracker.current_price
        )

        return Market(
            condition_id=str(raw.get("condition_id") or raw.get("conditionId") or ""),
            slug=str(raw.get("slug") or ""),
            question=str(raw.get("question") or ""),
            tokens=tokens,
            start_ts=window_start_ts,
            end_ts=window_start_ts + 300,
            strike_price=strike,
            active=bool(raw.get("active", True)),
            liquidity=float(raw.get("liquidityNum") or raw.get("liquidity_num") or 0),
        )

    async def _fetch_books(self, market: Market) -> dict[str, OrderBook]:
        books: dict[str, OrderBook] = {}
        for tok in market.tokens:
            try:
                book = await asyncio.to_thread(self.client.get_order_book, tok.token_id)
                books[tok.token_id] = book
            except Exception as e:  # pragma: no cover - network
                logger.warning("book_fetch_failed", token=tok.token_id, error=str(e))
        return books

    def _get_trailing_closes(self) -> list[float]:
        """Cached 1-minute BTC closes for vol estimation (refresh every 30s)."""
        now = time.time()
        if (now - self._closes_cache_ts) < 30 and self._closes_cache:
            return self._closes_cache
        try:
            minutes = max(self.config.strategy.vol_window_minutes + 5, 30)
            end_ms = int(now * 1000)
            start_ms = end_ms - minutes * 60_000
            klines = self.hist.fetch_klines(start_ms, end_ms)
            closes = [k.close for k in klines]
            if closes:
                self._closes_cache = closes
                self._closes_cache_ts = now
        except Exception as e:  # pragma: no cover - network
            logger.warning("closes_fetch_failed", error=str(e))
        return self._closes_cache
