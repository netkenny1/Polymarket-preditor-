"""Autonomous market discovery engine for Polymarket.

Scans, filters, ranks, and categorizes markets to find the best
trading opportunities across all active Polymarket prediction markets.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import structlog

from polymarket_bot.clients.polymarket import PolymarketClient
from polymarket_bot.config import TradingConfig
from polymarket_bot.data.models import Market, MarketCategory, OrderBook

logger = structlog.get_logger()

# ── Keyword sets for market discovery ────────────────────────────────

_BTC_DAILY_KEYWORDS = re.compile(
    r"bitcoin|btc",
    re.IGNORECASE,
)
_BTC_DAILY_CONTEXT = re.compile(
    r"close\s+above|close\s+below|open|daily|end\s+of\s+day|24h",
    re.IGNORECASE,
)

_TRUMP_KEYWORDS = re.compile(
    r"trump|president\s+trump|executive\s+order|tariff|tweet|truth\s+social",
    re.IGNORECASE,
)

_CRYPTO_KEYWORDS = re.compile(
    r"bitcoin|btc|ethereum|eth|solana|sol|crypto|defi|token|altcoin"
    r"|cardano|ada|polygon|matic|avalanche|avax|dogecoin|doge|ripple|xrp",
    re.IGNORECASE,
)

# Maximum number of markets to fetch per page from the Gamma API.
_PAGE_SIZE = 100

# Default maximum pages to scan (guard against runaway pagination).
_MAX_PAGES = 20


@dataclass
class RankedMarket:
    """A market annotated with opportunity scoring."""

    market: Market
    opportunity_score: float = 0.0
    liquidity_score: float = 0.0
    spread_score: float = 0.0
    volume_score: float = 0.0
    category: MarketCategory = MarketCategory.OTHER
    time_to_resolution_hours: float = 0.0


class MarketScanner:
    """Automatically discovers and ranks tradeable markets on Polymarket.

    Connects to the Gamma and CLOB APIs via ``PolymarketClient`` to fetch
    active markets, then applies configurable filters and a multi-factor
    ranking model to surface the best opportunities.
    """

    def __init__(
        self,
        client: PolymarketClient,
        config: TradingConfig,
        *,
        max_pages: int = _MAX_PAGES,
    ) -> None:
        self._client = client
        self._config = config
        self._max_pages = max_pages

    # ── Scanning ─────────────────────────────────────────────────

    async def scan_all_markets(self) -> list[Market]:
        """Fetch all active markets from Polymarket.

        Paginates through the Gamma API until no more results are returned
        or ``max_pages`` is reached.  The underlying HTTP calls are
        synchronous (``httpx.Client``), so each page fetch is offloaded to
        a thread to keep the event loop responsive.
        """
        all_markets: list[Market] = []
        offset = 0
        loop = asyncio.get_running_loop()

        for page in range(self._max_pages):
            logger.debug("scanning_page", page=page, offset=offset)
            try:
                batch = await loop.run_in_executor(
                    None,
                    lambda o=offset: self._client.get_markets(
                        limit=_PAGE_SIZE,
                        offset=o,
                        active=True,
                        closed=False,
                    ),
                )
            except Exception:
                logger.exception("scan_page_failed", page=page, offset=offset)
                break

            if not batch:
                break

            all_markets.extend(batch)
            offset += _PAGE_SIZE

            logger.debug(
                "scan_page_complete",
                page=page,
                batch_size=len(batch),
                total=len(all_markets),
            )

            # If we got fewer than a full page, there are no more results.
            if len(batch) < _PAGE_SIZE:
                break

        logger.info("scan_complete", total_markets=len(all_markets))
        return all_markets

    # ── Filtering ────────────────────────────────────────────────

    def filter_tradeable(self, markets: list[Market]) -> list[Market]:
        """Filter to markets worth trading.

        A market is considered tradeable when it satisfies *all* of:
        - ``active`` flag is ``True``
        - Liquidity meets the configured minimum (``min_liquidity_usd``)
        - Spread is at or below the configured maximum (``max_spread``)
        - At least one outcome token exists
        """
        tradeable: list[Market] = []
        min_liq = self._config.min_liquidity_usd
        max_spread = self._config.max_spread

        for m in markets:
            if not m.active:
                continue
            if not m.tokens:
                continue
            if m.liquidity < min_liq:
                continue
            if m.spread > max_spread:
                continue
            tradeable.append(m)

        logger.info(
            "filter_tradeable",
            input_count=len(markets),
            output_count=len(tradeable),
            min_liquidity=min_liq,
            max_spread=max_spread,
        )
        return tradeable

    # ── Ranking ──────────────────────────────────────────────────

    def rank_markets(
        self,
        markets: list[Market],
        order_books: dict[str, OrderBook],
    ) -> list[RankedMarket]:
        """Rank markets by a composite opportunity score.

        The score combines six normalised factors:
        1. **Liquidity depth** -- deeper order books are preferred.
        2. **Spread tightness** -- narrower spread means better execution.
        3. **Volume** -- higher 24 h volume signals better price discovery.
        4. **Time to resolution** -- sweet spot is 1--30 days.
        5. **Category diversity** -- we reward under-represented categories
           so the result set is balanced.
        6. **Price range** -- prices between 0.15 and 0.85 offer the best
           risk/reward.

        Parameters
        ----------
        markets:
            Pre-filtered list of tradeable markets.
        order_books:
            Mapping of ``token_id`` -> ``OrderBook``.  Not every market
            needs an entry; missing books receive a zero depth score.

        Returns
        -------
        list[RankedMarket]
            Markets sorted descending by ``opportunity_score``.
        """
        if not markets:
            return []

        # ---- Compute raw scores ----
        now = datetime.now(timezone.utc)

        # Collect category counts for diversity weighting.
        category_counts: dict[MarketCategory, int] = {}
        for m in markets:
            category_counts[m.category] = category_counts.get(m.category, 0) + 1
        total_markets = len(markets)

        # Determine normalisation ceilings from the data.
        max_volume = max((m.volume_24h for m in markets), default=1.0) or 1.0
        max_liquidity = max((m.liquidity for m in markets), default=1.0) or 1.0

        ranked: list[RankedMarket] = []
        for m in markets:
            ttr_hours = self._time_to_resolution_hours(m, now)

            # 1. Liquidity score (0-1): normalised against the deepest market.
            #    If an order book is available, use total depth; otherwise
            #    fall back to the liquidity figure from the Gamma API.
            book_depth = 0.0
            for t in m.tokens:
                ob = order_books.get(t.token_id)
                if ob is not None:
                    book_depth += ob.bid_depth + ob.ask_depth
            raw_liquidity = book_depth if book_depth > 0 else m.liquidity
            liquidity_score = min(raw_liquidity / max_liquidity, 1.0)

            # 2. Spread score (0-1): tighter is better.
            #    Use the best spread from available order books, else the
            #    token-price implied spread on the Market object.
            best_spread: Optional[float] = None
            for t in m.tokens:
                ob = order_books.get(t.token_id)
                if ob is not None and ob.spread is not None:
                    if best_spread is None or ob.spread < best_spread:
                        best_spread = ob.spread
            effective_spread = best_spread if best_spread is not None else m.spread
            # A spread of 0 is perfect (score 1); spread >= 0.20 gives 0.
            spread_score = max(0.0, 1.0 - effective_spread / 0.20)

            # 3. Volume score (0-1): normalised against highest-volume market.
            volume_score = min(m.volume_24h / max_volume, 1.0)

            # 4. Time-to-resolution score (0-1): peak at 3-14 days.
            time_score = self._time_score(ttr_hours)

            # 5. Category diversity bonus (0-1).
            #    Under-represented categories get a higher score.
            cat_freq = category_counts.get(m.category, 1) / total_markets
            diversity_score = 1.0 - cat_freq

            # 6. Price-range score (0-1): prefer 0.15-0.85.
            price_score = self._price_range_score(m)

            # ---- Weighted composite ----
            opportunity = (
                0.25 * liquidity_score
                + 0.20 * spread_score
                + 0.20 * volume_score
                + 0.15 * time_score
                + 0.10 * diversity_score
                + 0.10 * price_score
            )

            ranked.append(
                RankedMarket(
                    market=m,
                    opportunity_score=round(opportunity, 4),
                    liquidity_score=round(liquidity_score, 4),
                    spread_score=round(spread_score, 4),
                    volume_score=round(volume_score, 4),
                    category=m.category,
                    time_to_resolution_hours=round(ttr_hours, 1),
                )
            )

        ranked.sort(key=lambda r: r.opportunity_score, reverse=True)

        logger.info(
            "rank_complete",
            total=len(ranked),
            top_score=ranked[0].opportunity_score if ranked else 0,
        )
        return ranked

    # ── Targeted finders ─────────────────────────────────────────

    def find_btc_daily_markets(self, markets: list[Market]) -> list[Market]:
        """Find BTC daily open/close direction markets.

        Matches markets whose question mentions Bitcoin/BTC **and**
        contains resolution-oriented language such as *close above*,
        *close below*, *open*, or *daily*.
        """
        results: list[Market] = []
        for m in markets:
            text = f"{m.question} {m.description}"
            if _BTC_DAILY_KEYWORDS.search(text) and _BTC_DAILY_CONTEXT.search(text):
                results.append(m)

        logger.info("find_btc_daily", found=len(results))
        return results

    def find_trump_markets(self, markets: list[Market]) -> list[Market]:
        """Find Trump-related markets (tweets, executive orders, policy, tariffs)."""
        results = [
            m
            for m in markets
            if _TRUMP_KEYWORDS.search(f"{m.question} {m.description}")
        ]
        logger.info("find_trump_markets", found=len(results))
        return results

    def find_crypto_markets(self, markets: list[Market]) -> list[Market]:
        """Find all crypto-related markets (BTC, ETH, SOL, etc.)."""
        results = [
            m
            for m in markets
            if _CRYPTO_KEYWORDS.search(f"{m.question} {m.description}")
            or m.category == MarketCategory.CRYPTO
        ]
        logger.info("find_crypto_markets", found=len(results))
        return results

    # ── Timeframe categorisation ─────────────────────────────────

    def categorize_by_timeframe(
        self, markets: list[Market]
    ) -> dict[str, list[Market]]:
        """Group markets by resolution timeframe.

        Buckets:
        - ``today``      -- resolves within 24 hours
        - ``this_week``  -- resolves in 1--7 days
        - ``this_month`` -- resolves in 7--30 days
        - ``long_term``  -- resolves in 30+ days (or unknown end date)
        """
        buckets: dict[str, list[Market]] = {
            "today": [],
            "this_week": [],
            "this_month": [],
            "long_term": [],
        }

        now = datetime.now(timezone.utc)

        for m in markets:
            hours = self._time_to_resolution_hours(m, now)
            if hours <= 0 or m.end_date is None:
                buckets["long_term"].append(m)
            elif hours <= 24:
                buckets["today"].append(m)
            elif hours <= 24 * 7:
                buckets["this_week"].append(m)
            elif hours <= 24 * 30:
                buckets["this_month"].append(m)
            else:
                buckets["long_term"].append(m)

        logger.info(
            "categorize_by_timeframe",
            today=len(buckets["today"]),
            this_week=len(buckets["this_week"]),
            this_month=len(buckets["this_month"]),
            long_term=len(buckets["long_term"]),
        )
        return buckets

    # ── Private helpers ──────────────────────────────────────────

    @staticmethod
    def _time_to_resolution_hours(m: Market, now: datetime) -> float:
        """Return hours until market end date, or 0 if unknown/past."""
        if m.end_date is None:
            return 0.0
        end = m.end_date if m.end_date.tzinfo else m.end_date.replace(tzinfo=timezone.utc)
        delta = (end - now).total_seconds() / 3600.0
        return max(delta, 0.0)

    @staticmethod
    def _time_score(hours: float) -> float:
        """Score time-to-resolution with a sweet spot of 1--30 days (24--720 h).

        Returns a value between 0 and 1:
        - 0 h (unknown / already past): 0.1  (small residual)
        - 24--720 h (1--30 days):       peaks at 1.0 around 3--14 days
        - > 720 h:                       decays toward 0.2
        """
        if hours <= 0:
            return 0.1
        if hours < 24:
            # Resolving very soon -- moderate value.
            return 0.4 + 0.6 * (hours / 24.0)
        if hours <= 336:  # up to 14 days
            return 1.0
        if hours <= 720:  # 14--30 days
            # Linear decay from 1.0 to 0.7
            return 1.0 - 0.3 * ((hours - 336) / (720 - 336))
        # Long-term: slow decay, floor at 0.2
        return max(0.2, 0.7 - 0.5 * ((hours - 720) / 720))

    @staticmethod
    def _price_range_score(m: Market) -> float:
        """Score based on yes-token price proximity to the 0.15--0.85 band.

        Prices deep inside the band score 1.0.  Prices near the extremes
        (< 0.10 or > 0.90) score 0.
        """
        p = m.yes_price
        if 0.15 <= p <= 0.85:
            return 1.0
        if p < 0.15:
            return max(0.0, p / 0.15)
        # p > 0.85
        return max(0.0, (1.0 - p) / 0.15)
