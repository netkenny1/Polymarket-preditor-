"""Alpha Vantage client for equity/ETF price changes and market sentiment.

Fetches daily price data for key market ETFs via Alpha Vantage and supplements
it with the free Alternative.me Fear & Greed Index. Significant moves are
converted to NarrativeEvents. Degrades gracefully to fear/greed-only (no AV
key required for that endpoint) or full empty list when appropriate.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import aiohttp
import structlog

from polymarket_bot.data.models import NarrativeCategory, NarrativeEvent

logger = structlog.get_logger()

AV_BASE = "https://www.alphavantage.co/query"
FNG_URL  = "https://api.alternative.me/fng/?limit=1"

# symbol → (human_name, NarrativeCategory, threshold_pct)
TRACKED_SYMBOLS: dict[str, tuple[str, NarrativeCategory, float]] = {
    "SPY": ("S&P 500",             NarrativeCategory.MARKET_CRISIS, 1.5),
    "GLD": ("Gold ETF",            NarrativeCategory.GEOPOLITICAL,  1.0),
    "USO": ("Oil ETF",             NarrativeCategory.GEOPOLITICAL,  2.0),
    "XLE": ("Energy Sector ETF",   NarrativeCategory.GEOPOLITICAL,  2.0),
    "XLF": ("Financial Sector ETF",NarrativeCategory.MARKET_CRISIS, 2.0),
}


def _symbol_sentiment(symbol: str, change_pct: float) -> float:
    """Return a sentiment score for a given symbol and direction of move.

    Business logic:
      - SPY down → risk-off / MARKET_CRISIS bearish signal.
      - GLD up   → flight-to-safety / geopolitical fear hedge → bearish overall.
      - USO/XLE up → energy cost increase → mild bearish macro.
      - XLF down → financial stress → bearish.
      - Positive moves default to mild bullish.
    """
    rising = change_pct > 0

    if symbol == "SPY":
        return -0.5 if not rising else 0.3

    if symbol == "GLD":
        # Gold rising means investors fear something → negative for risk
        return -0.3 if rising else 0.2

    if symbol in ("USO", "XLE"):
        return -0.25 if rising else 0.15

    if symbol == "XLF":
        return -0.4 if not rising else 0.25

    return -0.2 if not rising else 0.2


def _symbol_content(symbol: str, name: str, change_pct: float) -> str:
    direction = "rallied" if change_pct > 0 else "fell"
    return (
        f"{name} ({symbol}) {direction} {abs(change_pct):.2f}% in today's session, "
        f"signaling {'positive' if change_pct > 0 else 'negative'} market momentum."
    )


class AlphaVantageClient:
    """Client for Alpha Vantage daily equity data and Alternative.me Fear/Greed.

    The Fear & Greed fetch requires no API key and always runs.
    ETF price fetches require an Alpha Vantage key; without one the method
    returns only fear/greed-derived events.

    Usage::

        client = AlphaVantageClient(api_key=os.environ.get("AV_API_KEY", ""))
        events = await client.get_market_snapshot()
    """

    def __init__(self, api_key: str, cache_ttl_minutes: int = 15) -> None:
        self._api_key = api_key
        self._cache_ttl_seconds = cache_ttl_minutes * 60
        # (monotonic_timestamp, events)
        self._cache: tuple[float, list[NarrativeEvent]] | None = None

    # ------------------------------------------------------------------
    # Low-level fetches
    # ------------------------------------------------------------------

    async def fetch_symbol_change(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
    ) -> float | None:
        """Return the day-over-day percentage change in closing price for *symbol*.

        Uses the Alpha Vantage ``TIME_SERIES_DAILY_ADJUSTED`` function.
        Returns ``None`` on any error or missing data.
        """
        if not self._api_key:
            return None

        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "apikey": self._api_key,
            "outputsize": "compact",
        }

        try:
            async with session.get(
                AV_BASE,
                params=params,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                if resp.status != 200:
                    logger.warning(
                        "av_http_error",
                        symbol=symbol,
                        status=resp.status,
                    )
                    return None

                data = await resp.json(content_type=None)

        except aiohttp.ClientError as exc:
            logger.warning("av_request_error", symbol=symbol, error=str(exc))
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("av_unexpected_error", symbol=symbol, error=str(exc))
            return None

        # Check for API limit / error messages returned inside a 200 response
        if "Note" in data or "Information" in data:
            msg = data.get("Note") or data.get("Information", "")
            logger.warning("av_api_limit_or_info", symbol=symbol, message=msg[:120])
            return None

        ts_key = "Time Series (Daily)"
        time_series: dict[str, Any] = data.get(ts_key, {})
        if not time_series:
            logger.warning("av_no_time_series", symbol=symbol)
            return None

        sorted_dates = sorted(time_series.keys(), reverse=True)
        if len(sorted_dates) < 2:
            logger.warning("av_insufficient_data", symbol=symbol, dates=len(sorted_dates))
            return None

        try:
            today_close     = float(time_series[sorted_dates[0]]["4. close"])
            yesterday_close = float(time_series[sorted_dates[1]]["4. close"])
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("av_parse_error", symbol=symbol, error=str(exc))
            return None

        if yesterday_close == 0.0:
            return None

        change_pct = (today_close - yesterday_close) / yesterday_close * 100.0
        logger.debug(
            "av_symbol_fetched",
            symbol=symbol,
            today=today_close,
            yesterday=yesterday_close,
            change_pct=round(change_pct, 4),
        )
        return change_pct

    async def fetch_fear_greed(self, session: aiohttp.ClientSession) -> int | None:
        """Fetch the current Fear & Greed Index from Alternative.me (no key needed).

        Returns an integer in [0, 100]: 0 = extreme fear, 100 = extreme greed.
        Returns ``None`` on any error.
        """
        try:
            async with session.get(
                FNG_URL,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    logger.warning("fng_http_error", status=resp.status)
                    return None

                data = await resp.json(content_type=None)
                entries: list[dict[str, Any]] = data.get("data", [])
                if not entries:
                    return None

                raw_value = entries[0].get("value")
                fng = int(raw_value)
                logger.debug("fng_fetched", value=fng)
                return fng

        except aiohttp.ClientError as exc:
            logger.warning("fng_request_error", error=str(exc))
            return None
        except (ValueError, TypeError, KeyError) as exc:
            logger.warning("fng_parse_error", error=str(exc))
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("fng_unexpected_error", error=str(exc))
            return None

    # ------------------------------------------------------------------
    # High-level snapshot
    # ------------------------------------------------------------------

    async def get_market_snapshot(self) -> list[NarrativeEvent]:
        """Fetch all tracked symbols and Fear/Greed Index concurrently.

        Produces NarrativeEvents for:
        - Any ETF whose absolute day-over-day change exceeds its threshold.
        - Fear/Greed Index < 25 (extreme fear) → MARKET_CRISIS bearish event.
        - Fear/Greed Index > 75 (greed / complacency) → mild positive event.

        Results are cached for ``cache_ttl_minutes`` minutes (set at init).
        Fear & Greed always runs even without an AV API key.
        """
        now = time.monotonic()
        if self._cache is not None:
            cached_at, cached_events = self._cache
            if now - cached_at < self._cache_ttl_seconds:
                logger.debug(
                    "av_cache_hit",
                    age_seconds=round(now - cached_at, 1),
                    events=len(cached_events),
                )
                return cached_events

        events: list[NarrativeEvent] = []

        async with aiohttp.ClientSession() as session:
            # Fear/Greed always runs; symbol tasks only if we have a key
            fng_task = asyncio.create_task(self.fetch_fear_greed(session))

            symbol_tasks: dict[str, asyncio.Task[float | None]] = {}
            if self._api_key:
                symbol_tasks = {
                    symbol: asyncio.create_task(
                        self.fetch_symbol_change(session, symbol)
                    )
                    for symbol in TRACKED_SYMBOLS
                }
            else:
                logger.debug(
                    "av_no_api_key",
                    msg="No Alpha Vantage key; skipping ETF price fetch",
                )

            # Await all concurrently
            fng_value = await fng_task
            symbol_results: dict[str, float | None] = {}
            for sym, task in symbol_tasks.items():
                try:
                    symbol_results[sym] = await task
                except Exception as exc:  # noqa: BLE001
                    logger.warning("av_symbol_task_error", symbol=sym, error=str(exc))
                    symbol_results[sym] = None

        # ── ETF events ───────────────────────────────────────────────
        for symbol, change_pct in symbol_results.items():
            if change_pct is None:
                continue

            name, category, threshold = TRACKED_SYMBOLS[symbol]
            if abs(change_pct) <= threshold:
                continue

            sentiment = _symbol_sentiment(symbol, change_pct)
            magnitude = min(abs(change_pct) / 5.0, 1.0)
            content   = _symbol_content(symbol, name, change_pct)

            event = NarrativeEvent(
                event_id=str(uuid.uuid4()),
                source="alpha_vantage",
                content=content,
                timestamp=datetime.now(timezone.utc),
                category=category,
                sentiment=sentiment,
                magnitude=magnitude,
                keywords=[symbol.lower(), name.lower().split()[0], "etf", "market"],
                metadata={
                    "symbol": symbol,
                    "name": name,
                    "change_pct": round(change_pct, 4),
                    "threshold": threshold,
                },
            )
            events.append(event)
            logger.info(
                "av_narrative_event",
                symbol=symbol,
                change_pct=round(change_pct, 4),
                sentiment=sentiment,
            )

        # ── Fear & Greed events ──────────────────────────────────────
        if fng_value is not None:
            if fng_value < 25:
                # Extreme fear: strong bearish macro signal
                content = (
                    f"Fear & Greed Index at {fng_value} (extreme fear). "
                    "Investors are panic-selling; risk assets under pressure."
                )
                events.append(NarrativeEvent(
                    event_id=str(uuid.uuid4()),
                    source="alpha_vantage",
                    content=content,
                    timestamp=datetime.now(timezone.utc),
                    category=NarrativeCategory.MARKET_CRISIS,
                    sentiment=-0.6,
                    magnitude=min((25 - fng_value) / 25.0, 1.0),
                    keywords=["fear", "greed", "index", "extreme", "panic"],
                    metadata={"fear_greed_index": fng_value},
                ))
                logger.info("fng_extreme_fear", value=fng_value)

            elif fng_value > 75:
                # Greed / complacency: light positive but also a contrarian flag
                content = (
                    f"Fear & Greed Index at {fng_value} (greed). "
                    "Market sentiment is bullish; watch for potential complacency."
                )
                events.append(NarrativeEvent(
                    event_id=str(uuid.uuid4()),
                    source="alpha_vantage",
                    content=content,
                    timestamp=datetime.now(timezone.utc),
                    category=NarrativeCategory.MARKET_CRISIS,
                    sentiment=0.25,
                    magnitude=min((fng_value - 75) / 25.0, 1.0),
                    keywords=["fear", "greed", "index", "bullish", "sentiment"],
                    metadata={"fear_greed_index": fng_value},
                ))
                logger.info("fng_greed", value=fng_value)

        self._cache = (now, events)
        logger.info("av_snapshot_complete", total_events=len(events))
        return events
