"""FRED (Federal Reserve Economic Data) client for real-time macro indicators.

Fetches key economic series from the St. Louis Fed API and converts significant
changes into NarrativeEvents for the narrative analysis pipeline. Gracefully
degrades to an empty list when no API key is configured.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

import aiohttp
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from polymarket_bot.data.models import NarrativeCategory, NarrativeEvent

logger = structlog.get_logger()

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# series_id → (human_name, NarrativeCategory, change_threshold_pct)
TRACKED_SERIES: dict[str, tuple[str, NarrativeCategory, float]] = {
    "CPIAUCSL":  ("CPI All Items",       NarrativeCategory.MONETARY_POLICY, 0.1),
    "DFF":       ("Fed Funds Rate",      NarrativeCategory.MONETARY_POLICY, 0.1),
    "UNRATE":    ("Unemployment Rate",   NarrativeCategory.MONETARY_POLICY, 0.2),
    "DGS10":     ("10Y Treasury Yield",  NarrativeCategory.MONETARY_POLICY, 0.05),
    "T10Y2Y":    ("Yield Curve 10Y-2Y",  NarrativeCategory.MARKET_CRISIS,   0.05),
    "DTWEXBGS":  ("Dollar Index",        NarrativeCategory.TRADE_WAR,        0.5),
    "DCOILWTICO":("Oil Price WTI",       NarrativeCategory.GEOPOLITICAL,    1.0),
}


def _compute_sentiment(series_id: str, change_pct: float) -> float:
    """Return a sentiment score in [-1, 1] based on the series and direction of change.

    Heuristics:
      - Rising CPI → inflationary pressure → bearish for risk assets.
      - Rising unemployment → economic weakness → bearish.
      - Yield curve turning negative → recession signal → very bearish.
      - Oil spike → geopolitical fear / supply shock → bearish.
      - All other signals use a mild directional heuristic.
    """
    rising = change_pct > 0

    if series_id == "CPIAUCSL":
        # Higher inflation is bearish for equities and crypto
        return -0.3 if rising else 0.2

    if series_id == "UNRATE":
        # Rising unemployment is bearish
        return -0.4 if rising else 0.3

    if series_id == "T10Y2Y":
        # Yield curve going further negative (or staying negative) is very bearish
        return -0.7 if rising else 0.3

    if series_id == "DCOILWTICO":
        # Oil spike → geopolitical fear / supply shock
        return -0.3 if rising else 0.1

    if series_id == "DFF":
        # Rate hike is bearish for risk assets; rate cut is bullish
        return -0.25 if rising else 0.25

    if series_id == "DGS10":
        # Rising long rates tighten financial conditions → mild bearish
        return -0.2 if rising else 0.15

    if series_id == "DTWEXBGS":
        # Stronger dollar can be bearish for commodities/EM; mild signal
        return -0.15 if rising else 0.15

    return 0.0


def _extract_keywords(name: str) -> list[str]:
    """Split the human-readable series name into keywords."""
    stopwords = {"the", "a", "an", "of", "in", "on", "at", "to", "and", "or", "for"}
    return [w.lower() for w in name.split() if w.lower() not in stopwords]


class FREDClient:
    """Client for the FRED API that emits NarrativeEvents on significant data moves.

    Usage::

        client = FREDClient(api_key=os.environ["FRED_API_KEY"])
        events = await client.fetch_all_tracked()

    Graceful degradation: if ``api_key`` is empty the client returns ``[]``
    from all public methods and logs a single debug message.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    # ------------------------------------------------------------------
    # Low-level fetch
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=8))
    async def fetch_series(
        self,
        session: aiohttp.ClientSession,
        series_id: str,
    ) -> list[dict[str, Any]]:
        """Fetch the last 5 observations for a FRED series.

        Parameters
        ----------
        session:
            An active ``aiohttp.ClientSession``.
        series_id:
            A valid FRED series identifier (e.g. ``"CPIAUCSL"``).

        Returns
        -------
        list[dict]
            Raw observation dicts from the FRED API (keys: ``date``, ``value``).
            Returns ``[]`` if the API key is absent, the series is unknown, or
            any HTTP / parsing error occurs.
        """
        if not self._api_key:
            return []

        params = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "limit": 5,
            "sort_order": "desc",
        }

        try:
            async with session.get(FRED_BASE, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    logger.warning(
                        "fred_http_error",
                        series_id=series_id,
                        status=resp.status,
                    )
                    return []

                data = await resp.json(content_type=None)
                observations: list[dict[str, Any]] = data.get("observations", [])
                logger.debug(
                    "fred_series_fetched",
                    series_id=series_id,
                    count=len(observations),
                )
                return observations

        except aiohttp.ClientError as exc:
            logger.warning("fred_request_error", series_id=series_id, error=str(exc))
            return []
        except Exception as exc:  # noqa: BLE001
            logger.warning("fred_unexpected_error", series_id=series_id, error=str(exc))
            return []

    # ------------------------------------------------------------------
    # High-level aggregation
    # ------------------------------------------------------------------

    async def fetch_all_tracked(self) -> list[NarrativeEvent]:
        """Fetch all TRACKED_SERIES concurrently and return significant moves.

        A ``NarrativeEvent`` is emitted for a series only when the absolute
        percentage change between the two most-recent observations exceeds the
        series-specific threshold defined in ``TRACKED_SERIES``.

        Returns an empty list immediately if no API key is configured.
        """
        if not self._api_key:
            logger.debug(
                "fred_no_api_key",
                msg="Set FRED_API_KEY for live economic data; returning empty list",
            )
            return []

        events: list[NarrativeEvent] = []

        async with aiohttp.ClientSession() as session:
            tasks = {
                series_id: asyncio.create_task(
                    self.fetch_series(session, series_id)
                )
                for series_id in TRACKED_SERIES
            }
            results: dict[str, list[dict[str, Any]]] = {
                sid: await task for sid, task in tasks.items()
            }

        for series_id, observations in results.items():
            name, category, threshold = TRACKED_SERIES[series_id]

            if len(observations) < 2:
                continue

            # FRED returns descending order: index 0 is the most recent
            try:
                latest_val = float(observations[0]["value"])
                prev_val   = float(observations[1]["value"])
            except (KeyError, ValueError, TypeError):
                logger.warning("fred_parse_error", series_id=series_id)
                continue

            if prev_val == 0.0:
                continue

            change_pct = (latest_val - prev_val) / abs(prev_val) * 100.0

            if abs(change_pct) <= threshold:
                continue

            sentiment = _compute_sentiment(series_id, change_pct)
            magnitude = min(abs(change_pct) / 5.0, 1.0)
            direction = "up" if change_pct > 0 else "down"
            content = (
                f"{name} moved {direction} {abs(change_pct):.2f}% "
                f"(latest: {latest_val}, prior: {prev_val})."
            )

            event = NarrativeEvent(
                event_id=str(uuid.uuid4()),
                source="fred_economic_data",
                content=content,
                timestamp=datetime.now(timezone.utc),
                category=category,
                sentiment=sentiment,
                magnitude=magnitude,
                keywords=_extract_keywords(name),
                metadata={
                    "series_id": series_id,
                    "series_name": name,
                    "latest_value": latest_val,
                    "previous_value": prev_val,
                    "change_pct": round(change_pct, 4),
                    "threshold": threshold,
                },
            )
            events.append(event)
            logger.info(
                "fred_narrative_event",
                series_id=series_id,
                change_pct=round(change_pct, 4),
                sentiment=sentiment,
                magnitude=round(magnitude, 4),
            )

        return events
