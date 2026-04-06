"""Fetch and cache real Polymarket historical data for backtesting."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import structlog

from polymarket_bot.data.models import Market, MarketCategory, Token
from polymarket_bot.utils.helpers import categorize_market

logger = structlog.get_logger()

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"


class HistoricalDataFetcher:
    """Fetch and cache real Polymarket historical data for backtesting."""

    def __init__(self, cache_dir: str = ".cache/historical") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._gamma = httpx.Client(base_url=GAMMA_BASE, timeout=30.0)
        self._clob = httpx.Client(base_url=CLOB_BASE, timeout=30.0)
        self._request_delay = 0.4

    def close(self) -> None:
        self._gamma.close()
        self._clob.close()

    def _cache_key(self, prefix: str, **kwargs: Any) -> str:
        raw = f"{prefix}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _load_cache(self, key: str) -> Any | None:
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            age_hours = (time.time() - path.stat().st_mtime) / 3600
            if age_hours < 24:
                return json.loads(path.read_text())
        return None

    def _save_cache(self, key: str, data: Any) -> None:
        path = self.cache_dir / f"{key}.json"
        path.write_text(json.dumps(data))

    # ── Market Discovery ─────────────────────────────────────────

    def fetch_active_markets(
        self,
        limit: int = 50,
        min_liquidity: float = 1000.0,
    ) -> list[dict]:
        """Fetch active markets from Gamma API, filtered by liquidity."""
        cache_key = self._cache_key("markets", limit=limit, min_liq=min_liquidity)
        cached = self._load_cache(cache_key)
        if cached:
            logger.info("markets_from_cache", count=len(cached))
            return cached

        all_markets: list[dict] = []
        offset = 0
        while len(all_markets) < limit:
            time.sleep(self._request_delay)
            try:
                resp = self._gamma.get(
                    "/markets",
                    params={
                        "limit": min(100, limit - len(all_markets)),
                        "offset": offset,
                        "active": True,
                        "closed": False,
                    },
                )
                resp.raise_for_status()
                batch = resp.json()
            except httpx.HTTPError as e:
                logger.error("gamma_fetch_failed", error=str(e), offset=offset)
                break

            if not batch:
                break

            for m in batch:
                liq = float(m.get("liquidityNum", 0) or 0)
                token_ids = self._extract_token_ids(m)
                if liq >= min_liquidity and len(token_ids) >= 2:
                    all_markets.append(m)

            offset += len(batch)
            logger.info("markets_fetched", batch=len(batch), total=len(all_markets))

            if len(batch) < 100:
                break

        self._save_cache(cache_key, all_markets[:limit])
        return all_markets[:limit]

    @staticmethod
    def _extract_token_ids(raw: dict) -> list[str]:
        """Extract token IDs from Gamma API response formats."""
        # Format 1: tokens array (newer)
        tokens = raw.get("tokens", [])
        if tokens and isinstance(tokens, list) and isinstance(tokens[0], dict):
            return [t.get("token_id", "") for t in tokens if t.get("token_id")]

        # Format 2: clobTokenIds as JSON string
        clob_raw = raw.get("clobTokenIds", "")
        if isinstance(clob_raw, str) and clob_raw:
            try:
                return json.loads(clob_raw)
            except (json.JSONDecodeError, TypeError):
                pass
        if isinstance(clob_raw, list):
            return clob_raw
        return []

    # ── Price History ────────────────────────────────────────────

    def fetch_price_history(
        self,
        token_id: str,
        interval: str = "1h",
        start_ts: int | None = None,
        end_ts: int | None = None,
        fidelity: int = 60,
    ) -> list[dict]:
        """Fetch price history for a single token. Returns list of {t, p}."""
        cache_key = self._cache_key(
            "prices", token=token_id, interval=interval,
            start=start_ts, end=end_ts, fidelity=fidelity,
        )
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        time.sleep(self._request_delay)
        params: dict[str, Any] = {"market": token_id, "interval": interval, "fidelity": fidelity}
        if start_ts:
            params["startTs"] = start_ts
        if end_ts:
            params["endTs"] = end_ts

        try:
            resp = self._clob.get("/prices-history", params=params)
            resp.raise_for_status()
            data = resp.json()
            history = data.get("history", [])
        except httpx.HTTPError as e:
            logger.error("price_history_failed", token=token_id[:16], error=str(e))
            return []

        self._save_cache(cache_key, history)
        logger.info("price_history_fetched", token=token_id[:16], points=len(history))
        return history

    def fetch_market_history(
        self,
        raw_market: dict,
        interval: str = "1h",
        days_back: int = 14,
    ) -> dict[str, list[dict]]:
        """Fetch price history for all tokens in a market."""
        now = int(datetime.now(timezone.utc).timestamp())
        start = now - days_back * 86400

        token_ids = self._extract_token_ids(raw_market)
        histories: dict[str, list[dict]] = {}
        for tid in token_ids:
            if not tid:
                continue
            history = self.fetch_price_history(tid, interval=interval, start_ts=start, end_ts=now)
            if history:
                histories[tid] = history

        return histories

    # ── Parse to Models ──────────────────────────────────────────

    def parse_market(self, raw: dict) -> Market:
        """Parse Gamma API market dict into a Market model."""
        tokens = []

        raw_tokens = raw.get("tokens", [])
        if raw_tokens and isinstance(raw_tokens, list) and isinstance(raw_tokens[0], dict):
            for t in raw_tokens:
                tokens.append(Token(
                    token_id=t.get("token_id", ""),
                    outcome=t.get("outcome", ""),
                    price=float(t.get("price", 0.5)),
                ))
        else:
            token_ids = self._extract_token_ids(raw)
            outcomes_raw = raw.get("outcomes", "[]")
            prices_raw = raw.get("outcomePrices", "[]")
            try:
                outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else (outcomes_raw or [])
                prices = json.loads(prices_raw) if isinstance(prices_raw, str) else (prices_raw or [])
            except (json.JSONDecodeError, TypeError):
                outcomes, prices = [], []

            for i, tid in enumerate(token_ids):
                outcome = outcomes[i] if i < len(outcomes) else f"Outcome_{i}"
                price = float(prices[i]) if i < len(prices) else 0.5
                tokens.append(Token(token_id=tid, outcome=outcome, price=price))

        end_date = None
        end_str = raw.get("end_date_iso") or raw.get("endDate")
        if end_str:
            try:
                end_date = datetime.fromisoformat(str(end_str).replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        question = raw.get("question", "")
        tags = raw.get("tags", []) or []
        cat = categorize_market(question, tags)

        return Market(
            condition_id=raw.get("condition_id", raw.get("conditionId", "")),
            question=question,
            slug=raw.get("slug", raw.get("market_slug", "")),
            tokens=tokens,
            category=MarketCategory(cat) if cat in [e.value for e in MarketCategory] else MarketCategory.OTHER,
            end_date=end_date,
            volume_24h=float(raw.get("volume_num_24hr", raw.get("volume24hr", raw.get("volume", 0))) or 0),
            liquidity=float(raw.get("liquidityNum", raw.get("liquidity_num", 0)) or 0),
            active=raw.get("active", True),
            description=raw.get("description", ""),
            tags=tags,
        )

    # ── Full Dataset Builder ─────────────────────────────────────

    def build_backtest_dataset(
        self,
        num_markets: int = 20,
        days_back: int = 14,
        min_liquidity: float = 2000.0,
        interval: str = "1h",
    ) -> dict:
        """Build a complete dataset for backtesting from real Polymarket data.

        Returns dict with:
        - markets: list of Market model objects
        - price_histories: {token_id: [float prices]}
        - timestamps: {token_id: [int unix timestamps]}
        - raw_markets: original API dicts
        - metadata: fetch info
        """
        cache_key = self._cache_key(
            "dataset", n=num_markets, days=days_back,
            liq=min_liquidity, interval=interval,
        )
        cached = self._load_cache(cache_key)
        if cached:
            logger.info("dataset_from_cache", markets=len(cached.get("raw_markets", [])))
            markets = [self.parse_market(r) for r in cached["raw_markets"]]
            return {
                "markets": markets,
                "price_histories": cached["price_histories"],
                "timestamps": cached["timestamps"],
                "raw_markets": cached["raw_markets"],
                "metadata": cached["metadata"],
            }

        logger.info("building_dataset", num_markets=num_markets, days_back=days_back)
        raw_markets = self.fetch_active_markets(limit=num_markets * 2, min_liquidity=min_liquidity)

        markets: list[Market] = []
        price_histories: dict[str, list[float]] = {}
        timestamps: dict[str, list[int]] = {}
        used_raw: list[dict] = []

        for raw in raw_markets:
            if len(markets) >= num_markets:
                break

            histories = self.fetch_market_history(raw, interval=interval, days_back=days_back)
            if not histories:
                continue

            min_points = 24 * max(1, days_back // 7)
            valid = all(len(h) >= min_points for h in histories.values())
            if not valid:
                continue

            market = self.parse_market(raw)
            markets.append(market)
            used_raw.append(raw)

            for tid, hist in histories.items():
                price_histories[tid] = [float(p["p"]) for p in hist]
                timestamps[tid] = [int(p["t"]) for p in hist]

            logger.info(
                "market_added",
                question=market.question[:60],
                tokens=len(histories),
                points=min(len(h) for h in histories.values()),
            )

        result = {
            "markets": markets,
            "price_histories": price_histories,
            "timestamps": timestamps,
            "raw_markets": used_raw,
            "metadata": {
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "num_markets": len(markets),
                "days_back": days_back,
                "interval": interval,
                "min_liquidity": min_liquidity,
            },
        }

        cache_data = {k: v for k, v in result.items() if k != "markets"}
        self._save_cache(cache_key, cache_data)

        logger.info("dataset_built", markets=len(markets), tokens=len(price_histories))
        return result
