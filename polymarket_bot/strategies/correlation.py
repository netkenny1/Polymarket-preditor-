"""Cross-market correlation trading strategy.

Core idea: Related prediction markets often move together, but with
a lag. When Event A affects Event B's probability, the A market
prices it in first and the B market lags. We detect these correlations
and trade the lagging market.

Examples:
- "Trump wins GOP primary" -> "Trump wins general election"
- "Fed raises rates" -> "S&P 500 drops 5%"
- "Lakers win Game 6" -> "Lakers win championship"

Also detects negative correlations:
- "Candidate A wins" vs "Candidate B wins" in same race

Uses rolling correlation of price changes to detect and trade these
relationships. The edge comes from the market B's slow adjustment.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import numpy as np
import structlog

from polymarket_bot.data.models import Market, OrderBook, Side, Signal
from polymarket_bot.strategies.base import BaseStrategy
from polymarket_bot.utils.helpers import clamp

logger = structlog.get_logger()


class CorrelationStrategy(BaseStrategy):
    """Trade lagging correlated markets."""

    name = "correlation"

    def __init__(
        self,
        min_correlation: float = 0.60,
        min_history: int = 30,
        lag_threshold: float = 0.03,
        min_edge: float = 0.04,
    ) -> None:
        self.min_correlation = min_correlation
        self.min_history = min_history
        self.lag_threshold = lag_threshold
        self.min_edge = min_edge
        self._correlation_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._cache_time: float = 0.0

    def generate_signals(
        self,
        markets: list[Market],
        order_books: dict[str, OrderBook],
        context: dict[str, Any],
    ) -> list[Signal]:
        signals = []
        tradeable = self.filter_tradeable_markets(markets, order_books)

        if len(tradeable) < 2:
            return signals

        # Collect price histories
        histories: dict[str, list[float]] = {}
        market_map: dict[str, Market] = {}
        for m in tradeable:
            hist = context.get(f"price_history_{m.condition_id}", [])
            if len(hist) >= self.min_history:
                histories[m.condition_id] = hist
                market_map[m.condition_id] = m

        if len(histories) < 2:
            return signals

        # Find correlated pairs and detect leads/lags
        pairs = self._find_correlated_pairs(histories)

        for (leader_id, lagger_id), corr_info in pairs.items():
            try:
                signal = self._generate_lag_signal(
                    leader_id, lagger_id, corr_info,
                    market_map, histories, order_books,
                )
                if signal is not None:
                    signals.append(signal)
            except Exception as e:
                logger.error("correlation_error", error=str(e))

        return signals

    def _find_correlated_pairs(
        self, histories: dict[str, list[float]]
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """Find market pairs with significant correlation."""
        if self._correlation_cache and time.monotonic() - self._cache_time < 300:
            return self._correlation_cache

        pairs: dict[tuple[str, str], dict[str, Any]] = {}
        ids = list(histories.keys())

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id_a, id_b = ids[i], ids[j]
                hist_a = np.array(histories[id_a])
                hist_b = np.array(histories[id_b])

                # Align lengths
                min_len = min(len(hist_a), len(hist_b))
                if min_len < self.min_history:
                    continue

                ret_a = np.diff(hist_a[-min_len:])
                ret_b = np.diff(hist_b[-min_len:])

                # Contemporaneous correlation
                corr = np.corrcoef(ret_a, ret_b)[0, 1]
                if np.isnan(corr) or abs(corr) < self.min_correlation:
                    continue

                # Lead-lag analysis: does A lead B or B lead A?
                # Check correlation with 1-step lag
                if len(ret_a) > 2:
                    corr_a_leads = np.corrcoef(ret_a[:-1], ret_b[1:])[0, 1]
                    corr_b_leads = np.corrcoef(ret_b[:-1], ret_a[1:])[0, 1]

                    if np.isnan(corr_a_leads):
                        corr_a_leads = 0
                    if np.isnan(corr_b_leads):
                        corr_b_leads = 0

                    # The market with higher lagged correlation is the leader
                    if abs(corr_a_leads) > abs(corr_b_leads) and abs(corr_a_leads) > self.min_correlation * 0.8:
                        leader, lagger = id_a, id_b
                        lag_corr = corr_a_leads
                    elif abs(corr_b_leads) > self.min_correlation * 0.8:
                        leader, lagger = id_b, id_a
                        lag_corr = corr_b_leads
                    else:
                        continue

                    pairs[(leader, lagger)] = {
                        "correlation": float(corr),
                        "lag_correlation": float(lag_corr),
                        "direction": "positive" if lag_corr > 0 else "negative",
                    }

        self._correlation_cache = pairs
        self._cache_time = time.monotonic()
        return pairs

    def _generate_lag_signal(
        self,
        leader_id: str,
        lagger_id: str,
        corr_info: dict[str, Any],
        market_map: dict[str, Market],
        histories: dict[str, list[float]],
        order_books: dict[str, OrderBook],
    ) -> Signal | None:
        """Generate a signal on the lagging market based on leader's recent move."""
        leader_hist = histories[leader_id]
        lagger_hist = histories[lagger_id]

        # Recent move in leader
        leader_move = leader_hist[-1] - leader_hist[-3]  # 3-step recent move

        # Recent move in lagger
        lagger_move = lagger_hist[-1] - lagger_hist[-3]

        # Expected move in lagger based on correlation
        if corr_info["direction"] == "positive":
            expected_lagger_move = leader_move
        else:
            expected_lagger_move = -leader_move

        # Gap: how much has the lagger not yet caught up?
        gap = expected_lagger_move - lagger_move

        if abs(gap) < self.lag_threshold:
            return None  # Not enough lag

        lagger_market = market_map.get(lagger_id)
        if lagger_market is None:
            return None

        current_price = lagger_hist[-1]
        fair_value = clamp(current_price + gap * 0.6, 0.05, 0.95)  # Expect 60% convergence
        edge = abs(fair_value - current_price)

        if edge < self.min_edge:
            return None

        # Direction
        if gap > 0:
            # Lagger should go up -> buy YES
            token = next((t for t in lagger_market.tokens if t.outcome == "Yes"), None)
            side = Side.BUY
            market_price = current_price
        else:
            # Lagger should go down -> buy NO
            token = next((t for t in lagger_market.tokens if t.outcome == "No"), None)
            if token is None:
                return None
            side = Side.BUY
            fair_value = 1.0 - fair_value
            market_price = 1.0 - current_price
            edge = abs(fair_value - market_price)

        if token is None:
            return None

        confidence = min(0.80, abs(corr_info["lag_correlation"]) * abs(gap) * 5)

        return Signal(
            market_condition_id=lagger_id,
            token_id=token.token_id,
            side=side,
            outcome=token.outcome,
            estimated_fair_value=fair_value,
            market_price=market_price,
            edge=edge,
            confidence=confidence,
            strategy=self.name,
            metadata={
                "leader_market": leader_id,
                "correlation": round(corr_info["correlation"], 3),
                "lag_correlation": round(corr_info["lag_correlation"], 3),
                "gap": round(float(gap), 4),
                "direction": corr_info["direction"],
            },
        )
