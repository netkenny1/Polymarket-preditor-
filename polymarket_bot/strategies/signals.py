"""Signal aggregation and ensemble scoring.

Combines signals from multiple strategies into a single ranked list
of trade opportunities. Uses confidence-weighted averaging when multiple
strategies agree on the same market.

Key insight: Strategies that independently agree on a trade are much
more reliable than a single strategy signal. We reward consensus.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import structlog

from polymarket_bot.data.models import Signal, Side

logger = structlog.get_logger()


# Strategy reliability weights (tuned based on backtesting)
STRATEGY_WEIGHTS: dict[str, float] = {
    "arbitrage": 1.5,     # Highest weight - near risk-free
    "statistical": 1.2,   # Strong models
    "correlation": 1.1,   # Cross-market lead-lag (structural edge)
    "momentum": 1.1,      # Trend-following (strong in trending regimes)
    "contrarian": 1.1,    # Mean-reversion (strong in ranging regimes)
    "btc_daily": 1.1,     # BTC direction (intraday momentum + seasonality)
    "event_catalyst": 1.05,  # Scheduled event mispricings
    "news_reactor": 1.0,  # Breaking news / Trump tweet signals
    "volatility": 1.0,    # Vol clustering / Bollinger regime
    "time_decay": 1.0,    # Expiry dynamics
    "sentiment": 1.0,     # Good but noisy
    "microstructure": 0.9,  # Order book signals (fast-decaying edge)
    "market_maker": 0.4,  # Lower edge per trade — reduced to avoid noise
    "narrative_analysis": 1.15,  # Historical pattern + multi-scenario consensus
}


class SignalAggregator:
    """Aggregate and rank signals from multiple strategies.

    When multiple strategies signal the same market+direction, combine
    them into a stronger composite signal. Conflicting signals cancel out.
    """

    def __init__(self, min_composite_edge: float = 0.03) -> None:
        self.min_composite_edge = min_composite_edge

    def aggregate(self, signals: list[Signal]) -> list[Signal]:
        """Combine signals for the same token into composite signals.

        Returns:
            Sorted list of signals, strongest first.
        """
        if not signals:
            return []

        # Group signals by (token_id, side)
        groups: dict[tuple[str, Side], list[Signal]] = defaultdict(list)
        for sig in signals:
            key = (sig.token_id, sig.side)
            groups[key].append(sig)

        composites: list[Signal] = []

        for (token_id, side), group in groups.items():
            if len(group) == 1:
                sig = group[0]
                # Apply strategy weight
                weight = STRATEGY_WEIGHTS.get(sig.strategy, 1.0)
                sig_copy = Signal(
                    market_condition_id=sig.market_condition_id,
                    token_id=sig.token_id,
                    side=sig.side,
                    outcome=sig.outcome,
                    estimated_fair_value=sig.estimated_fair_value,
                    market_price=sig.market_price,
                    edge=sig.edge,
                    confidence=sig.confidence * weight,
                    strategy=sig.strategy,
                    metadata=sig.metadata,
                    timestamp=sig.timestamp,
                )
                composites.append(sig_copy)
            else:
                # Multiple strategies agree - combine
                composite = self._combine_signals(group)
                if composite is not None:
                    composites.append(composite)

        # Filter by minimum edge
        composites = [s for s in composites if s.edge >= self.min_composite_edge]

        # Sort by edge * confidence (expected value)
        composites.sort(key=lambda s: s.edge * s.confidence, reverse=True)

        logger.info(
            "signals_aggregated",
            input_count=len(signals),
            output_count=len(composites),
        )

        return composites

    def _combine_signals(self, signals: list[Signal]) -> Signal | None:
        """Combine multiple signals for the same token+direction."""
        if not signals:
            return None

        # Weighted average of fair values and edges
        total_weight = 0.0
        weighted_fv = 0.0
        weighted_edge = 0.0
        max_confidence = 0.0
        strategies = []

        for sig in signals:
            w = STRATEGY_WEIGHTS.get(sig.strategy, 1.0) * sig.confidence
            total_weight += w
            weighted_fv += sig.estimated_fair_value * w
            weighted_edge += sig.edge * w
            max_confidence = max(max_confidence, sig.confidence)
            strategies.append(sig.strategy)

        if total_weight == 0:
            return None

        avg_fv = weighted_fv / total_weight
        avg_edge = weighted_edge / total_weight

        # Consensus bonus: multiple strategies agreeing boosts confidence
        consensus_bonus = min(0.35, 0.15 * (len(signals) - 1))
        combined_confidence = min(1.0, max_confidence + consensus_bonus)

        base = signals[0]
        return Signal(
            market_condition_id=base.market_condition_id,
            token_id=base.token_id,
            side=base.side,
            outcome=base.outcome,
            estimated_fair_value=avg_fv,
            market_price=base.market_price,
            edge=avg_edge,
            confidence=combined_confidence,
            strategy="ensemble",
            metadata={
                "strategies": strategies,
                "consensus_count": len(signals),
                "individual_edges": [s.edge for s in signals],
            },
            timestamp=base.timestamp,
        )

    def check_conflicts(self, signals: list[Signal]) -> list[tuple[Signal, Signal]]:
        """Identify conflicting signals (same market, opposite directions)."""
        conflicts = []
        by_market: dict[str, list[Signal]] = defaultdict(list)

        for sig in signals:
            by_market[sig.market_condition_id].append(sig)

        for market_id, market_signals in by_market.items():
            buys = [s for s in market_signals if s.side == Side.BUY]
            sells = [s for s in market_signals if s.side == Side.SELL]

            # Check for conflicting outcomes
            for buy in buys:
                for sell in sells:
                    if buy.token_id == sell.token_id:
                        conflicts.append((buy, sell))

        return conflicts
