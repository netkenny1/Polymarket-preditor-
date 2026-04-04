"""Smart position exit management.

The entry is only half the trade - exits determine realized profit.
This module implements multiple exit strategies that work together:

1. Profit Targets: Scale out at predefined profit levels
   - Take 50% off at 2x edge, let rest ride with trailing stop
   - Take 100% off at 3x edge (full target)

2. Time-Based Exits: Close positions that aren't working
   - If position hasn't moved in our favor after N steps, close
   - Prevents capital lock-up in "dead" positions

3. Edge Decay Exits: Close when the edge that justified the trade
   has decayed below a threshold

4. Market Resolution Exits: Close positions as markets approach
   resolution date for certainty capture

Professional market makers and quant funds use multi-layered exit
strategies. A single stop loss is amateur hour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

from polymarket_bot.data.models import Position, Side
from polymarket_bot.utils.helpers import clamp, time_to_expiry_hours

logger = structlog.get_logger()


@dataclass
class ExitRule:
    """A triggered exit rule for a position."""

    token_id: str
    position: Position
    exit_type: str  # profit_target, time_exit, edge_decay, resolution_exit
    reason: str
    sell_fraction: float  # 0.0 to 1.0 of position to sell
    urgency: float  # 0.0 to 1.0, higher = more aggressive pricing


@dataclass
class PositionMeta:
    """Metadata tracked per position for exit decisions."""

    entry_step: int = 0
    entry_edge: float = 0.0
    initial_size: float = 0.0
    partial_exits: int = 0
    highest_unrealized_pnl: float = 0.0
    steps_in_profit: int = 0
    steps_in_loss: int = 0
    market_end_date: datetime | None = None


class ExitManager:
    """Manage position exits with multi-layered exit strategy.

    Tracks each position's lifecycle and generates exit signals
    when profit targets, time limits, or other conditions are met.
    """

    def __init__(
        self,
        profit_target_1x: float = 0.12,   # Take 50% at 12% profit
        profit_target_2x: float = 0.25,   # Take rest at 25% profit
        max_hold_steps: int = 100,         # Close after 100 steps if flat
        stale_threshold: float = 0.02,     # Close if < 2% move after 30 steps
        stale_steps: int = 30,
        resolution_hours_threshold: float = 4.0,  # Close 4h before resolution
    ) -> None:
        self.profit_target_1x = profit_target_1x
        self.profit_target_2x = profit_target_2x
        self.max_hold_steps = max_hold_steps
        self.stale_threshold = stale_threshold
        self.stale_steps = stale_steps
        self.resolution_hours = resolution_hours_threshold

        # Track metadata per position
        self._meta: dict[str, PositionMeta] = {}
        self._current_step: int = 0

    def register_entry(
        self,
        token_id: str,
        edge: float,
        size: float,
        market_end_date: datetime | None = None,
    ) -> None:
        """Register a new position entry for tracking."""
        self._meta[token_id] = PositionMeta(
            entry_step=self._current_step,
            entry_edge=edge,
            initial_size=size,
            market_end_date=market_end_date,
        )

    def advance_step(self) -> None:
        """Advance the step counter (called each trading cycle)."""
        self._current_step += 1

    def check_exits(
        self,
        positions: dict[str, Position],
    ) -> list[ExitRule]:
        """Check all positions for exit conditions.

        Returns a list of ExitRule objects describing what to close and why.
        """
        exits: list[ExitRule] = []

        for token_id, pos in positions.items():
            if pos.size <= 0:
                continue

            meta = self._meta.get(token_id)
            if meta is None:
                # Position not tracked, create basic meta
                meta = PositionMeta(entry_step=self._current_step, initial_size=pos.size)
                self._meta[token_id] = meta

            # Update tracking
            pnl_pct = (pos.current_price - pos.avg_entry_price) / pos.avg_entry_price if pos.avg_entry_price > 0 else 0
            if pnl_pct > meta.highest_unrealized_pnl:
                meta.highest_unrealized_pnl = pnl_pct
            if pnl_pct > 0.005:
                meta.steps_in_profit += 1
            elif pnl_pct < -0.005:
                meta.steps_in_loss += 1

            hold_steps = self._current_step - meta.entry_step

            # ── Rule 1: Profit Targets ───────────────────────────
            exit = self._check_profit_targets(token_id, pos, meta, pnl_pct)
            if exit:
                exits.append(exit)
                continue  # Don't double-exit

            # ── Rule 2: Time-Based Exit ──────────────────────────
            exit = self._check_time_exit(token_id, pos, meta, pnl_pct, hold_steps)
            if exit:
                exits.append(exit)
                continue

            # ── Rule 3: Stale Position ───────────────────────────
            exit = self._check_stale(token_id, pos, meta, pnl_pct, hold_steps)
            if exit:
                exits.append(exit)
                continue

            # ── Rule 4: Resolution Approach ──────────────────────
            exit = self._check_resolution(token_id, pos, meta)
            if exit:
                exits.append(exit)

        return exits

    def _check_profit_targets(
        self, token_id: str, pos: Position, meta: PositionMeta, pnl_pct: float
    ) -> ExitRule | None:
        """Take profits at predefined levels."""
        if pnl_pct >= self.profit_target_2x and meta.partial_exits >= 1:
            # Full exit at 2x target
            meta.partial_exits += 1
            return ExitRule(
                token_id=token_id, position=pos,
                exit_type="profit_target",
                reason=f"Full profit target hit: {pnl_pct:.1%}",
                sell_fraction=1.0,
                urgency=0.3,  # Not desperate, can wait for good fill
            )
        elif pnl_pct >= self.profit_target_1x and meta.partial_exits == 0:
            # Partial exit at 1x target (sell half)
            meta.partial_exits += 1
            return ExitRule(
                token_id=token_id, position=pos,
                exit_type="profit_target",
                reason=f"Partial profit target hit: {pnl_pct:.1%}",
                sell_fraction=0.5,
                urgency=0.3,
            )
        return None

    def _check_time_exit(
        self, token_id: str, pos: Position, meta: PositionMeta,
        pnl_pct: float, hold_steps: int,
    ) -> ExitRule | None:
        """Close positions held too long without working."""
        if hold_steps >= self.max_hold_steps:
            return ExitRule(
                token_id=token_id, position=pos,
                exit_type="time_exit",
                reason=f"Max hold time ({hold_steps} steps), P&L: {pnl_pct:.1%}",
                sell_fraction=1.0,
                urgency=0.5,
            )
        return None

    def _check_stale(
        self, token_id: str, pos: Position, meta: PositionMeta,
        pnl_pct: float, hold_steps: int,
    ) -> ExitRule | None:
        """Close positions that aren't moving."""
        if hold_steps >= self.stale_steps and abs(pnl_pct) < self.stale_threshold:
            return ExitRule(
                token_id=token_id, position=pos,
                exit_type="stale_exit",
                reason=f"Stale position: {pnl_pct:.1%} after {hold_steps} steps",
                sell_fraction=1.0,
                urgency=0.4,
            )
        return None

    def _check_resolution(
        self, token_id: str, pos: Position, meta: PositionMeta,
    ) -> ExitRule | None:
        """Close positions approaching market resolution."""
        if meta.market_end_date is None:
            return None

        hours_left = time_to_expiry_hours(meta.market_end_date)
        if hours_left <= self.resolution_hours:
            pnl_pct = (pos.current_price - pos.avg_entry_price) / pos.avg_entry_price if pos.avg_entry_price > 0 else 0
            return ExitRule(
                token_id=token_id, position=pos,
                exit_type="resolution_exit",
                reason=f"Market resolving in {hours_left:.1f}h, P&L: {pnl_pct:.1%}",
                sell_fraction=1.0,
                urgency=0.7,  # Fairly urgent near resolution
            )
        return None

    def remove_position(self, token_id: str) -> None:
        """Clean up tracking data when a position is fully closed."""
        self._meta.pop(token_id, None)
