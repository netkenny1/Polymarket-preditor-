"""Prediction tracking and self-improvement module.

This module closes the feedback loop for the trading bot. Every signal
that turns into a trade is recorded as a ``PredictionRecord``. When the
position closes we record the realized outcome. From that running ledger
we derive three self-improvement behaviours:

1. ``PredictionTracker`` — persistent store of predictions with rolling
   analytics (accuracy, Sharpe, calibration) sliced by strategy or
   narrative category.
2. ``StrategyAutoTuner`` — rebalances the hardcoded ``STRATEGY_WEIGHTS``
   dictionary based on each strategy's rolling Sharpe ratio, with a
   regime-aware overlay that boosts strategies suited to the current
   market regime.
3. ``CalibrationAdjuster`` — remaps raw confidence scores through an
   empirical calibration curve so that a "70% confident" signal actually
   wins ~70% of the time.

The module is intentionally decoupled from the rest of the bot: it takes
``Signal`` / ``TradeResult`` dataclasses and a plain ``dict`` of weights,
and returns plain Python objects. Integration happens at the call-site.
"""

from __future__ import annotations

import json
import uuid
from bisect import bisect_left
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from polymarket_bot.data.models import Side, Signal, TradeResult
# SQLiteStore imported lazily to avoid circular imports
_SQLiteStore = None
try:
    from polymarket_bot.learning.sqlite_store import SQLiteStore as _SQLiteStore
except Exception:
    pass

logger = structlog.get_logger()


# ────────────────────────────────────────────────────────────────────────
# Prediction record
# ────────────────────────────────────────────────────────────────────────


@dataclass
class PredictionRecord:
    """A single prediction tracked from entry through outcome.

    ``correct`` is ``True`` when the directional bet paid off (a BUY that
    exited higher than entry, or a SELL that exited lower). ``realized_pnl``
    is signed dollars, net of fees when available.
    """

    prediction_id: str
    timestamp: datetime
    strategy: str
    market_condition_id: str
    token_id: str
    side: str  # "BUY" or "SELL"
    entry_price: float
    predicted_fair_value: float
    predicted_edge: float
    confidence: float
    size_usd: float
    narrative_category: str | None = None
    exit_price: float | None = None
    realized_pnl: float | None = None
    correct: bool | None = None
    closed: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PredictionRecord":
        d = dict(d)
        ts = d.get("timestamp")
        if isinstance(ts, str):
            d["timestamp"] = datetime.fromisoformat(ts)
        return cls(**d)


# ────────────────────────────────────────────────────────────────────────
# Prediction tracker
# ────────────────────────────────────────────────────────────────────────


class PredictionTracker:
    """Persistent store of predictions with rolling analytics.

    Predictions are kept in-memory in a list and optionally mirrored to a
    JSON file. Analytics methods all accept a ``window_days`` parameter so
    callers can get "last 30 days" style stats without filtering manually.
    """

    CONFIDENCE_BUCKETS: tuple[tuple[float, float], ...] = (
        (0.0, 0.2),
        (0.2, 0.4),
        (0.4, 0.6),
        (0.6, 0.8),
        (0.8, 1.01),
    )

    def __init__(self, persist_path: str | None = None, db: Any = None) -> None:
        self.persist_path: Path | None = Path(persist_path) if persist_path else None
        self.records: list[PredictionRecord] = []
        self._by_id: dict[str, PredictionRecord] = {}
        self._db = db  # SQLiteStore instance for dual-write persistence
        if self.persist_path and self.persist_path.exists():
            try:
                self.load()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("prediction_tracker_load_failed", error=str(exc))

    # ── Recording ───────────────────────────────────────────────────

    def record_prediction(
        self,
        signal: Signal,
        size_usd: float,
        trade_result: TradeResult,
        narrative_category: str | None = None,
    ) -> str:
        """Persist a new open prediction from a signal + executed trade.

        The entry price is taken from the ``TradeResult`` fill when
        available, otherwise from the signal's market price.
        """
        entry_price = (
            trade_result.fill_price
            if trade_result.success and trade_result.fill_price > 0
            else signal.market_price
        )
        side = signal.side.value if isinstance(signal.side, Side) else str(signal.side)

        record = PredictionRecord(
            prediction_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            strategy=signal.strategy,
            market_condition_id=signal.market_condition_id,
            token_id=signal.token_id,
            side=side,
            entry_price=entry_price,
            predicted_fair_value=signal.estimated_fair_value,
            predicted_edge=signal.edge,
            confidence=signal.confidence,
            size_usd=size_usd,
            narrative_category=narrative_category,
        )
        self.records.append(record)
        self._by_id[record.prediction_id] = record

        logger.info(
            "prediction_recorded",
            prediction_id=record.prediction_id,
            strategy=record.strategy,
            side=record.side,
            edge=round(record.predicted_edge, 4),
            confidence=round(record.confidence, 3),
            size_usd=round(size_usd, 2),
        )

        if self.persist_path is not None:
            self.save()

        # Dual-write to SQLite for persistent self-improvement
        if self._db is not None:
            try:
                self._db.save_prediction(record)
            except Exception:
                pass

        return record.prediction_id

    def close_prediction(
        self,
        prediction_id: str,
        exit_price: float,
        realized_pnl: float,
    ) -> None:
        """Mark a prediction as closed and compute correctness."""
        record = self._by_id.get(prediction_id)
        if record is None:
            logger.warning("close_prediction_not_found", prediction_id=prediction_id)
            return

        record.exit_price = exit_price
        record.realized_pnl = realized_pnl
        if record.side == "BUY":
            record.correct = exit_price > record.entry_price
        else:
            record.correct = exit_price < record.entry_price
        record.closed = True

        logger.info(
            "prediction_closed",
            prediction_id=prediction_id,
            strategy=record.strategy,
            correct=record.correct,
            pnl=round(realized_pnl, 2),
        )

        if self.persist_path is not None:
            self.save()

        # Dual-write close to SQLite
        if self._db is not None:
            try:
                self._db.close_prediction(
                    prediction_id, exit_price, realized_pnl,
                    correct=record.correct,
                )
            except Exception:
                pass

    # ── Query helpers ───────────────────────────────────────────────

    def _filter_window(
        self,
        window_days: int,
        *,
        closed_only: bool = True,
    ) -> list[PredictionRecord]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
        out: list[PredictionRecord] = []
        for r in self.records:
            ts = r.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts < cutoff:
                continue
            if closed_only and not r.closed:
                continue
            out.append(r)
        return out

    @staticmethod
    def _empty_stats() -> dict[str, Any]:
        return {
            "n_predictions": 0,
            "win_rate": 0.0,
            "avg_edge": 0.0,
            "avg_pnl": 0.0,
            "sharpe": 0.0,
            "accuracy_by_confidence_bucket": {},
        }

    def _summarize(self, records: list[PredictionRecord]) -> dict[str, Any]:
        if not records:
            return self._empty_stats()

        pnls = np.array([r.realized_pnl or 0.0 for r in records], dtype=float)
        wins = np.array([1.0 if r.correct else 0.0 for r in records], dtype=float)
        edges = np.array([r.predicted_edge for r in records], dtype=float)

        if pnls.std(ddof=0) > 1e-9:
            sharpe = float(pnls.mean() / pnls.std(ddof=0) * np.sqrt(252))
        else:
            sharpe = 0.0

        buckets: dict[str, dict[str, float]] = {}
        for lo, hi in self.CONFIDENCE_BUCKETS:
            mask = [lo <= r.confidence < hi for r in records]
            count = int(sum(mask))
            if count == 0:
                continue
            bucket_wins = float(np.mean([w for w, m in zip(wins, mask) if m]))
            label = f"{lo:.1f}-{hi:.1f}"
            buckets[label] = {"n": count, "win_rate": round(bucket_wins, 4)}

        return {
            "n_predictions": len(records),
            "win_rate": round(float(wins.mean()), 4),
            "avg_edge": round(float(edges.mean()), 4),
            "avg_pnl": round(float(pnls.mean()), 4),
            "sharpe": round(sharpe, 4),
            "accuracy_by_confidence_bucket": buckets,
        }

    def get_strategy_accuracy(
        self, strategy: str, window_days: int = 30
    ) -> dict[str, Any]:
        """Rolling accuracy stats for a single strategy."""
        records = [
            r for r in self._filter_window(window_days) if r.strategy == strategy
        ]
        stats = self._summarize(records)
        stats["strategy"] = strategy
        return stats

    def get_category_accuracy(
        self, category: str, window_days: int = 30
    ) -> dict[str, Any]:
        """Rolling accuracy stats for a narrative category."""
        records = [
            r
            for r in self._filter_window(window_days)
            if r.narrative_category == category
        ]
        stats = self._summarize(records)
        stats["category"] = category
        return stats

    def get_confidence_calibration(
        self, window_days: int = 30, n_buckets: int = 10
    ) -> list[dict[str, Any]]:
        """Return points for a reliability/calibration diagram.

        Each dict has ``confidence_bucket`` (midpoint), ``predicted``
        (mean confidence in bucket), ``actual_win_rate``, and ``n``.
        """
        records = self._filter_window(window_days)
        if not records:
            return []

        edges = np.linspace(0.0, 1.0, n_buckets + 1)
        out: list[dict[str, Any]] = []
        for i in range(n_buckets):
            lo, hi = edges[i], edges[i + 1]
            hi_eff = hi + (1e-9 if i == n_buckets - 1 else 0.0)
            in_bucket = [r for r in records if lo <= r.confidence < hi_eff]
            if not in_bucket:
                continue
            predicted = float(np.mean([r.confidence for r in in_bucket]))
            actual = float(np.mean([1.0 if r.correct else 0.0 for r in in_bucket]))
            out.append(
                {
                    "confidence_bucket": round((lo + hi) / 2, 3),
                    "predicted": round(predicted, 4),
                    "actual_win_rate": round(actual, 4),
                    "n": len(in_bucket),
                }
            )
        return out

    # ── Persistence ─────────────────────────────────────────────────

    def save(self) -> None:
        """Write the full record list to ``persist_path`` as JSON."""
        if self.persist_path is None:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [r.to_dict() for r in self.records]
        tmp = self.persist_path.with_suffix(self.persist_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.persist_path)

    def load(self) -> None:
        """Load records from ``persist_path`` (replaces in-memory state)."""
        if self.persist_path is None or not self.persist_path.exists():
            return
        raw = json.loads(self.persist_path.read_text())
        self.records = [PredictionRecord.from_dict(d) for d in raw]
        self._by_id = {r.prediction_id: r for r in self.records}
        logger.info("prediction_tracker_loaded", count=len(self.records))


# ────────────────────────────────────────────────────────────────────────
# Strategy auto-tuner
# ────────────────────────────────────────────────────────────────────────


class StrategyAutoTuner:
    """Auto-tune ``STRATEGY_WEIGHTS`` from rolling Sharpe ratios.

    The tuner keeps an exponential moving average of the live weights so
    adjustments are smooth: new target weights are blended with current
    weights using ``learning_rate``. Sharpe ratio is the signal because
    it penalises high-variance strategies that happen to be up on the
    window.
    """

    # Per-regime multipliers applied on top of the learned weights.
    _REGIME_MULTIPLIERS: dict[str, dict[str, float]] = {
        "trending": {
            "momentum": 1.25,
            "btc_daily": 1.20,
            "contrarian": 0.80,
            "market_maker": 0.85,
        },
        "ranging": {
            "contrarian": 1.25,
            "market_maker": 1.25,
            "momentum": 0.80,
            "btc_daily": 0.85,
        },
        "volatile": {
            "volatility": 1.30,
            "narrative_analysis": 1.20,
            "market_maker": 0.70,
            "microstructure": 0.85,
        },
    }

    WEIGHT_MIN: float = 0.2
    WEIGHT_MAX: float = 2.0

    def __init__(
        self,
        tracker: PredictionTracker,
        base_weights: dict[str, float],
        window_days: int = 30,
        learning_rate: float = 0.1,
        min_samples: int = 10,
    ) -> None:
        self.tracker = tracker
        self.base_weights: dict[str, float] = dict(base_weights)
        self.window_days = window_days
        self.learning_rate = learning_rate
        self.min_samples = min_samples
        # EMA of weights tracked internally for smooth updates.
        self._ema_weights: dict[str, float] = dict(base_weights)

    # ── Core computation ────────────────────────────────────────────

    def compute_weight_adjustments(self) -> dict[str, float]:
        """Compute new weights based on each strategy's recent Sharpe.

        Rules:

        * Strategies with < ``min_samples`` closed predictions keep their
          base weight.
        * Sharpe > 0.5 → linear boost toward ``WEIGHT_MAX``.
        * Sharpe < 0  → linear penalty toward ``WEIGHT_MIN``.
        * In between  → hold steady at base weight.
        """
        new_weights: dict[str, float] = {}
        for strategy, base in self.base_weights.items():
            stats = self.tracker.get_strategy_accuracy(strategy, self.window_days)
            n = stats["n_predictions"]
            sharpe = stats["sharpe"]

            if n < self.min_samples:
                target = base
            elif sharpe > 0.5:
                # Sharpe of 2.0+ pushes to the ceiling.
                boost_frac = min(1.0, (sharpe - 0.5) / 1.5)
                target = base + (self.WEIGHT_MAX - base) * boost_frac
            elif sharpe < 0.0:
                # Sharpe of -1.0 pushes to the floor.
                penalty_frac = min(1.0, -sharpe / 1.0)
                target = base - (base - self.WEIGHT_MIN) * penalty_frac
            else:
                target = base

            # EMA blend with previously stored weight for smoothness.
            prev = self._ema_weights.get(strategy, base)
            blended = prev + self.learning_rate * (target - prev)
            clamped = float(np.clip(blended, self.WEIGHT_MIN, self.WEIGHT_MAX))
            new_weights[strategy] = round(clamped, 4)
            self._ema_weights[strategy] = clamped

        return new_weights

    def apply_adjustments(self, current_weights: dict[str, float]) -> dict[str, float]:
        """Merge computed adjustments into ``current_weights`` and log diffs."""
        adjustments = self.compute_weight_adjustments()
        updated = dict(current_weights)
        changes: dict[str, tuple[float, float]] = {}

        for strategy, new_w in adjustments.items():
            old_w = current_weights.get(strategy, self.base_weights.get(strategy, 1.0))
            if abs(new_w - old_w) >= 0.01:
                changes[strategy] = (round(old_w, 4), new_w)
            updated[strategy] = new_w

        if changes:
            logger.info(
                "strategy_weights_updated",
                window_days=self.window_days,
                changes={k: {"old": v[0], "new": v[1]} for k, v in changes.items()},
            )
        else:
            logger.debug("strategy_weights_unchanged", window_days=self.window_days)

        return updated

    # ── Regime overlay ──────────────────────────────────────────────

    def get_regime_weights(self, regime: str) -> dict[str, float]:
        """Return the current learned weights modulated by market regime.

        ``regime`` should be one of ``"trending"``, ``"ranging"``,
        ``"volatile"``. Unknown regimes fall back to the learned weights
        unchanged.
        """
        learned = self.compute_weight_adjustments()
        multipliers = self._REGIME_MULTIPLIERS.get(regime.lower(), {})
        if not multipliers:
            return learned

        out: dict[str, float] = {}
        for strategy, weight in learned.items():
            mult = multipliers.get(strategy, 1.0)
            adjusted = float(np.clip(weight * mult, self.WEIGHT_MIN, self.WEIGHT_MAX))
            out[strategy] = round(adjusted, 4)

        logger.debug("regime_weights_computed", regime=regime, weights=out)
        return out


# ────────────────────────────────────────────────────────────────────────
# Calibration adjuster
# ────────────────────────────────────────────────────────────────────────


class CalibrationAdjuster:
    """Remap raw confidence through an empirical calibration curve.

    A confidence of 70% should translate into a 70% empirical win rate.
    If the bot is systematically over- or under-confident we learn a
    monotonic mapping from the tracker's history and apply it with
    linear interpolation.
    """

    def __init__(self, tracker: PredictionTracker) -> None:
        self.tracker = tracker
        self._curve: list[tuple[float, float]] = []

    def build_calibration_curve(
        self, n_buckets: int = 10, window_days: int = 90
    ) -> list[tuple[float, float]]:
        """Learn the (predicted, actual) calibration curve.

        Anchors ``(0,0)`` and ``(1,1)`` are always added so interpolation
        is well-defined at the edges. Points are sorted by predicted
        confidence and then smoothed to be monotonic non-decreasing so
        the inverse mapping is stable.
        """
        points = self.tracker.get_confidence_calibration(
            window_days=window_days, n_buckets=n_buckets
        )
        curve: list[tuple[float, float]] = [(0.0, 0.0)]
        for p in points:
            curve.append((float(p["predicted"]), float(p["actual_win_rate"])))
        curve.append((1.0, 1.0))

        # Deduplicate on x and enforce monotonic non-decreasing y.
        curve.sort(key=lambda xy: xy[0])
        deduped: list[tuple[float, float]] = []
        for x, y in curve:
            if deduped and abs(x - deduped[-1][0]) < 1e-9:
                deduped[-1] = (x, max(deduped[-1][1], y))
            else:
                deduped.append((x, y))

        running_max = 0.0
        monotone: list[tuple[float, float]] = []
        for x, y in deduped:
            running_max = max(running_max, y)
            monotone.append((x, running_max))

        self._curve = monotone
        logger.debug("calibration_curve_built", points=len(monotone))
        return monotone

    def adjust_confidence(self, raw_confidence: float) -> float:
        """Map a raw confidence through the learned curve.

        Falls back to the identity function when no curve has been built
        yet (e.g. cold start with no closed predictions).
        """
        raw = float(np.clip(raw_confidence, 0.0, 1.0))
        if not self._curve:
            self.build_calibration_curve()
        if len(self._curve) < 2:
            return raw

        xs = [p[0] for p in self._curve]
        ys = [p[1] for p in self._curve]

        if raw <= xs[0]:
            return float(ys[0])
        if raw >= xs[-1]:
            return float(ys[-1])

        idx = bisect_left(xs, raw)
        x0, x1 = xs[idx - 1], xs[idx]
        y0, y1 = ys[idx - 1], ys[idx]
        if x1 - x0 < 1e-9:
            return float(y1)
        frac = (raw - x0) / (x1 - x0)
        return float(np.clip(y0 + frac * (y1 - y0), 0.0, 1.0))
