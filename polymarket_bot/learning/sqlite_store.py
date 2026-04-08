"""SQLite persistence layer for the self-improvement loop.

Provides durable storage for predictions, strategy weights, pattern weights,
agent insights, and narrative history — so the bot learns across restarts.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prediction_log (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    strategy TEXT NOT NULL,
    market_condition_id TEXT NOT NULL,
    token_id TEXT NOT NULL,
    side TEXT NOT NULL,
    narrative_category TEXT,
    entry_price REAL NOT NULL,
    predicted_fair_value REAL NOT NULL,
    predicted_edge REAL NOT NULL,
    confidence REAL NOT NULL,
    size_usd REAL NOT NULL,
    exit_price REAL,
    realized_pnl REAL,
    correct INTEGER,
    closed INTEGER NOT NULL DEFAULT 0,
    pattern_id TEXT
);

CREATE TABLE IF NOT EXISTS strategy_weights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    strategy TEXT NOT NULL,
    weight REAL NOT NULL,
    sharpe_30d REAL,
    win_rate_30d REAL,
    n_predictions INTEGER
);

CREATE TABLE IF NOT EXISTS pattern_weights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    pattern_id TEXT NOT NULL,
    weight REAL NOT NULL,
    accuracy_30d REAL,
    n_matches INTEGER NOT NULL DEFAULT 0,
    auto_discovered INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS agent_insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    raw_analysis TEXT NOT NULL,
    events_generated INTEGER NOT NULL DEFAULT 0,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS narrative_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    narrative_id TEXT NOT NULL,
    category TEXT NOT NULL,
    strength REAL NOT NULL,
    predicted_direction REAL NOT NULL,
    outcome_correct INTEGER,
    events_count INTEGER NOT NULL DEFAULT 0,
    historical_analog TEXT,
    current_phase TEXT
);

CREATE INDEX IF NOT EXISTS idx_pred_strategy ON prediction_log(strategy);
CREATE INDEX IF NOT EXISTS idx_pred_category ON prediction_log(narrative_category);
CREATE INDEX IF NOT EXISTS idx_pred_closed ON prediction_log(closed);
CREATE INDEX IF NOT EXISTS idx_pred_timestamp ON prediction_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_sw_strategy ON strategy_weights(strategy);
CREATE INDEX IF NOT EXISTS idx_ai_agent ON agent_insights(agent_name);
CREATE INDEX IF NOT EXISTS idx_nh_category ON narrative_history(category);
"""


class SQLiteStore:
    """Thread-safe SQLite persistence for the self-improvement loop."""

    def __init__(self, db_path: str = "bot_data.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._path),
            check_same_thread=False,
            isolation_level=None,  # autocommit
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        logger.info("sqlite_store_initialized", path=str(self._path))

    def close(self) -> None:
        self._conn.close()

    # ── Predictions ──────────────────────────────────────────────────

    def save_prediction(self, record: Any) -> None:
        """Insert a new open prediction (PredictionRecord or dict)."""
        try:
            d = record.to_dict() if hasattr(record, "to_dict") else dict(record)
            self._conn.execute(
                """
                INSERT OR REPLACE INTO prediction_log
                (id, timestamp, strategy, market_condition_id, token_id, side,
                 narrative_category, entry_price, predicted_fair_value, predicted_edge,
                 confidence, size_usd, exit_price, realized_pnl, correct, closed, pattern_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    d.get("prediction_id", ""),
                    d.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    d.get("strategy", ""),
                    d.get("market_condition_id", ""),
                    d.get("token_id", ""),
                    d.get("side", ""),
                    d.get("narrative_category"),
                    d.get("entry_price", 0.0),
                    d.get("predicted_fair_value", 0.0),
                    d.get("predicted_edge", 0.0),
                    d.get("confidence", 0.0),
                    d.get("size_usd", 0.0),
                    d.get("exit_price"),
                    d.get("realized_pnl"),
                    int(d["correct"]) if d.get("correct") is not None else None,
                    int(d.get("closed", False)),
                    d.get("pattern_id"),
                ),
            )
        except Exception as exc:
            logger.warning("sqlite_save_prediction_failed", error=str(exc))

    def close_prediction(
        self,
        prediction_id: str,
        exit_price: float,
        pnl: float,
        correct: bool | None = None,
    ) -> None:
        """Mark a prediction as closed with outcome data."""
        try:
            self._conn.execute(
                """
                UPDATE prediction_log
                SET exit_price=?, realized_pnl=?, correct=?, closed=1
                WHERE id=?
                """,
                (exit_price, pnl, int(correct) if correct is not None else None, prediction_id),
            )
        except Exception as exc:
            logger.warning("sqlite_close_prediction_failed", error=str(exc))

    def get_strategy_stats(self, strategy: str, window_days: int = 30) -> dict[str, Any]:
        """Compute win rate, avg PnL, and count for a strategy over a window."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
        try:
            rows = self._conn.execute(
                """
                SELECT correct, realized_pnl FROM prediction_log
                WHERE strategy=? AND closed=1 AND timestamp>=?
                """,
                (strategy, cutoff),
            ).fetchall()
            if not rows:
                return {"n": 0, "win_rate": 0.0, "avg_pnl": 0.0}
            corrects = [r["correct"] for r in rows if r["correct"] is not None]
            pnls = [r["realized_pnl"] or 0.0 for r in rows]
            win_rate = sum(corrects) / len(corrects) if corrects else 0.0
            avg_pnl = sum(pnls) / len(pnls) if pnls else 0.0
            return {"n": len(rows), "win_rate": round(win_rate, 4), "avg_pnl": round(avg_pnl, 4)}
        except Exception as exc:
            logger.warning("sqlite_get_strategy_stats_failed", error=str(exc))
            return {"n": 0, "win_rate": 0.0, "avg_pnl": 0.0}

    # ── Strategy weights ─────────────────────────────────────────────

    def save_strategy_weights(
        self,
        weights: dict[str, float],
        stats: dict[str, Any] | None = None,
    ) -> None:
        """Persist the current live strategy weight snapshot."""
        ts = datetime.now(timezone.utc).isoformat()
        stats = stats or {}
        try:
            for strategy, weight in weights.items():
                s = stats.get(strategy, {})
                self._conn.execute(
                    """
                    INSERT INTO strategy_weights
                    (timestamp, strategy, weight, sharpe_30d, win_rate_30d, n_predictions)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (ts, strategy, weight, s.get("sharpe"), s.get("win_rate"), s.get("n")),
                )
        except Exception as exc:
            logger.warning("sqlite_save_weights_failed", error=str(exc))

    def load_latest_weights(self) -> dict[str, float]:
        """Restore the most recent strategy weight snapshot."""
        try:
            rows = self._conn.execute(
                """
                SELECT strategy, weight FROM strategy_weights
                WHERE timestamp = (SELECT MAX(timestamp) FROM strategy_weights)
                """
            ).fetchall()
            return {r["strategy"]: r["weight"] for r in rows}
        except Exception as exc:
            logger.warning("sqlite_load_weights_failed", error=str(exc))
            return {}

    # ── Pattern weights ──────────────────────────────────────────────

    def save_pattern_weight(
        self,
        pattern_id: str,
        weight: float,
        accuracy: float | None = None,
        n_matches: int = 0,
        auto_discovered: bool = False,
    ) -> None:
        try:
            self._conn.execute(
                """
                INSERT INTO pattern_weights
                (timestamp, pattern_id, weight, accuracy_30d, n_matches, auto_discovered)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    pattern_id, weight, accuracy, n_matches, int(auto_discovered),
                ),
            )
        except Exception as exc:
            logger.warning("sqlite_save_pattern_weight_failed", error=str(exc))

    def load_pattern_weights(self) -> dict[str, float]:
        """Load the latest weight per pattern."""
        try:
            rows = self._conn.execute(
                """
                SELECT pw1.pattern_id, pw1.weight FROM pattern_weights pw1
                INNER JOIN (
                    SELECT pattern_id, MAX(timestamp) as max_ts
                    FROM pattern_weights GROUP BY pattern_id
                ) pw2 ON pw1.pattern_id = pw2.pattern_id AND pw1.timestamp = pw2.max_ts
                """
            ).fetchall()
            return {r["pattern_id"]: r["weight"] for r in rows}
        except Exception as exc:
            logger.warning("sqlite_load_pattern_weights_failed", error=str(exc))
            return {}

    # ── Agent insights ───────────────────────────────────────────────

    def save_agent_insight(
        self,
        agent_name: str,
        raw_analysis: str,
        events_generated: int,
        tokens_used: int,
        cost_usd: float,
    ) -> None:
        try:
            self._conn.execute(
                """
                INSERT INTO agent_insights
                (timestamp, agent_name, raw_analysis, events_generated, tokens_used, cost_usd)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    agent_name,
                    raw_analysis[:10000],  # cap to avoid huge blobs
                    events_generated,
                    tokens_used,
                    cost_usd,
                ),
            )
        except Exception as exc:
            logger.warning("sqlite_save_insight_failed", error=str(exc))

    def get_agent_costs(self, days_back: int = 1) -> dict[str, float]:
        """Return total Claude spend per agent for the last N days."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
        try:
            rows = self._conn.execute(
                """
                SELECT agent_name, SUM(cost_usd) as total
                FROM agent_insights
                WHERE timestamp >= ?
                GROUP BY agent_name
                """,
                (cutoff,),
            ).fetchall()
            return {r["agent_name"]: round(r["total"] or 0.0, 6) for r in rows}
        except Exception as exc:
            logger.warning("sqlite_get_agent_costs_failed", error=str(exc))
            return {}

    def get_total_daily_cost(self) -> float:
        """Return total Claude spend today (UTC)."""
        costs = self.get_agent_costs(days_back=1)
        return round(sum(costs.values()), 6)

    # ── Narrative history ────────────────────────────────────────────

    def save_narrative_record(
        self,
        narrative_id: str,
        category: str,
        strength: float,
        direction: float,
        events_count: int,
        historical_analog: str | None = None,
        current_phase: str | None = None,
    ) -> None:
        try:
            self._conn.execute(
                """
                INSERT INTO narrative_history
                (timestamp, narrative_id, category, strength, predicted_direction,
                 events_count, historical_analog, current_phase)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    narrative_id,
                    category,
                    strength,
                    direction,
                    events_count,
                    historical_analog,
                    current_phase,
                ),
            )
        except Exception as exc:
            logger.warning("sqlite_save_narrative_failed", error=str(exc))

    def get_narrative_history(self, days_back: int = 30) -> list[dict[str, Any]]:
        """Return recent narrative records for pattern discovery."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
        try:
            rows = self._conn.execute(
                """
                SELECT * FROM narrative_history
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT 500
                """,
                (cutoff,),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("sqlite_get_narrative_history_failed", error=str(exc))
            return []
