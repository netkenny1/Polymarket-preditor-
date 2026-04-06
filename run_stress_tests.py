#!/usr/bin/env python3
"""Stress-test runner — exercises the trading bot under 6 market regimes.

For each regime the MarketSimulator methods are monkey-patched so that
create_simulated_markets, step_prices, and generate_order_book use
regime-specific volatility, mean-reversion, spread, and true-prob parameters.

Usage:
    python run_stress_tests.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import structlog

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(50))  # CRITICAL only

from polymarket_bot.backtesting.simulator import MarketSimulator, SimulatedMarket
from polymarket_bot.data.models import OrderBook, OrderBookLevel
from run_monte_carlo import run_single_backtest, DetailedResult

# ---------------------------------------------------------------------------
# Regime definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Regime:
    name: str
    vol_lo: float
    vol_hi: float
    mean_reversion: float
    spread_lo: float
    spread_hi: float
    true_prob_lo: float
    true_prob_hi: float


REGIMES: list[Regime] = [
    Regime("BULL",            0.01, 0.03,  0.010, 0.01, 0.03, 0.60, 0.90),
    Regime("BEAR",            0.04, 0.10,  0.002, 0.03, 0.08, 0.10, 0.40),
    Regime("SIDEWAYS",        0.005, 0.015, 0.020, 0.01, 0.02, 0.35, 0.65),
    Regime("CRASH",           0.08, 0.15,  0.001, 0.05, 0.10, 0.05, 0.95),
    Regime("HIGH_VOLATILITY", 0.06, 0.12,  0.005, 0.02, 0.06, 0.10, 0.90),
    Regime("DEFAULT",         0.0,  0.0,   0.0,   0.0,  0.0,  0.0,  0.0),   # sentinel — use originals
]

SEEDS = list(range(1, 6))

# ---------------------------------------------------------------------------
# Save originals
# ---------------------------------------------------------------------------
_orig_create = MarketSimulator.create_simulated_markets
_orig_step   = MarketSimulator.step_prices
_orig_book   = MarketSimulator.generate_order_book

# ---------------------------------------------------------------------------
# Patched methods
# ---------------------------------------------------------------------------

def _make_patched_create(regime: Regime) -> Callable:
    """Return a patched create_simulated_markets that applies regime params."""

    def patched_create(self: MarketSimulator, count: int = 10) -> list[SimulatedMarket]:
        from datetime import datetime, timedelta
        from polymarket_bot.data.models import Market, MarketCategory, Token

        markets = []
        categories = [
            ("crypto", MarketCategory.CRYPTO, [
                "Will Bitcoin be above $100,000 by end of Q2?",
                "Will Ethereum reach $5,000 this month?",
                "Will Solana flip Ethereum in daily volume?",
            ]),
            ("politics", MarketCategory.POLITICS, [
                "Will the incumbent win the next presidential election?",
                "Will the Senate pass the infrastructure bill?",
                "Will the governor win re-election?",
            ]),
            ("sports", MarketCategory.SPORTS, [
                "Will the Lakers win the NBA Championship?",
                "Will Team A beat Team B in the finals?",
                "Will the underdog win the Super Bowl?",
            ]),
        ]

        for i in range(count):
            cat_name, cat_enum, questions = categories[i % len(categories)]
            question = questions[i % len(questions)]

            raw = self.rng.beta(2, 2)
            true_prob = regime.true_prob_lo + raw * (regime.true_prob_hi - regime.true_prob_lo)
            true_prob = max(0.05, min(0.95, true_prob))

            noise = self.rng.normal(0, 0.08)
            initial_price = max(0.05, min(0.95, true_prob + noise))

            market = Market(
                condition_id=f"sim_{i:04d}",
                question=question,
                slug=f"sim-market-{i}",
                tokens=[
                    Token(token_id=f"sim_{i:04d}_yes", outcome="Yes", price=initial_price),
                    Token(token_id=f"sim_{i:04d}_no", outcome="No", price=1.0 - initial_price),
                ],
                category=cat_enum,
                end_date=datetime.utcnow() + timedelta(days=30),
                volume_24h=self.rng.uniform(1000, 100000),
                liquidity=self.rng.uniform(500, 50000),
                active=True,
                tags=[cat_name],
            )

            sim = SimulatedMarket(
                market=market,
                true_probability=true_prob,
                volatility=self.rng.uniform(regime.vol_lo, regime.vol_hi),
                price_history=[initial_price],
            )
            markets.append(sim)

        return markets

    return patched_create


def _make_patched_step(regime: Regime) -> Callable:
    """Return a patched step_prices with regime mean-reversion."""

    def patched_step(self: MarketSimulator, sim_markets: list[SimulatedMarket]) -> None:
        for sim in sim_markets:
            current = sim.price_history[-1]
            reversion = regime.mean_reversion * (sim.true_probability - current)
            noise = self.rng.normal(0, sim.volatility)
            new_price = current + reversion + noise
            new_price = max(0.02, min(0.98, new_price))
            sim.price_history.append(new_price)
            for token in sim.market.tokens:
                if token.outcome == "Yes":
                    token.price = new_price
                else:
                    token.price = 1.0 - new_price

    return patched_step


def _make_patched_book(regime: Regime) -> Callable:
    """Return a patched generate_order_book with regime spread."""

    def patched_book(
        self: MarketSimulator, sim: SimulatedMarket, depth: int = 5
    ) -> dict[str, OrderBook]:
        books = {}
        for token in sim.market.tokens:
            mid = token.price
            spread = self.rng.uniform(regime.spread_lo, regime.spread_hi)
            bids, asks = [], []
            for i in range(depth):
                bid_price = max(0.01, mid - spread / 2 - i * 0.01)
                ask_price = min(0.99, mid + spread / 2 + i * 0.01)
                bid_size = self.rng.uniform(10, 200) * (depth - i) / depth
                ask_size = self.rng.uniform(10, 200) * (depth - i) / depth
                bids.append(OrderBookLevel(price=round(bid_price, 2), size=round(bid_size, 2)))
                asks.append(OrderBookLevel(price=round(ask_price, 2), size=round(ask_size, 2)))
            bids.sort(key=lambda x: x.price, reverse=True)
            asks.sort(key=lambda x: x.price)
            books[token.token_id] = OrderBook(token_id=token.token_id, bids=bids, asks=asks)
        return books

    return patched_book


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_regime(regime: Regime) -> None:
    if regime.name == "DEFAULT":
        _restore_originals()
        return
    MarketSimulator.create_simulated_markets = _make_patched_create(regime)
    MarketSimulator.step_prices = _make_patched_step(regime)
    MarketSimulator.generate_order_book = _make_patched_book(regime)


def _restore_originals() -> None:
    MarketSimulator.create_simulated_markets = _orig_create
    MarketSimulator.step_prices = _orig_step
    MarketSimulator.generate_order_book = _orig_book


@dataclass
class RegimeStats:
    name: str
    seeds: list[int]
    returns: list[float]
    sharpes: list[float]
    max_dds: list[float]
    win_rates: list[float]
    trade_counts: list[int]
    final_values: list[float]


def _run_regime(regime: Regime) -> RegimeStats:
    _apply_regime(regime)
    stats = RegimeStats(
        name=regime.name, seeds=[], returns=[], sharpes=[],
        max_dds=[], win_rates=[], trade_counts=[], final_values=[],
    )
    for seed in SEEDS:
        dr = run_single_backtest(seed)
        r = dr.result
        stats.seeds.append(seed)
        stats.returns.append(r.total_return_pct)
        stats.sharpes.append(r.sharpe_ratio)
        stats.max_dds.append(r.max_drawdown_pct)
        stats.win_rates.append(r.win_rate)
        stats.trade_counts.append(r.num_trades)
        stats.final_values.append(r.final_portfolio_value)
    _restore_originals()
    return stats


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_W = 72

def _print_regime_table(rs: RegimeStats) -> None:
    print(f"\n{'─' * _W}")
    print(f"  REGIME: {rs.name}")
    print(f"{'─' * _W}")
    hdr = (
        f"  {'Seed':>4s}  {'Final$':>9s}  {'Return':>8s}  {'Sharpe':>7s}  "
        f"{'MaxDD':>7s}  {'WinRate':>7s}  {'Trades':>6s}"
    )
    print(hdr)
    print(f"  {'─' * 4}  {'─' * 9}  {'─' * 8}  {'─' * 7}  {'─' * 7}  {'─' * 7}  {'─' * 6}")
    for i, seed in enumerate(rs.seeds):
        print(
            f"  {seed:>4d}  ${rs.final_values[i]:>8.2f}  {rs.returns[i]:>+7.1%}  "
            f"{rs.sharpes[i]:>7.2f}  {rs.max_dds[i]:>6.1%}  "
            f"{rs.win_rates[i]:>6.0%}  {rs.trade_counts[i]:>6d}"
        )
    print(f"  {'─' * 4}  {'─' * 9}  {'─' * 8}  {'─' * 7}  {'─' * 7}  {'─' * 7}  {'─' * 6}")
    print(
        f"  {'AVG':>4s}  ${np.mean(rs.final_values):>8.2f}  {np.mean(rs.returns):>+7.1%}  "
        f"{np.mean(rs.sharpes):>7.2f}  {np.mean(rs.max_dds):>6.1%}  "
        f"{np.mean(rs.win_rates):>6.0%}  {int(np.mean(rs.trade_counts)):>6d}"
    )


def _print_comparison(all_stats: list[RegimeStats]) -> None:
    print(f"\n{'=' * _W}")
    print(f"  REGIME COMPARISON (5 seeds each)")
    print(f"{'=' * _W}")
    print(
        f"  {'Regime':<16s} {'MeanRet':>8s} {'MedRet':>8s} {'Sharpe':>7s} "
        f"{'MaxDD':>7s} {'WinRate':>7s} {'Trades':>7s} {'ProfSd':>7s}"
    )
    print(
        f"  {'─' * 16} {'─' * 8} {'─' * 8} {'─' * 7} "
        f"{'─' * 7} {'─' * 7} {'─' * 7} {'─' * 7}"
    )
    for rs in all_stats:
        prof = sum(1 for r in rs.returns if r > 0)
        print(
            f"  {rs.name:<16s} {np.mean(rs.returns):>+7.1%} {np.median(rs.returns):>+7.1%} "
            f"{np.mean(rs.sharpes):>7.2f} {np.mean(rs.max_dds):>6.1%} "
            f"{np.mean(rs.win_rates):>6.0%} {int(np.mean(rs.trade_counts)):>7d} "
            f"{prof}/{len(rs.seeds)}"
        )
    print()

    print(f"  {'Regime':<16s} {'BestRet':>8s} {'WorstRet':>9s} {'StdRet':>8s} {'WorstDD':>8s}")
    print(f"  {'─' * 16} {'─' * 8} {'─' * 9} {'─' * 8} {'─' * 8}")
    for rs in all_stats:
        print(
            f"  {rs.name:<16s} {max(rs.returns):>+7.1%} {min(rs.returns):>+8.1%} "
            f"{np.std(rs.returns):>7.1%} {max(rs.max_dds):>7.1%}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"\n{'=' * _W}")
    print(f"  STRESS TEST — POLYMARKET TRADING BOT")
    print(f"  6 market regimes × 5 seeds = 30 simulations")
    print(f"{'=' * _W}\n")

    all_stats: list[RegimeStats] = []
    t_total = time.time()

    for regime in REGIMES:
        t0 = time.time()
        print(f"  Running regime {regime.name} ...", end="", flush=True)
        rs = _run_regime(regime)
        elapsed = time.time() - t0
        print(f" done in {elapsed:.1f}s")
        _print_regime_table(rs)
        all_stats.append(rs)

    total_elapsed = time.time() - t_total

    _print_comparison(all_stats)

    print(f"  Total runtime: {total_elapsed:.1f}s "
          f"({total_elapsed / (len(REGIMES) * len(SEEDS)):.1f}s per sim)")
    print(f"{'=' * _W}\n")


if __name__ == "__main__":
    main()
