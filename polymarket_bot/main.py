"""CLI dispatcher for the BTC 5-minute Polymarket bot.

Subcommands:

    python -m polymarket_bot backtest [--days 7] [--lag 30] [--capital 100]
        Run the BTC 5-min strategy against real Binance historical data.

    python -m polymarket_bot paper
        Run the live loop in paper-trading mode (no real orders).

    python -m polymarket_bot live
        Run the live loop and submit real POST_ONLY orders via the CLOB.
        Requires PRIVATE_KEY + POLY_FUNDER env vars.

This file is deliberately tiny — the interesting code lives in
`polymarket_bot.backtest.runner`, `polymarket_bot.bot`, and
`polymarket_bot.strategies.btc_5min`.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Optional

import structlog

from polymarket_bot.backtest.runner import BacktestConfig, run_backtest
from polymarket_bot.bot import LiveBot
from polymarket_bot.config import BotConfig


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(message)s",
        level=level,
        stream=sys.stdout,
    )
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(level),
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
    )


# ── Subcommand handlers ───────────────────────────────────────────────

def _cmd_backtest(args: argparse.Namespace) -> int:
    cfg = BacktestConfig(
        days=args.days,
        initial_capital=args.capital,
        lag_seconds=args.lag,
        spread_cents=args.spread,
        min_edge=args.min_edge,
        kelly_fraction=args.kelly,
        max_position_usd=args.max_position,
        verbose=args.verbose,
    )
    result = run_backtest(cfg)
    print(result.summary())

    if args.show_trades and result.trades:
        print("\nRecent trades:")
        for t in result.trades[-20:]:
            print("  ", t.as_dict())
    return 0


def _cmd_paper(args: argparse.Namespace) -> int:
    os.environ["PAPER_TRADING"] = "true"
    config = BotConfig.from_env()
    bot = LiveBot(config)
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        bot.stop()
    return 0


def _cmd_live(args: argparse.Namespace) -> int:
    if not os.getenv("PRIVATE_KEY"):
        print(
            "ERROR: live mode requires PRIVATE_KEY (and typically POLY_FUNDER) "
            "env vars. Use `paper` to run without a funded wallet.",
            file=sys.stderr,
        )
        return 2
    os.environ["PAPER_TRADING"] = "false"
    config = BotConfig.from_env()
    bot = LiveBot(config)
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        bot.stop()
    return 0


# ── Arg parser ────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polymarket_bot",
        description="BTC 5-minute Polymarket bot (Black-Scholes + Kelly)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    sub = parser.add_subparsers(dest="command", required=True)

    # backtest
    p_bt = sub.add_parser("backtest", help="Replay the strategy on real Binance klines")
    p_bt.add_argument("--days", type=int, default=7, help="Days of history to replay")
    p_bt.add_argument("--capital", type=float, default=100.0, help="Initial capital USD")
    p_bt.add_argument("--lag", type=int, default=30,
                      help="Assumed Polymarket MM lag in seconds (edge source)")
    p_bt.add_argument("--spread", type=float, default=0.02,
                      help="Assumed YES/NO bid-ask spread in cents")
    p_bt.add_argument("--min-edge", type=float, default=0.03,
                      help="Min fair-vs-market edge to open")
    p_bt.add_argument("--kelly", type=float, default=0.25,
                      help="Kelly fraction (quarter-Kelly default)")
    p_bt.add_argument("--max-position", type=float, default=25.0,
                      help="Max USD per trade")
    p_bt.add_argument("--show-trades", action="store_true",
                      help="Print the last 20 trades")
    p_bt.set_defaults(func=_cmd_backtest)

    # paper
    p_paper = sub.add_parser("paper", help="Paper-trade the live 5-min market")
    p_paper.set_defaults(func=_cmd_paper)

    # live
    p_live = sub.add_parser("live", help="Submit REAL POST_ONLY orders (requires PRIVATE_KEY)")
    p_live.set_defaults(func=_cmd_live)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
