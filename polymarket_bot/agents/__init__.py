"""AI council agents package.

Agents analyse different information streams (Trump/political news,
macroeconomics, geopolitics, crypto) and emit ``NarrativeEvent`` objects
consumed by the ``NarrativeEngine``.

Usage::

    from polymarket_bot.agents import CouncilOrchestrator
"""

from __future__ import annotations

# Lazy import keeps startup cost low — the orchestrator pulls in aiohttp
# and all sub-agents only when actually instantiated.
def __getattr__(name: str):  # noqa: N807
    if name == "CouncilOrchestrator":
        from polymarket_bot.agents.orchestrator import CouncilOrchestrator  # noqa: PLC0415
        return CouncilOrchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CouncilOrchestrator"]
