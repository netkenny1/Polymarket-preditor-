"""Base class for all AI council agents."""

from __future__ import annotations

import asyncio
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from polymarket_bot.data.models import NarrativeEvent, NarrativeCategory

logger = structlog.get_logger()

# Common English stop-words used when extracting keywords from text.
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "will", "the", "be", "is", "on", "in", "at", "to", "by", "a",
        "an", "of", "for", "and", "or", "this", "that", "it", "with",
        "from", "has", "have", "was", "were", "been", "are", "what",
        "when", "where", "how", "does", "did", "not", "but", "can",
        "its", "our", "their", "they", "them", "would", "could", "should",
        "also", "which", "who", "any", "all", "said", "just", "more",
    }
)


# ── AgentInsight dataclass ───────────────────────────────────────────────


@dataclass
class AgentInsight:
    """Output produced by one council agent after a full analysis cycle."""

    agent_name: str
    events: list[NarrativeEvent] = field(default_factory=list)
    cross_asset_impact: dict[str, float] = field(default_factory=dict)
    raw_analysis: str = ""
    confidence: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ── BaseAgent ABC ────────────────────────────────────────────────────────


class BaseAgent(ABC):
    """Abstract base for all AI council agents.

    Subclasses implement ``fetch_data`` and ``build_prompt``.  The default
    ``analyze`` orchestration calls both, sends the prompt to Claude, and
    returns a fully populated ``AgentInsight``.
    """

    # Claude model identifiers (override per-agent if needed)
    name: str = "base"
    model: str = "claude-haiku-4-5-20251001"   # fast, cheap — default for all agents
    deep_model: str = "claude-opus-4-6"         # reserved for high-stakes calls

    # Approximate cost per input+output token for Haiku (blended)
    _cost_per_token: float = 0.000_001

    def __init__(self, anthropic_client: Any, config: dict) -> None:
        self._client = anthropic_client   # anthropic.Anthropic | None
        self._config = config

    # ── Abstract interface ───────────────────────────────────────────

    @abstractmethod
    async def fetch_data(self) -> list[dict]:
        """Fetch raw data items from external sources.

        Returns a list of dicts, each containing at minimum a ``text`` key
        with human-readable content plus optional ``source`` and ``url``
        fields.
        """

    @abstractmethod
    def build_prompt(self, data: list[dict]) -> str:
        """Build the Claude prompt from fetched data items."""

    # ── Public analysis entry-point ──────────────────────────────────

    async def analyze(self) -> AgentInsight:
        """Full analysis cycle: fetch → prompt → Claude → parse → insight."""
        try:
            data = await self.fetch_data()
        except Exception as exc:
            logger.warning(
                "agent_fetch_failed", agent=self.name, error=str(exc)
            )
            return self._empty_insight()

        if not data:
            logger.info("agent_no_data", agent=self.name)
            return self._empty_insight()

        prompt = self.build_prompt(data)
        if not prompt:
            return self._empty_insight()

        raw_text, tokens = await self._call_claude(prompt)
        if not raw_text:
            return self._empty_insight()

        parsed = self._parse_json_from_text(raw_text)
        cost = tokens * self._cost_per_token

        events = self._build_events_from_parsed(parsed, data)
        cross_asset = parsed.get("affected_assets", {})
        confidence = float(parsed.get("magnitude", 0.5))

        logger.info(
            "agent_analyzed",
            agent=self.name,
            events=len(events),
            tokens=tokens,
            cost_usd=round(cost, 6),
        )

        return AgentInsight(
            agent_name=self.name,
            events=events,
            cross_asset_impact=cross_asset,
            raw_analysis=raw_text,
            confidence=confidence,
            tokens_used=tokens,
            cost_usd=cost,
        )

    # ── Claude call ──────────────────────────────────────────────────

    async def _call_claude(
        self,
        prompt: str,
        use_deep_model: bool = False,
    ) -> tuple[str, int]:
        """Send *prompt* to Claude and return ``(response_text, tokens_used)``.

        Retries once on rate-limit (HTTP 429).  Returns ``("", 0)`` when
        the Anthropic client is not configured (graceful degradation).
        """
        if self._client is None:
            logger.debug("claude_client_none_skipping", agent=self.name)
            return ("", 0)

        model = self.deep_model if use_deep_model else self.model
        max_retries = 2

        for attempt in range(max_retries):
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._client.messages.create(
                        model=model,
                        max_tokens=1024,
                        messages=[{"role": "user", "content": prompt}],
                    ),
                )
                text = response.content[0].text if response.content else ""
                tokens = (
                    response.usage.input_tokens + response.usage.output_tokens
                    if hasattr(response, "usage")
                    else len(prompt.split()) + len(text.split())
                )
                return (text, tokens)

            except Exception as exc:
                error_str = str(exc)
                is_rate_limit = "429" in error_str or "rate_limit" in error_str.lower()
                if is_rate_limit and attempt < max_retries - 1:
                    wait_seconds = 15 * (attempt + 1)
                    logger.warning(
                        "claude_rate_limited_retrying",
                        agent=self.name,
                        attempt=attempt + 1,
                        wait_seconds=wait_seconds,
                    )
                    await asyncio.sleep(wait_seconds)
                    continue
                logger.error(
                    "claude_call_failed",
                    agent=self.name,
                    model=model,
                    error=error_str,
                )
                return ("", 0)

        return ("", 0)

    # ── Parsing helpers ──────────────────────────────────────────────

    def _parse_json_from_text(self, text: str) -> dict:
        """Extract a JSON object from a Claude response.

        Tries ```json … ``` fenced block first, then bare ``{…}`` object.
        Returns ``{}`` on any failure.
        """
        import json  # local import keeps module startup cheap

        # Try fenced JSON block
        fence_match = re.search(r"```json\s*([\s\S]*?)```", text, re.IGNORECASE)
        if fence_match:
            candidate = fence_match.group(1).strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # Try bare JSON object (first { ... } block)
        brace_match = re.search(r"\{[\s\S]*\}", text)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        logger.debug("json_parse_failed", agent=self.name, text_snippet=text[:120])
        return {}

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract meaningful keywords from text (mirrors NarrativeEngine logic).

        Words must be at least 4 characters and not in the stop-word list.
        Returns the first 10 matches.
        """
        words = re.findall(r"[a-zA-Z]{4,}", text.lower())
        return [w for w in words if w not in _STOP_WORDS][:10]

    # ── Event builder (default, overridden by subclasses) ────────────

    def _build_events_from_parsed(
        self, parsed: dict, raw_data: list[dict]
    ) -> list[NarrativeEvent]:
        """Default: build one generic event from the parsed JSON.

        Subclasses typically override ``analyze()`` directly to produce
        more semantically rich events.
        """
        if not parsed:
            return []

        sentiment = float(parsed.get("sentiment", 0.0))
        magnitude = float(parsed.get("magnitude", 0.5))
        summary = parsed.get(
            "narrative_summary",
            parsed.get("dominant_narrative", "No summary available"),
        )
        keywords = self._extract_keywords(summary)

        event = NarrativeEvent(
            event_id=str(uuid.uuid4())[:12],
            source=self.name,
            content=summary,
            timestamp=datetime.now(timezone.utc),
            category=NarrativeCategory.OTHER,
            sentiment=max(-1.0, min(1.0, sentiment)),
            magnitude=max(0.0, min(1.0, magnitude)),
            keywords=keywords,
        )
        return [event]

    # ── Convenience factory ──────────────────────────────────────────

    def _empty_insight(self) -> AgentInsight:
        """Return an AgentInsight with no events (used for error paths)."""
        return AgentInsight(agent_name=self.name)
