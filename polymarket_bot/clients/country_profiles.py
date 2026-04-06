"""Country-level economic profiles and free RSS headline ingestion for macro context."""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from xml.etree import ElementTree as ET

import httpx
import structlog

logger = structlog.get_logger()

# ─── Economic status ordering (higher = worse) ─────────────────────────
_STATUS_RANK: dict[str, int] = {
    "strong": 0,
    "weakening": 1,
    "recession": 2,
    "crisis": 3,
}
_STATUS_BY_RANK: dict[int, str] = {v: k for k, v in _STATUS_RANK.items()}

_CRISIS_WORDS = frozenset(
    {
        "war",
        "invasion",
        "missile",
        "strike",
        "sanction",
        "default",
        "bank run",
        "collapse",
        "escalat",
        "terror",
        "embargo",
        "blackout",
    }
)
_STRESS_WORDS = frozenset(
    {
        "recession",
        "slowdown",
        "crisis",
        "tariff",
        "inflation",
        "unrest",
        "protest",
        "coup",
        "sanctions",
        "downgrade",
    }
)
_RELIEF_WORDS = frozenset(
    {
        "deal",
        "ceasefire",
        "truce",
        "recovery",
        "expansion",
        "cut rates",
        "rate cut",
        "breakthrough",
        "eased",
        "optimism",
    }
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _strip_html(text: str) -> str:
    if not text:
        return ""
    t = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", t).strip()


def _parse_pub_date(raw: str | None) -> datetime | None:
    if not raw:
        return None
    raw = raw.strip()
    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        pass
    try:
        if raw.endswith("Z"):
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
        return datetime.fromisoformat(raw).astimezone(timezone.utc)
    except ValueError:
        return None


def _parse_rss_items(xml_text: str) -> list[dict[str, Any]]:
    """Parse RSS 2.0 or Atom into dicts with title, description, pub_date."""
    out: list[dict[str, Any]] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.warning("rss_parse_error", error=str(e))
        return out

    tag = root.tag.split("}")[-1].lower()
    if tag == "rss" or root.find("channel") is not None:
        ch = root.find("channel")
        if ch is None:
            return out
        for item in ch.findall("item"):
            title = _strip_html((item.findtext("title") or "").strip())
            desc = _strip_html((item.findtext("description") or "").strip())
            pub_raw = item.findtext("pubDate")
            out.append(
                {
                    "title": title,
                    "description": desc,
                    "pub_date": _parse_pub_date(pub_raw),
                }
            )
        return out

    if tag == "feed":
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("a:entry", ns):
            title_el = entry.find("a:title", ns)
            title = _strip_html((title_el.text or "").strip() if title_el is not None else "")
            summary_el = entry.find("a:summary", ns)
            content_el = entry.find("a:content", ns)
            desc = ""
            if summary_el is not None and summary_el.text:
                desc = _strip_html(summary_el.text.strip())
            elif content_el is not None and content_el.text:
                desc = _strip_html(content_el.text.strip())
            pub_el = entry.find("a:updated", ns) or entry.find("a:published", ns)
            pub_raw = pub_el.text if pub_el is not None else None
            out.append(
                {
                    "title": title,
                    "description": desc,
                    "pub_date": _parse_pub_date(pub_raw),
                }
            )
    return out


@dataclass
class CountryProfile:
    country_code: str
    country_name: str
    economic_status: str
    gdp_growth_pct: float
    inflation_pct: float
    debt_to_gdp_pct: float
    currency_strength: float
    trade_balance_bn: float
    key_risks: list[str]
    key_opportunities: list[str]
    geopolitical_alignment: str
    last_updated: datetime
    news_summary: str


# Keywords → country_code (lowercase matching on headline)
_COUNTRY_KEYWORDS: dict[str, list[str]] = {
    "US": [
        "united states",
        "u.s.",
        "u.s.a",
        "america",
        "american",
        "washington",
        "white house",
        "federal reserve",
        "fed ",
        "powell",
        "trump",
        "dollar",
        "usd",
        "treasury",
        "congress",
    ],
    "CN": [
        "china",
        "chinese",
        "beijing",
        "xi jinping",
        "xi ",
        "yuan",
        "renminbi",
        "rmb",
        "cny",
        "prc",
    ],
    "SA": [
        "saudi",
        "riyadh",
        "mbs",
        "mohammed bin salman",
        "opec",
        "petrodollar",
        "gcc",
    ],
    "IR": [
        "iran",
        "iranian",
        "tehran",
        "khamenei",
        "irgc",
        "persian gulf",
    ],
    "IL": [
        "israel",
        "israeli",
        "netanyahu",
        "idf",
        "gaza",
        "tel aviv",
    ],
    "RU": [
        "russia",
        "russian",
        "moscow",
        "putin",
        "kremlin",
        "ruble",
        "rouble",
    ],
    "DE": [
        "germany",
        "german",
        "berlin",
        "scholz",
        "eurozone",
        "ecb",
        "european central bank",
        "eu ",
        "e.u.",
        "european union",
        "brussels",
    ],
    "UK": [
        "britain",
        "british",
        "uk ",
        "u.k.",
        "united kingdom",
        "london",
        "starmer",
        "brexit",
        "pound sterling",
        "gbp",
        "bank of england",
    ],
    "JP": [
        "japan",
        "japanese",
        "tokyo",
        "yen",
        "jpy",
        "boj",
        "bank of japan",
        "kishida",
    ],
    "IN": [
        "india",
        "indian",
        "new delhi",
        "modi",
        "rupee",
        "inr",
    ],
    "BR": [
        "brazil",
        "brazilian",
        "brasilia",
        "lula",
        "real",
        "brl",
    ],
    "TR": [
        "turkey",
        "turkish",
        "ankara",
        "erdogan",
        "lira",
        "try",
    ],
    "KR": [
        "south korea",
        "korean",
        "seoul",
        "won",
        "krw",
    ],
    "TW": [
        "taiwan",
        "taiwanese",
        "taipei",
        "tsmc",
        "strait",
    ],
    "AR": [
        "argentina",
        "argentinian",
        "buenos aires",
        "milei",
        "peso",
    ],
}


def _default_profiles() -> dict[str, CountryProfile]:
    now = _utcnow()
    return {
        "US": CountryProfile(
            country_code="US",
            country_name="United States",
            economic_status="weakening",
            gdp_growth_pct=1.4,
            inflation_pct=3.1,
            debt_to_gdp_pct=124.0,
            currency_strength=0.55,
            trade_balance_bn=-920.0,
            key_risks=[
                "Rolling tariff shocks and trade-policy whiplash",
                "Fiscal trajectory and debt-service crowding out",
                "Dollar centrality under multipolar settlement pressure",
            ],
            key_opportunities=[
                "AI capex cycle and reshoring subsidies",
                "Energy independence buffer vs. Middle East shocks",
            ],
            geopolitical_alignment="western",
            last_updated=now,
            news_summary=(
                "April 2026: growth softening amid tariff chaos; real rates still tight; "
                "dollar strong but policy uncertainty erodes safe-haven premium."
            ),
        ),
        "CN": CountryProfile(
            country_code="CN",
            country_name="China",
            economic_status="weakening",
            gdp_growth_pct=4.2,
            inflation_pct=0.6,
            debt_to_gdp_pct=308.0,
            currency_strength=-0.15,
            trade_balance_bn=820.0,
            key_risks=[
                "Property overhang and local-government debt",
                "Export hit from Western tariffs and tech curbs",
                "Demographics and deflationary bias",
            ],
            key_opportunities=[
                "Yuan internationalization and BRICS settlement rails",
                "EV, batteries, and industrial goods market share",
            ],
            geopolitical_alignment="brics",
            last_updated=now,
            news_summary=(
                "Retaliatory tariff posture; growth slowing from pre-COVID pace; "
                "push to diversify reserves and trade away from USD clearing."
            ),
        ),
        "SA": CountryProfile(
            country_code="SA",
            country_name="Saudi Arabia",
            economic_status="weakening",
            gdp_growth_pct=2.0,
            inflation_pct=2.4,
            debt_to_gdp_pct=22.0,
            currency_strength=0.05,
            trade_balance_bn=168.0,
            key_risks=[
                "Oil-demand elasticity if global recession deepens",
                "Security exposure along Red Sea/Gulf lanes",
                "Fiscal breakeven oil still above spot stress scenarios",
            ],
            key_opportunities=[
                "BRICS+ alignment and non-dollar oil invoicing pilots",
                "Strategic leverage as swing producer",
            ],
            geopolitical_alignment="brics",
            last_updated=now,
            news_summary=(
                "Petrodollar recycling under strain; deeper ties to China/India settlement; "
                "oil as implicit geopolitical lever amid Mideast escalation."
            ),
        ),
        "IR": CountryProfile(
            country_code="IR",
            country_name="Iran",
            economic_status="crisis",
            gdp_growth_pct=-2.8,
            inflation_pct=38.0,
            debt_to_gdp_pct=45.0,
            currency_strength=-0.82,
            trade_balance_bn=12.0,
            key_risks=[
                "Direct military escalation with Israel/US",
                "Sanctions on energy and finance",
                "Domestic unrest under price shocks",
            ],
            key_opportunities=[
                "Proxy leverage reshaping regional order",
                "Underground trade and crypto rails (sanctions evasion)",
            ],
            geopolitical_alignment="brics",
            last_updated=now,
            news_summary=(
                "Proxy-war intensity rising; nuclear/sanctions diplomacy stalled; "
                "economy under severe external pressure."
            ),
        ),
        "IL": CountryProfile(
            country_code="IL",
            country_name="Israel",
            economic_status="recession",
            gdp_growth_pct=-1.2,
            inflation_pct=4.8,
            debt_to_gdp_pct=64.0,
            currency_strength=-0.35,
            trade_balance_bn=-18.0,
            key_risks=[
                "Prolonged multi-front conflict and mobilization costs",
                "Sovereign rating and funding stress",
                "Regional escalation drawing great-power involvement",
            ],
            key_opportunities=[
                "Defense-tech export demand",
                "Energy security if offshore gas flows stabilize",
            ],
            geopolitical_alignment="western",
            last_updated=now,
            news_summary=(
                "Active conflict; defense spending surge; growth shock; "
                "security premium dominates macro."
            ),
        ),
        "RU": CountryProfile(
            country_code="RU",
            country_name="Russia",
            economic_status="weakening",
            gdp_growth_pct=1.1,
            inflation_pct=8.4,
            debt_to_gdp_pct=19.0,
            currency_strength=-0.45,
            trade_balance_bn=118.0,
            key_risks=[
                "Sanctions leakage vs. enforcement tightening",
                "Military spend crowding out consumption",
                "Energy price volatility",
            ],
            key_opportunities=[
                "BRICS chairmanship and commodity pivot to Asia",
                "War economy industrial utilization",
            ],
            geopolitical_alignment="brics",
            last_updated=now,
            news_summary=(
                "Sanctions-adapted economy; energy leverage in Asia; "
                "positioning as BRICS anchor amid Western isolation."
            ),
        ),
        "DE": CountryProfile(
            country_code="DE",
            country_name="Germany",
            economic_status="recession",
            gdp_growth_pct=-0.6,
            inflation_pct=2.2,
            debt_to_gdp_pct=64.0,
            currency_strength=-0.05,
            trade_balance_bn=168.0,
            key_risks=[
                "Manufacturing slump and China export dependency",
                "Energy-intensive industry structurally repriced",
                "EU trade-war collateral between US and China",
            ],
            key_opportunities=[
                "Defense-industrial rebuild",
                "Green hydrogen long-cycle if subsidies hold",
            ],
            geopolitical_alignment="western",
            last_updated=now,
            news_summary=(
                "Eurozone core in mild recession; Germany as bellwether; "
                "caught between US tariffs and Chinese competition."
            ),
        ),
        "UK": CountryProfile(
            country_code="UK",
            country_name="United Kingdom",
            economic_status="weakening",
            gdp_growth_pct=0.9,
            inflation_pct=2.9,
            debt_to_gdp_pct=101.0,
            currency_strength=-0.2,
            trade_balance_bn=-168.0,
            key_risks=[
                "Post-Brexit stagnation and weak investment",
                "Housing and mortgage stress as rates normalize slowly",
                "Fiscal headroom limited",
            ],
            key_opportunities=[
                "Services exports and AI/fintech niche",
                "Energy transition North Sea angle",
            ],
            geopolitical_alignment="western",
            last_updated=now,
            news_summary=(
                "Low trend growth; political churn; sterling sensitive to risk and rate differentials."
            ),
        ),
        "JP": CountryProfile(
            country_code="JP",
            country_name="Japan",
            economic_status="weakening",
            gdp_growth_pct=0.4,
            inflation_pct=2.6,
            debt_to_gdp_pct=256.0,
            currency_strength=-0.55,
            trade_balance_bn=-68.0,
            key_risks=[
                "Yen volatility as BOJ exits ultra-easy regime",
                "Aging and debt sustainability optics",
                "Energy import bill on weak yen",
            ],
            key_opportunities=[
                "Corporate governance reform and buybacks",
                "Semiconductor materials and equipment niche",
            ],
            geopolitical_alignment="western",
            last_updated=now,
            news_summary=(
                "Yen under acute pressure episodes; policy pivot discontinuity; "
                "imported inflation mixing with weak domestic demand."
            ),
        ),
        "IN": CountryProfile(
            country_code="IN",
            country_name="India",
            economic_status="weakening",
            gdp_growth_pct=6.2,
            inflation_pct=5.1,
            debt_to_gdp_pct=84.0,
            currency_strength=-0.12,
            trade_balance_bn=-78.0,
            key_risks=[
                "Food and energy inflation politics",
                "Fiscal deficits and state debt",
                "Border/security friction with China/Pakistan",
            ],
            key_opportunities=[
                "Demographic dividend and manufacturing incentives",
                "Multi-aligned trade: buys Russian oil, sells to West",
            ],
            geopolitical_alignment="neutral",
            last_updated=now,
            news_summary=(
                "Fastest large-economy growth but inflation and deficits cap euphoria; "
                "strategic neutrality monetized."
            ),
        ),
        "BR": CountryProfile(
            country_code="BR",
            country_name="Brazil",
            economic_status="weakening",
            gdp_growth_pct=2.1,
            inflation_pct=4.5,
            debt_to_gdp_pct=88.0,
            currency_strength=-0.28,
            trade_balance_bn=28.0,
            key_risks=[
                "Fiscal consolidation fights",
                "Commodity price swings",
                "Amazon/climate policy friction with trade partners",
            ],
            key_opportunities=[
                "BRICS agenda and south-south trade",
                "Agri/mineral export market share",
            ],
            geopolitical_alignment="brics",
            last_updated=now,
            news_summary=(
                "BRICS participant; macro sensitive to China demand and agri terms of trade."
            ),
        ),
        "TR": CountryProfile(
            country_code="TR",
            country_name="Turkey",
            economic_status="crisis",
            gdp_growth_pct=2.8,
            inflation_pct=62.0,
            debt_to_gdp_pct=32.0,
            currency_strength=-0.72,
            trade_balance_bn=-48.0,
            key_risks=[
                "Chronic inflation and lira volatility",
                "External funding gaps",
                "Syria/Iraq/Caucasus spillovers",
            ],
            key_opportunities=[
                "Drone/defense exports",
                "Energy hub rhetoric if Black Sea stabilizes",
            ],
            geopolitical_alignment="neutral",
            last_updated=now,
            news_summary=(
                "Persistent currency crisis dynamics; geopolitical pivoting between NATO and Eurasia."
            ),
        ),
        "KR": CountryProfile(
            country_code="KR",
            country_name="South Korea",
            economic_status="weakening",
            gdp_growth_pct=1.0,
            inflation_pct=2.4,
            debt_to_gdp_pct=52.0,
            currency_strength=-0.18,
            trade_balance_bn=48.0,
            key_risks=[
                "US-China chip war exposure",
                "North Korea escalation tail risk",
                "Household debt and property stress",
            ],
            key_opportunities=[
                "Memory/HBM AI cycle",
                "US alliance tech subsidies",
            ],
            geopolitical_alignment="western",
            last_updated=now,
            news_summary=(
                "Tech supply chain in crossfire; security discount in assets during shocks."
            ),
        ),
        "TW": CountryProfile(
            country_code="TW",
            country_name="Taiwan",
            economic_status="weakening",
            gdp_growth_pct=2.6,
            inflation_pct=2.1,
            debt_to_gdp_pct=29.0,
            currency_strength=0.08,
            trade_balance_bn=82.0,
            key_risks=[
                "Invasion/blockade tail risk",
                "Concentration in semiconductors",
                "Sanctions spillover if conflict",
            ],
            key_opportunities=[
                "TSMC oligopoly in leading-edge nodes",
                "Critical leverage in US-China tech contest",
            ],
            geopolitical_alignment="western",
            last_updated=now,
            news_summary=(
                "Strait tensions elevated; semiconductor leverage is strategic and market-moving."
            ),
        ),
        "AR": CountryProfile(
            country_code="AR",
            country_name="Argentina",
            economic_status="crisis",
            gdp_growth_pct=-1.5,
            inflation_pct=186.0,
            debt_to_gdp_pct=45.0,
            currency_strength=-0.68,
            trade_balance_bn=6.0,
            key_risks=[
                "Social backlash to austerity",
                "Reserve fragility",
                "Dollarization path dependence",
            ],
            key_opportunities=[
                "Milei reforms unlocking IFI capital if sustained",
                "Vaca Muerta energy upside",
            ],
            geopolitical_alignment="neutral",
            last_updated=now,
            news_summary=(
                "Dollarization experiment and shock therapy; high inflation legacy; "
                "volatile politics around reform cadence."
            ),
        ),
    }


class CountryProfileManager:
    """In-memory country profiles with headline-driven nudges and macro synthesis."""

    def __init__(self) -> None:
        self._profiles: dict[str, CountryProfile] = _default_profiles()

    def get_profile(self, country_code: str) -> CountryProfile | None:
        return self._profiles.get(country_code.upper())

    def get_all_profiles(self) -> list[dict[str, Any]]:
        """Serialize profiles for downstream narrative engines."""
        rows: list[dict[str, Any]] = []
        for p in sorted(self._profiles.values(), key=lambda x: x.country_code):
            d = asdict(p)
            d["last_updated"] = p.last_updated.isoformat()
            rows.append(d)
        return rows

    def _match_countries(self, text: str) -> set[str]:
        lower = text.lower()
        hit: set[str] = set()
        for code, kws in _COUNTRY_KEYWORDS.items():
            for kw in kws:
                if kw in lower:
                    hit.add(code)
                    break
        return hit

    def update_from_news(self, headlines: list[str]) -> None:
        """Apply lightweight keyword/sentiment nudges from headline text."""
        if not headlines:
            return
        now = _utcnow()
        for raw in headlines:
            if not raw or not str(raw).strip():
                continue
            h = str(raw).strip()
            combined = h
            codes = self._match_countries(combined)
            if not codes:
                continue
            lower = combined.lower()
            crisis_hit = any(w in lower for w in _CRISIS_WORDS)
            stress_hit = any(w in lower for w in _STRESS_WORDS)
            relief_hit = any(w in lower for w in _RELIEF_WORDS)

            for code in codes:
                p = self._profiles[code]
                snippet = h if len(h) <= 200 else h[:197] + "..."
                extra = f"[{now.strftime('%Y-%m-%d %H:%M')} UTC] {snippet}"
                if len(p.news_summary) > 1200:
                    p.news_summary = p.news_summary[-900:]
                p.news_summary = (p.news_summary + " | " + extra).strip(" |")
                p.last_updated = now

                rank = _STATUS_RANK.get(p.economic_status, 1)
                if crisis_hit:
                    rank = min(3, rank + 2)
                elif stress_hit and not relief_hit:
                    rank = min(3, rank + 1)
                elif relief_hit and not crisis_hit:
                    rank = max(0, rank - 1)
                p.economic_status = _STATUS_BY_RANK[rank]

                if crisis_hit:
                    p.currency_strength = max(-1.0, min(1.0, p.currency_strength - 0.04))
                elif relief_hit:
                    p.currency_strength = max(-1.0, min(1.0, p.currency_strength + 0.02))

        logger.info("country_profiles_updated_from_news", headline_count=len(headlines))

    def get_macro_outlook(self) -> dict[str, Any]:
        """Aggregate macro narrative for strategy layers."""
        profiles = list(self._profiles.values())
        crisis_n = sum(1 for p in profiles if p.economic_status == "crisis")
        rec_n = sum(1 for p in profiles if p.economic_status == "recession")
        weak_n = sum(1 for p in profiles if p.economic_status == "weakening")

        global_risk_level = min(
            1.0,
            0.35 + 0.12 * crisis_n + 0.07 * rec_n + 0.03 * weak_n,
        )

        dominant_narratives = [
            "Multipolar trade and payment rails eroding single-currency hegemony",
            "Middle East escalation premium in energy and safe-haven flows",
            "Tariff and tech-war shocks to manufacturing supply chains",
            "High public debt and political constraints on fiscal adjustment",
        ]

        bearish_codes = {
            p.country_code
            for p in profiles
            if p.economic_status in ("crisis", "recession")
            or p.currency_strength < -0.4
        }
        bullish_codes = {
            p.country_code
            for p in profiles
            if p.economic_status == "strong"
            or (p.gdp_growth_pct >= 4.0 and p.economic_status not in ("crisis", "recession"))
        }
        # Net commodity/BRICS tilt can look "bullish" on macro carry even if politics ugly
        for p in profiles:
            if p.country_code in ("SA", "BR", "RU") and p.trade_balance_bn > 40:
                bullish_codes.add(p.country_code)

        crypto_impact_score = max(
            -1.0,
            min(
                1.0,
                0.25 * (global_risk_level - 0.5)
                + 0.15 * (crisis_n - 1)
                + 0.1 * (1.0 if any("sanction" in r.lower() for p in profiles for r in p.key_risks) else 0),
            ),
        )

        return {
            "global_risk_level": round(global_risk_level, 3),
            "dominant_narratives": dominant_narratives,
            "bullish_regions": sorted(bullish_codes),
            "bearish_regions": sorted(bearish_codes),
            "crypto_impact_score": round(crypto_impact_score, 3),
        }

    def get_geopolitical_tensions(self) -> list[dict[str, Any]]:
        """Active conflict dyads and heuristic severity (0–1)."""
        return [
            {
                "pair": ("IL", "IR"),
                "severity": 0.92,
                "theme": "Direct and proxy confrontation; nuclear/sanctions subtext",
            },
            {
                "pair": ("US", "CN"),
                "severity": 0.78,
                "theme": "Tariffs, tech decoupling, and South China Sea/Taiwan linkage",
            },
            {
                "pair": ("RU", "NATO"),
                "severity": 0.74,
                "theme": "Ukraine spillover; sanctions; energy security",
            },
            {
                "pair": ("CN", "TW"),
                "severity": 0.88,
                "theme": "Strait crisis risk; semiconductor leverage",
            },
            {
                "pair": ("IR", "US"),
                "severity": 0.81,
                "theme": "Sanctions, maritime security, Israel alignment",
            },
            {
                "pair": ("SA", "IR"),
                "severity": 0.62,
                "theme": "Gulf rivalry; Yemen/Red Sea externalities",
            },
        ]


RSS_FEEDS_DEFAULT = (
    "https://feeds.reuters.com/reuters/topNews",
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
)

_FETCH_INTERVAL_SEC = 15 * 60


class FreeNewsCollector:
    """Fetch headlines from free RSS feeds with per-feed rate limiting."""

    def __init__(
        self,
        feed_urls: tuple[str, ...] | list[str] | None = None,
        timeout_sec: float = 25.0,
    ) -> None:
        self._feeds = tuple(feed_urls) if feed_urls is not None else RSS_FEEDS_DEFAULT
        self._timeout = timeout_sec
        self._last_fetch_mono: dict[str, float] = {}
        self._cache_headlines: dict[str, list[str]] = {}

    def _should_fetch(self, url: str) -> bool:
        last = self._last_fetch_mono.get(url)
        if last is None:
            return True
        return (time.monotonic() - last) >= _FETCH_INTERVAL_SEC

    async def fetch_headlines(self) -> list[str]:
        headers = {
            "User-Agent": "PolymarketBot/1.0 (+https://polymarket.com; macro RSS reader)",
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
        }
        out: list[str] = []
        async with httpx.AsyncClient(timeout=self._timeout, headers=headers, follow_redirects=True) as client:
            for url in self._feeds:
                if not self._should_fetch(url):
                    cached = self._cache_headlines.get(url, [])
                    out.extend(cached)
                    logger.debug("rss_rate_limited_using_cache", url=url, cached=len(cached))
                    continue
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    items = _parse_rss_items(resp.text)
                    headlines: list[str] = []
                    for it in items:
                        title = (it.get("title") or "").strip()
                        desc = (it.get("description") or "").strip()
                        if title:
                            line = title if not desc else f"{title} — {desc}"
                            headlines.append(line[:500])
                    self._cache_headlines[url] = headlines
                    self._last_fetch_mono[url] = time.monotonic()
                    out.extend(headlines)
                    logger.info("rss_fetched", url=url, count=len(headlines))
                except Exception as e:
                    logger.warning("rss_fetch_failed", url=url, error=str(e))
                    out.extend(self._cache_headlines.get(url, []))
        return out

    def update_profiles(self, manager: CountryProfileManager) -> None:
        """Sync wrapper: fetch headlines and push into the manager."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            headlines = asyncio.run(self.fetch_headlines())
        else:
            raise RuntimeError(
                "update_profiles() cannot be called from a running event loop; "
                "await fetch_headlines() and then manager.update_from_news(headlines)."
            )
        manager.update_from_news(headlines)
