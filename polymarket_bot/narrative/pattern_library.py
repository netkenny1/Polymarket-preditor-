"""Built-in historical pattern library for predictive history matching.

Contains ~15 hardcoded patterns derived from major market-moving episodes
(2016-2026).  Each pattern is broken into 3-5 phases with realistic
durations, keyword sets, and directional market-impact estimates.
"""

from __future__ import annotations

from polymarket_bot.data.models import (
    HistoricalPattern,
    HistoricalPhase,
    NarrativeCategory,
)


def _p(
    phase_name: str,
    description: str,
    duration_days: int,
    market_impact: dict[str, float],
    keywords: list[str],
    sequence_index: int,
) -> HistoricalPhase:
    """Shorthand factory for HistoricalPhase."""
    return HistoricalPhase(
        phase_name=phase_name,
        description=description,
        duration_days=duration_days,
        market_impact=market_impact,
        keywords=keywords,
        sequence_index=sequence_index,
    )


def get_builtin_patterns() -> list[HistoricalPattern]:
    """Return the full set of built-in historical patterns."""

    patterns: list[HistoricalPattern] = []

    # ------------------------------------------------------------------ 1
    patterns.append(
        HistoricalPattern(
            pattern_id="hist_trade_war_2018",
            name="2018-2019 US-China Trade War",
            category=NarrativeCategory.TRADE_WAR,
            description=(
                "Escalating tariff conflict between the US and China that "
                "roiled global equity and commodity markets over ~18 months."
            ),
            trigger_keywords=[
                "tariff", "china", "trade war", "duties",
                "import tax", "retaliation",
            ],
            timeline_days=540,
            market_impact={"politics": -0.3, "crypto": -0.1, "other": -0.2},
            outcome_direction=-0.3,
            outcome_magnitude=0.15,
            phases=[
                _p(
                    "Tariff Escalation",
                    "Initial tariff announcements on steel/aluminum expand to broader goods.",
                    90,
                    {"politics": -0.3, "crypto": -0.05, "other": -0.2},
                    ["tariff", "duties", "import tax", "section 301"],
                    0,
                ),
                _p(
                    "Retaliation",
                    "China retaliates with counter-tariffs; tit-for-tat escalation.",
                    120,
                    {"politics": -0.4, "crypto": -0.1, "other": -0.3},
                    ["retaliation", "counter-tariff", "soybean", "rare earth"],
                    1,
                ),
                _p(
                    "Negotiation",
                    "Both sides signal willingness to negotiate; markets stabilize.",
                    180,
                    {"politics": 0.1, "crypto": 0.0, "other": 0.05},
                    ["negotiation", "trade deal", "talks", "ceasefire"],
                    2,
                ),
                _p(
                    "Partial Resolution",
                    "Phase-one deal signed; some tariffs rolled back.",
                    150,
                    {"politics": 0.15, "crypto": 0.05, "other": 0.1},
                    ["phase one", "deal", "signing", "rollback"],
                    3,
                ),
            ],
            similarity_threshold=0.4,
            source_period="2018-01 to 2020-01",
        )
    )

    # ------------------------------------------------------------------ 2
    patterns.append(
        HistoricalPattern(
            pattern_id="hist_fed_hikes_2022",
            name="2022 Fed Rate Hike Cycle",
            category=NarrativeCategory.MONETARY_POLICY,
            description=(
                "Aggressive Federal Reserve tightening cycle to combat "
                "post-pandemic inflation, taking rates from near-zero to "
                "over 5%."
            ),
            trigger_keywords=[
                "fed", "rate hike", "inflation", "tightening",
                "powell", "fomc",
            ],
            timeline_days=365,
            market_impact={"politics": -0.1, "crypto": -0.3, "other": -0.2},
            outcome_direction=-0.2,
            outcome_magnitude=0.12,
            phases=[
                _p(
                    "Initial Hike",
                    "First 25bp hike signals the start of the tightening cycle.",
                    60,
                    {"politics": -0.05, "crypto": -0.15, "other": -0.1},
                    ["rate hike", "25 basis points", "liftoff", "fomc"],
                    0,
                ),
                _p(
                    "Acceleration",
                    "Fed shifts to 50bp and 75bp hikes; hawkish forward guidance.",
                    120,
                    {"politics": -0.15, "crypto": -0.4, "other": -0.3},
                    ["75 basis points", "hawkish", "inflation", "cpi"],
                    1,
                ),
                _p(
                    "Market Adjustment",
                    "Equities and crypto reprice; growth stocks hammered.",
                    120,
                    {"politics": -0.1, "crypto": -0.35, "other": -0.25},
                    ["bear market", "selloff", "valuation", "recession"],
                    2,
                ),
                _p(
                    "Plateau",
                    "Rate hikes slow; markets anticipate a pause or pivot.",
                    65,
                    {"politics": 0.05, "crypto": 0.1, "other": 0.1},
                    ["pause", "pivot", "peak rate", "disinflation"],
                    3,
                ),
            ],
            similarity_threshold=0.4,
            source_period="2022-03 to 2023-07",
        )
    )

    # ------------------------------------------------------------------ 3
    patterns.append(
        HistoricalPattern(
            pattern_id="hist_china_crypto_ban_2021",
            name="2021 China Crypto Ban",
            category=NarrativeCategory.CRYPTO_REGULATION,
            description=(
                "China's sweeping ban on cryptocurrency mining and trading "
                "triggered a major crypto selloff and global hash-rate "
                "redistribution."
            ),
            trigger_keywords=[
                "crypto", "ban", "china", "mining", "regulation",
            ],
            timeline_days=180,
            market_impact={"politics": 0.0, "crypto": -0.5, "other": -0.05},
            outcome_direction=-0.4,
            outcome_magnitude=0.20,
            phases=[
                _p(
                    "Rumor",
                    "Leaked documents and social-media rumors about an imminent ban.",
                    14,
                    {"politics": 0.0, "crypto": -0.2, "other": 0.0},
                    ["rumor", "leak", "crackdown", "warning"],
                    0,
                ),
                _p(
                    "Announcement",
                    "State Council officially declares crypto mining and trading illegal.",
                    7,
                    {"politics": 0.0, "crypto": -0.6, "other": -0.05},
                    ["ban", "announcement", "state council", "illegal"],
                    1,
                ),
                _p(
                    "Enforcement",
                    "Mining farms shut down; exchanges exit China; hash-rate plunges.",
                    60,
                    {"politics": 0.0, "crypto": -0.45, "other": -0.05},
                    ["enforcement", "shutdown", "mining", "hashrate"],
                    2,
                ),
                _p(
                    "Adaptation",
                    "Miners relocate; network recovers; prices stabilize.",
                    99,
                    {"politics": 0.0, "crypto": 0.15, "other": 0.0},
                    ["relocation", "recovery", "hashrate recovery", "stabilize"],
                    3,
                ),
            ],
            similarity_threshold=0.4,
            source_period="2021-05 to 2021-12",
        )
    )

    # ------------------------------------------------------------------ 4
    patterns.append(
        HistoricalPattern(
            pattern_id="hist_covid_crash_2020",
            name="2020 COVID Market Crash",
            category=NarrativeCategory.MARKET_CRISIS,
            description=(
                "Global pandemic triggered the fastest bear market in "
                "history, followed by massive fiscal/monetary stimulus and "
                "a V-shaped recovery."
            ),
            trigger_keywords=[
                "covid", "pandemic", "lockdown", "stimulus", "fed",
            ],
            timeline_days=180,
            market_impact={"politics": 0.1, "crypto": 0.15, "other": 0.05},
            outcome_direction=0.1,
            outcome_magnitude=0.25,
            phases=[
                _p(
                    "Initial Shock",
                    "First reports of global spread; markets begin to price in disruption.",
                    14,
                    {"politics": -0.1, "crypto": -0.2, "other": -0.15},
                    ["covid", "virus", "outbreak", "WHO"],
                    0,
                ),
                _p(
                    "Panic Selloff",
                    "Circuit breakers triggered; S&P 500 drops ~34% in weeks.",
                    21,
                    {"politics": -0.3, "crypto": -0.5, "other": -0.4},
                    ["crash", "selloff", "circuit breaker", "panic"],
                    1,
                ),
                _p(
                    "Stimulus Response",
                    "Fed cuts to zero; Congress passes CARES Act; unlimited QE.",
                    30,
                    {"politics": 0.2, "crypto": 0.3, "other": 0.25},
                    ["stimulus", "CARES", "QE", "rate cut", "fed"],
                    2,
                ),
                _p(
                    "Recovery",
                    "Markets rally sharply; tech leads; new highs by late summer.",
                    115,
                    {"politics": 0.15, "crypto": 0.4, "other": 0.2},
                    ["recovery", "rally", "V-shape", "new highs"],
                    3,
                ),
            ],
            similarity_threshold=0.4,
            source_period="2020-02 to 2020-08",
        )
    )

    # ------------------------------------------------------------------ 5
    patterns.append(
        HistoricalPattern(
            pattern_id="hist_trump_election_2016",
            name="2016 Trump Election Surprise",
            category=NarrativeCategory.ELECTION,
            description=(
                "Donald Trump's unexpected presidential victory initially "
                "shocked futures markets but led to a sustained equity rally "
                "on tax-cut and deregulation expectations."
            ),
            trigger_keywords=[
                "trump", "election", "polls", "surprise", "upset",
            ],
            timeline_days=120,
            market_impact={"politics": 0.3, "crypto": 0.1, "other": 0.15},
            outcome_direction=0.2,
            outcome_magnitude=0.15,
            phases=[
                _p(
                    "Polls Diverge",
                    "Polling aggregates tighten; prediction markets still favor Clinton.",
                    30,
                    {"politics": -0.05, "crypto": 0.0, "other": -0.05},
                    ["polls", "tightening", "prediction market", "odds"],
                    0,
                ),
                _p(
                    "Election Night Shock",
                    "Trump wins key swing states; futures plunge then reverse.",
                    1,
                    {"politics": -0.4, "crypto": 0.0, "other": -0.3},
                    ["election night", "upset", "surprise", "swing state"],
                    1,
                ),
                _p(
                    "Market Rally",
                    "Equities surge on expectations of tax cuts and deregulation.",
                    30,
                    {"politics": 0.4, "crypto": 0.1, "other": 0.3},
                    ["rally", "tax cuts", "deregulation", "infrastructure"],
                    2,
                ),
                _p(
                    "Policy Uncertainty",
                    "Markets digest the reality of governance; volatility rises.",
                    59,
                    {"politics": 0.1, "crypto": 0.05, "other": 0.05},
                    ["uncertainty", "executive order", "cabinet", "policy"],
                    3,
                ),
            ],
            similarity_threshold=0.4,
            source_period="2016-10 to 2017-03",
        )
    )

    # ------------------------------------------------------------------ 6
    patterns.append(
        HistoricalPattern(
            pattern_id="hist_svb_crisis_2023",
            name="2023 Banking Crisis (SVB)",
            category=NarrativeCategory.MARKET_CRISIS,
            description=(
                "Silicon Valley Bank's sudden failure sparked contagion "
                "fears across regional banks before Fed backstops "
                "stabilized the system."
            ),
            trigger_keywords=[
                "bank", "svb", "failure", "contagion", "fdic",
            ],
            timeline_days=60,
            market_impact={"politics": -0.05, "crypto": 0.1, "other": -0.1},
            outcome_direction=-0.1,
            outcome_magnitude=0.12,
            phases=[
                _p(
                    "Bank Failure",
                    "SVB discloses massive bond losses; depositors flee; FDIC takes over.",
                    3,
                    {"politics": -0.15, "crypto": -0.1, "other": -0.2},
                    ["svb", "bank run", "failure", "fdic", "insolvency"],
                    0,
                ),
                _p(
                    "Contagion Fear",
                    "Signature Bank and First Republic wobble; KBW index plunges.",
                    10,
                    {"politics": -0.2, "crypto": -0.15, "other": -0.25},
                    ["contagion", "signature bank", "first republic", "regional bank"],
                    1,
                ),
                _p(
                    "Fed Response",
                    "Fed creates BTFP lending facility; Treasury backstops deposits.",
                    14,
                    {"politics": 0.1, "crypto": 0.2, "other": 0.1},
                    ["btfp", "backstop", "fed", "emergency lending"],
                    2,
                ),
                _p(
                    "Stabilization",
                    "Deposit flows normalize; focus shifts back to inflation.",
                    33,
                    {"politics": 0.05, "crypto": 0.15, "other": 0.05},
                    ["stabilize", "normalize", "confidence", "resolution"],
                    3,
                ),
            ],
            similarity_threshold=0.4,
            source_period="2023-03 to 2023-05",
        )
    )

    # ------------------------------------------------------------------ 7
    patterns.append(
        HistoricalPattern(
            pattern_id="hist_iran_tensions_2019",
            name="2019 Trump-Iran Tensions",
            category=NarrativeCategory.GEOPOLITICAL,
            description=(
                "US-Iran tensions escalated sharply after the Soleimani "
                "strike but de-escalated without broader conflict."
            ),
            trigger_keywords=[
                "iran", "tensions", "military", "strike", "sanctions",
            ],
            timeline_days=60,
            market_impact={"politics": 0.0, "crypto": 0.05, "other": -0.05},
            outcome_direction=0.0,
            outcome_magnitude=0.08,
            phases=[
                _p(
                    "Provocation",
                    "Drone attacks on Saudi oil facilities; US blames Iran.",
                    14,
                    {"politics": -0.1, "crypto": 0.05, "other": -0.1},
                    ["provocation", "drone", "saudi", "oil"],
                    0,
                ),
                _p(
                    "Escalation",
                    "US kills Soleimani; Iran retaliates with missile strikes on Iraqi bases.",
                    7,
                    {"politics": -0.2, "crypto": 0.1, "other": -0.15},
                    ["soleimani", "strike", "missile", "retaliation"],
                    1,
                ),
                _p(
                    "De-escalation",
                    "Both sides signal restraint; no casualties from missile response.",
                    14,
                    {"politics": 0.1, "crypto": 0.0, "other": 0.05},
                    ["de-escalation", "restraint", "diplomacy", "stand down"],
                    2,
                ),
                _p(
                    "Status Quo",
                    "Markets return to pre-crisis levels; attention shifts elsewhere.",
                    25,
                    {"politics": 0.05, "crypto": 0.0, "other": 0.02},
                    ["normalize", "status quo", "calm", "resolution"],
                    3,
                ),
            ],
            similarity_threshold=0.4,
            source_period="2019-09 to 2020-01",
        )
    )

    # ------------------------------------------------------------------ 8
    patterns.append(
        HistoricalPattern(
            pattern_id="hist_ukraine_2022",
            name="2022 Russia-Ukraine Conflict",
            category=NarrativeCategory.GEOPOLITICAL,
            description=(
                "Russia's invasion of Ukraine caused a commodity shock, "
                "energy crisis in Europe, and sustained geopolitical "
                "risk premium."
            ),
            trigger_keywords=[
                "russia", "ukraine", "invasion", "sanctions", "nato",
            ],
            timeline_days=365,
            market_impact={"politics": -0.2, "crypto": -0.15, "other": -0.25},
            outcome_direction=-0.3,
            outcome_magnitude=0.18,
            phases=[
                _p(
                    "Military Buildup",
                    "Satellite imagery shows Russian troop buildup; diplomatic talks fail.",
                    60,
                    {"politics": -0.1, "crypto": -0.05, "other": -0.1},
                    ["buildup", "troops", "border", "diplomacy"],
                    0,
                ),
                _p(
                    "Invasion",
                    "Full-scale invasion begins; global condemnation; markets plunge.",
                    14,
                    {"politics": -0.3, "crypto": -0.2, "other": -0.35},
                    ["invasion", "war", "attack", "kyiv"],
                    1,
                ),
                _p(
                    "Sanctions Wave",
                    "Unprecedented Western sanctions; SWIFT ban; energy embargo talks.",
                    60,
                    {"politics": -0.25, "crypto": -0.1, "other": -0.3},
                    ["sanctions", "swift", "embargo", "energy"],
                    2,
                ),
                _p(
                    "War of Attrition",
                    "Conflict becomes protracted; commodity prices stabilize at high levels.",
                    231,
                    {"politics": -0.1, "crypto": -0.05, "other": -0.15},
                    ["attrition", "stalemate", "commodity", "energy crisis"],
                    3,
                ),
            ],
            similarity_threshold=0.4,
            source_period="2021-12 to 2023-01",
        )
    )

    # ------------------------------------------------------------------ 9
    patterns.append(
        HistoricalPattern(
            pattern_id="hist_meme_stock_2021",
            name="2021 Meme Stock Mania",
            category=NarrativeCategory.MARKET_CRISIS,
            description=(
                "Retail-driven short squeeze on GameStop and other heavily "
                "shorted stocks disrupted markets and drew regulatory "
                "scrutiny."
            ),
            trigger_keywords=[
                "gamestop", "meme", "short squeeze", "robinhood", "wsb",
            ],
            timeline_days=90,
            market_impact={"politics": 0.05, "crypto": 0.1, "other": 0.0},
            outcome_direction=0.0,
            outcome_magnitude=0.15,
            phases=[
                _p(
                    "Retail Surge",
                    "WallStreetBets community coordinates buying of GME and AMC.",
                    14,
                    {"politics": 0.05, "crypto": 0.1, "other": 0.1},
                    ["wsb", "reddit", "retail", "gamestop", "buy"],
                    0,
                ),
                _p(
                    "Short Squeeze",
                    "GME rockets from $20 to $480; Melvin Capital suffers billions in losses.",
                    7,
                    {"politics": 0.1, "crypto": 0.15, "other": 0.15},
                    ["short squeeze", "gamma squeeze", "melvin", "hedge fund"],
                    1,
                ),
                _p(
                    "Regulatory Scrutiny",
                    "Robinhood restricts buying; Congressional hearings announced.",
                    21,
                    {"politics": -0.1, "crypto": 0.0, "other": -0.1},
                    ["robinhood", "restriction", "hearing", "sec", "regulation"],
                    2,
                ),
                _p(
                    "Normalization",
                    "Prices deflate; attention fades; new regulations discussed.",
                    48,
                    {"politics": 0.0, "crypto": 0.0, "other": -0.05},
                    ["deflate", "normalize", "regulation", "retail"],
                    3,
                ),
            ],
            similarity_threshold=0.4,
            source_period="2021-01 to 2021-04",
        )
    )

    # ------------------------------------------------------------------ 10
    patterns.append(
        HistoricalPattern(
            pattern_id="hist_btc_etf_2024",
            name="2024 Bitcoin ETF Approval",
            category=NarrativeCategory.CRYPTO_REGULATION,
            description=(
                "Years of speculation culminated in SEC approval of spot "
                "Bitcoin ETFs, triggering a brief sell-the-news dip "
                "followed by a sustained rally."
            ),
            trigger_keywords=[
                "bitcoin", "etf", "sec", "approval", "spot",
            ],
            timeline_days=120,
            market_impact={"politics": 0.05, "crypto": 0.4, "other": 0.05},
            outcome_direction=0.3,
            outcome_magnitude=0.12,
            phases=[
                _p(
                    "Speculation",
                    "Grayscale court victory and BlackRock filing fuel ETF optimism.",
                    60,
                    {"politics": 0.0, "crypto": 0.25, "other": 0.0},
                    ["etf", "filing", "grayscale", "blackrock", "speculation"],
                    0,
                ),
                _p(
                    "Approval",
                    "SEC officially approves 11 spot Bitcoin ETFs on January 10, 2024.",
                    1,
                    {"politics": 0.05, "crypto": 0.3, "other": 0.05},
                    ["approval", "sec", "spot etf", "launch"],
                    1,
                ),
                _p(
                    "Sell the News",
                    "BTC briefly dips as early speculators take profit.",
                    14,
                    {"politics": 0.0, "crypto": -0.15, "other": 0.0},
                    ["sell the news", "profit taking", "dip", "correction"],
                    2,
                ),
                _p(
                    "Sustained Rally",
                    "Institutional inflows drive BTC past previous highs.",
                    45,
                    {"politics": 0.05, "crypto": 0.5, "other": 0.05},
                    ["inflows", "institutional", "rally", "all-time high", "adoption"],
                    3,
                ),
            ],
            similarity_threshold=0.4,
            source_period="2023-10 to 2024-03",
        )
    )

    # ------------------------------------------------------------------ 11
    patterns.append(
        HistoricalPattern(
            pattern_id="hist_trump_tariff_shock_2026",
            name="2025-2026 Trump Tariff Shock",
            category=NarrativeCategory.TRADE_WAR,
            description=(
                "Trump's 'Liberation Day' tariffs (April 2025) trigger "
                "global retaliation, supply chain chaos, and stagflationary "
                "pressure across trade-linked assets."
            ),
            trigger_keywords=[
                "tariff", "liberation day", "retaliation", "trade war",
                "import duties", "supply chain", "stagflation",
            ],
            timeline_days=450,
            market_impact={"politics": -0.3, "crypto": -0.2, "other": -0.4},
            outcome_direction=-0.3,
            outcome_magnitude=0.16,
            phases=[
                _p(
                    "Initial Shock",
                    "Sweeping tariff announcements hit expectations; equities and "
                    "trade-sensitive sectors reprice sharply.",
                    30,
                    {"politics": -0.35, "crypto": -0.2, "other": -0.35},
                    ["tariff", "liberation day", "announcement", "duties"],
                    0,
                ),
                _p(
                    "Retaliation Wave",
                    "Major trading partners impose counter-tariffs; diplomatic "
                    "rhetoric hardens and uncertainty spikes.",
                    90,
                    {"politics": -0.4, "crypto": -0.25, "other": -0.45},
                    ["retaliation", "counter-tariff", "trade partner", "escalation"],
                    1,
                ),
                _p(
                    "Supply Chain Disruption",
                    "Rerouting, inventory builds, and input-cost shocks ripple "
                    "through manufacturing and logistics.",
                    150,
                    {"politics": -0.25, "crypto": -0.15, "other": -0.45},
                    ["supply chain", "logistics", "inventory", "manufacturing"],
                    2,
                ),
                _p(
                    "Stagflationary Pressure",
                    "Slower growth meets sticky inflation; markets discount "
                    "persistent policy and pricing friction.",
                    180,
                    {"politics": -0.3, "crypto": -0.2, "other": -0.4},
                    ["stagflation", "inflation", "slowdown", "pricing"],
                    3,
                ),
            ],
            similarity_threshold=0.4,
            source_period="2025-04 to 2026-04",
        )
    )

    # ------------------------------------------------------------------ 12
    patterns.append(
        HistoricalPattern(
            pattern_id="hist_petrodollar_collapse_2026",
            name="2025-2026 Petrodollar Collapse",
            category=NarrativeCategory.MARKET_CRISIS,
            description=(
                "Saudi Arabia and allies accept yuan and non-dollar settlement; "
                "BRICS currency talks accelerate; USD reserve status erodes "
                "and bond markets come under stress."
            ),
            trigger_keywords=[
                "petrodollar", "brics", "dollar", "reserve currency", "yuan",
                "saudi", "dedollarization",
            ],
            timeline_days=540,
            market_impact={"politics": -0.2, "crypto": 0.3, "other": -0.5},
            outcome_direction=-0.15,
            outcome_magnitude=0.18,
            phases=[
                _p(
                    "De-dollarization Signals",
                    "Oil and trade invoicing shift toward non-dollar currencies; "
                    "policy elites debate reserve diversification.",
                    120,
                    {"politics": -0.15, "crypto": 0.15, "other": -0.35},
                    ["dedollarization", "invoicing", "settlement", "reserve"],
                    0,
                ),
                _p(
                    "BRICS Currency Push",
                    "Summits and working groups float alternative payment rails "
                    "and basket-currency ideas.",
                    150,
                    {"politics": -0.2, "crypto": 0.25, "other": -0.45},
                    ["brics", "payment rail", "basket currency", "summit"],
                    1,
                ),
                _p(
                    "USD Weakness Cascade",
                    "Dollar selling, widening credit spreads, and Treasury "
                    "volatility as foreign demand for US paper wavers.",
                    150,
                    {"politics": -0.25, "crypto": 0.35, "other": -0.55},
                    ["dollar", "treasury", "spread", "volatility", "fx"],
                    2,
                ),
                _p(
                    "New Order Emergence",
                    "Markets price a multipolar monetary system; winners and "
                    "losers in funding markets become clearer.",
                    120,
                    {"politics": -0.15, "crypto": 0.4, "other": -0.4},
                    ["multipolar", "new order", "funding", "allocation"],
                    3,
                ),
            ],
            similarity_threshold=0.4,
            source_period="2025-01 to 2026-04",
        )
    )

    # ------------------------------------------------------------------ 13
    patterns.append(
        HistoricalPattern(
            pattern_id="hist_middle_east_escalation_2026",
            name="2025-2026 Middle East Escalation",
            category=NarrativeCategory.GEOPOLITICAL,
            description=(
                "Iran-Israel tensions widen into proxy fights (Yemen Houthis, "
                "Hezbollah), risking oil supply disruption and a global "
                "energy-price shock."
            ),
            trigger_keywords=[
                "iran", "israel", "middle east", "houthi", "hezbollah", "oil",
                "strait of hormuz", "yemen",
            ],
            timeline_days=420,
            market_impact={"politics": -0.3, "crypto": 0.0, "other": -0.35},
            outcome_direction=-0.28,
            outcome_magnitude=0.17,
            phases=[
                _p(
                    "Proxy Escalation",
                    "Missile and drone exchanges intensify via proxies; "
                    "shipping premiums and insurance costs rise.",
                    90,
                    {"politics": -0.25, "crypto": -0.1, "other": -0.3},
                    ["houthi", "hezbollah", "proxy", "drone", "missile"],
                    0,
                ),
                _p(
                    "Direct Confrontation",
                    "Risk of direct Iran-Israel engagement dominates headlines; "
                    "safe-haven flows compete with deleveraging.",
                    60,
                    {"politics": -0.4, "crypto": 0.1, "other": -0.4},
                    ["iran", "israel", "confrontation", "strike", "military"],
                    1,
                ),
                _p(
                    "Oil Supply Shock",
                    "Strait of Hormuz fears and outages drive crude sharply "
                    "higher; stagflation fears spread.",
                    120,
                    {"politics": -0.35, "crypto": -0.05, "other": -0.5},
                    ["oil", "strait of hormuz", "opec", "crude", "energy"],
                    2,
                ),
                _p(
                    "Diplomatic Scramble",
                    "Ceasefire talks, sanctions tweaks, and emergency releases "
                    "modulate but do not erase the risk premium.",
                    150,
                    {"politics": -0.15, "crypto": 0.05, "other": -0.2},
                    ["diplomacy", "ceasefire", "sanctions", "talks", "release"],
                    3,
                ),
            ],
            similarity_threshold=0.4,
            source_period="2025-06 to 2026-04",
        )
    )

    # ------------------------------------------------------------------ 14
    patterns.append(
        HistoricalPattern(
            pattern_id="hist_global_recession_2026",
            name="2025-2026 Global Recession",
            category=NarrativeCategory.MARKET_CRISIS,
            description=(
                "Trade-war drag, rate uncertainty, and geopolitical risk "
                "compound into a consumer slowdown, earnings compression, "
                "and rising credit stress."
            ),
            trigger_keywords=[
                "recession", "slowdown", "layoffs", "earnings", "gdp",
                "unemployment", "consumer",
            ],
            timeline_days=600,
            market_impact={"politics": -0.35, "crypto": -0.35, "other": -0.4},
            outcome_direction=-0.35,
            outcome_magnitude=0.18,
            phases=[
                _p(
                    "Leading Indicators Turn",
                    "PMIs, freight indices, and credit impulse roll over; "
                    "markets debate hard vs. soft landing.",
                    120,
                    {"politics": -0.2, "crypto": -0.25, "other": -0.3},
                    ["pmi", "leading indicator", "freight", "credit impulse"],
                    0,
                ),
                _p(
                    "Consumer Slowdown",
                    "Real spending weakens; retail misses and delinquencies tick up.",
                    150,
                    {"politics": -0.3, "crypto": -0.3, "other": -0.35},
                    ["consumer", "retail", "spending", "delinquency"],
                    1,
                ),
                _p(
                    "Corporate Stress",
                    "Earnings revisions deepen; layoffs and guidance cuts "
                    "spread beyond cyclical sectors.",
                    180,
                    {"politics": -0.4, "crypto": -0.4, "other": -0.45},
                    ["earnings", "layoffs", "guidance", "revisions"],
                    2,
                ),
                _p(
                    "Policy Response",
                    "Fiscal and monetary easing arrive but lag the damage; "
                    "risk assets only partially stabilize.",
                    150,
                    {"politics": -0.25, "crypto": -0.3, "other": -0.35},
                    ["fed", "rate cut", "stimulus", "fiscal", "easing"],
                    3,
                ),
            ],
            similarity_threshold=0.4,
            source_period="2025-01 to 2026-04",
        )
    )

    # ------------------------------------------------------------------ 15
    patterns.append(
        HistoricalPattern(
            pattern_id="hist_crypto_regime_shift_2026",
            name="2025-2026 Crypto Regime Shift",
            category=NarrativeCategory.CRYPTO_REGULATION,
            description=(
                "DeFi enforcement and stablecoin rules tighten, then spot "
                "ETF and institutional pipelines mature; Bitcoin's "
                "'digital gold' narrative strengthens amid macro chaos."
            ),
            trigger_keywords=[
                "crypto regulation", "stablecoin", "etf", "digital gold",
                "defi", "institutional",
            ],
            timeline_days=365,
            market_impact={"politics": 0.0, "crypto": 0.15, "other": -0.05},
            outcome_direction=0.25,
            outcome_magnitude=0.16,
            phases=[
                _p(
                    "Regulatory Crackdown",
                    "Agencies target DeFi venues and stablecoin issuers; "
                    "compliance costs and delisting fears weigh on altcoins.",
                    90,
                    {"politics": -0.05, "crypto": -0.3, "other": -0.05},
                    ["defi", "stablecoin", "enforcement", "compliance", "sec"],
                    0,
                ),
                _p(
                    "Institutional Entry",
                    "ETF flows, custody upgrades, and corporate treasuries "
                    "normalize large-size BTC allocation.",
                    120,
                    {"politics": 0.05, "crypto": 0.2, "other": 0.0},
                    ["etf", "institutional", "custody", "treasury", "inflows"],
                    1,
                ),
                _p(
                    "Safe Haven Narrative",
                    "Macro shocks drive narrative demand for scarce digital "
                    "assets alongside traditional hedges.",
                    90,
                    {"politics": 0.0, "crypto": 0.35, "other": -0.1},
                    ["digital gold", "safe haven", "macro", "hedge", "btc"],
                    2,
                ),
                _p(
                    "New Equilibrium",
                    "Liquidity, basis trades, and regulated products define "
                    "a higher structural crypto beta to policy cycles.",
                    65,
                    {"politics": 0.05, "crypto": 0.4, "other": 0.0},
                    ["equilibrium", "liquidity", "regulated", "basis", "adoption"],
                    3,
                ),
            ],
            similarity_threshold=0.4,
            source_period="2025-01 to 2026-04",
        )
    )

    return patterns
