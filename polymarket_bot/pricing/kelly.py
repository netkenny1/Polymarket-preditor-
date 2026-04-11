"""Kelly criterion sizing for Polymarket binary contracts.

A Polymarket binary contract costs `p` (0 < p < 1) and pays exactly 1 USDC
if the outcome occurs, 0 if not. Given the true probability `q`, the
Kelly fraction of bankroll that maximizes expected log-growth is:

    YES side (buy when q > p):   f* = (q - p) / (1 - p)
    NO  side (buy when q < p):   f* = (p - q) / p

Full Kelly assumes `q` is known exactly. With model noise, full Kelly has
high drawdown variance and the log-growth curve is near-flat between half
and full Kelly — so practitioners almost always use **fractional Kelly**
(quarter to half). We default to 0.25.

References:
  - Kelly (1956), "A New Interpretation of Information Rate"
  - Thorp (1997), "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market"
  - MacLean, Thorp, Ziemba (2010), "Good and Bad Properties of the Kelly Criterion"
"""

from __future__ import annotations

from polymarket_bot.data.models import Side


def kelly_fraction(
    q: float,
    p: float,
    side: Side = Side.BUY,
    outcome_is_yes: bool = True,
    fraction: float = 0.25,
    cap: float = 0.25,
) -> float:
    """Kelly fraction of bankroll for a binary-contract bet.

    Args:
        q: Our estimate of the true probability of the bet winning.
        p: Market price of the contract we're buying (0 < p < 1).
        side: BUY or SELL (SELL not natively supported here; use opposite-side BUY).
        outcome_is_yes: True if we're buying the YES/UP token, False for NO/DOWN.
            Interpretation only matters for the edge direction check — the
            formula is symmetric when you treat "buying NO at cost p_no"
            as "selling YES at (1 - p_no)".
        fraction: Fractional-Kelly multiplier, e.g. 0.25 = quarter Kelly.
        cap: Absolute ceiling on the returned fraction (safety net).

    Returns:
        Fraction of bankroll to stake, in [0, cap]. Zero if no edge or
        inputs are out of bounds.
    """
    if not (0.0 < p < 1.0):
        return 0.0
    if not (0.0 <= q <= 1.0):
        return 0.0
    if side != Side.BUY:
        # We express everything as BUYs on the correct token.
        return 0.0

    if outcome_is_yes:
        # We're buying the token whose price `p` is the market-implied
        # probability of the event we think has true probability `q`.
        if q <= p:
            return 0.0
        f_star = (q - p) / (1.0 - p)
    else:
        # Buying the opposite (NO) token at price (1 - p_yes). Let the
        # caller pass p_no for `p` and q_no = (1 - q_yes) for `q` and this
        # same branch works.
        if q <= p:
            return 0.0
        f_star = (q - p) / (1.0 - p)

    scaled = fraction * f_star
    if scaled <= 0:
        return 0.0
    return min(scaled, cap)


def shrink_toward_market(
    q_model: float,
    p_market: float,
    confidence: float,
) -> float:
    """Bayesian blend of the model's fair value toward the market price.

    When we're uncertain, the market price is a strong prior — it's the
    consensus of all other participants. `confidence = 1.0` trusts the
    model fully, `confidence = 0.0` collapses to the market price (no
    edge, no trade).

    q_adj = w * q_model + (1 - w) * p_market,  w = clip(confidence, 0, 1)
    """
    w = max(0.0, min(1.0, confidence))
    return w * q_model + (1.0 - w) * p_market


def polymarket_taker_fee(price: float, peak_fee: float = 0.018) -> float:
    """Approximate Polymarket dynamic taker fee curve.

    Fee peaks at the mid-book (p = 0.5) and decays toward the tails.
    The exact curve is proprietary; this `0.018 * 4 * p * (1-p)` shape
    is the documented practitioner approximation.

    For a maker-only strategy, pass 0.0 to ignore.
    """
    if not (0.0 < price < 1.0):
        return peak_fee
    return peak_fee * 4.0 * price * (1.0 - price)


def kelly_size_usd(
    bankroll: float,
    q: float,
    p: float,
    confidence: float = 1.0,
    fraction: float = 0.25,
    cap: float = 0.25,
    max_position: float | None = None,
) -> float:
    """Convenience: Kelly fraction times bankroll, with shrinkage + caps.

    Returns a dollar amount to stake.
    """
    if bankroll <= 0:
        return 0.0
    q_adj = shrink_toward_market(q, p, confidence)
    f = kelly_fraction(q_adj, p, fraction=fraction, cap=cap)
    size = f * bankroll
    if max_position is not None:
        size = min(size, max_position)
    return max(0.0, size)
