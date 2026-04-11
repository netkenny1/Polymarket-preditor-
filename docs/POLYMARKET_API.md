# Getting API access to Polymarket

This guide takes you from zero to placing your first programmatic order on
the BTC 5-minute up/down market. Polymarket runs on **Polygon (chain 137)**
with **USDC.e** as collateral. All trading goes through the **CLOB API** at
`https://clob.polymarket.com`.

If you just want to read market data (prices, token IDs, slugs), the
**Gamma API** (`https://gamma-api.polymarket.com`) is public and
unauthenticated — no key needed.

---

## 1. Wallet setup

Polymarket always routes your funds through a **proxy wallet** that is
different from your signing EOA. Three modes exist:

| `signature_type` | Signer (private key)    | Funder (USDC holder)              | Typical user              |
| ---------------- | ----------------------- | --------------------------------- | ------------------------- |
| `0`              | Your EOA                | Same EOA                          | Fresh bot wallet          |
| `1`              | Magic.link-derived key  | Magic-deployed Safe proxy         | Signed up with email      |
| `2`              | Your MetaMask / browser | Gnosis-Safe 1-of-1 proxy          | Connected browser wallet  |

**For a bot, the cleanest path is `signature_type=0`**: generate a fresh EOA
with `eth_account`, treat its private key as a production secret, and fund
it directly. The signer and funder are the same address; no proxy discovery
needed.

**To find your funder if you already have a Magic/Browser account:** log
into polymarket.com, open the Deposit modal, copy the address shown there.
That is your proxy wallet. It is NOT your MetaMask address for type 1/2.

---

## 2. Funding the proxy

Send **USDC.e (bridged USDC) on Polygon** to the proxy address.

- **From an exchange**: Binance, OKX, Kraken, Coinbase all support USDC
  withdrawals to Polygon. Use the "Polygon" network, not "Ethereum".
- **From Ethereum mainnet**: bridge via the official Polygon PoS bridge,
  Across, or Polymarket's built-in bridge (Deposit page, accepts ETH/BTC/SOL).
- **Fiat**: MoonPay via the website, ~$30 minimum.

Gas on Polygon is ~fractions of a cent. A practical minimum for testing
is $5–$10.

**Do not send via Ethereum mainnet, BSC, or Solana** — Polymarket only
recognizes funds on Polygon. Wrong-chain sends are typically lost.

---

## 3. Generating CLOB API credentials (L2)

The `(api_key, api_secret, api_passphrase)` tuple is issued by
`POST https://clob.polymarket.com/auth/api-key`. You do NOT do this through
the website — you do it from a script that signs an EIP-712 payload with
your private key. The easiest way is via `py-clob-client`:

```python
from py_clob_client.client import ClobClient
from py_clob_client.constants import POLYGON

HOST = "https://clob.polymarket.com"
client = ClobClient(
    HOST,
    key="0xYOUR_PRIVATE_KEY",
    chain_id=POLYGON,        # 137
    signature_type=0,        # 0=EOA, 1=Magic, 2=Browser Safe proxy
    funder="0xFUNDER_ADDRESS",  # same as signer for signature_type=0
)

# This call either generates new credentials or returns the existing ones
# associated with (key, nonce=0). Save them — losing them forces a new nonce.
creds = client.create_or_derive_api_creds()
print(creds.api_key, creds.api_secret, creds.api_passphrase)
client.set_api_creds(creds)
```

Run this **once** from a secure machine, then store the three values in
your secrets manager or `.env`. Subsequent runs just `set_api_creds(...)`
from env vars — no need to re-create.

---

## 4. L1 vs L2 auth in plain English

- **L1** = wallet private key signing an EIP-712 message.
  Required for: creating API keys, signing individual **order payloads**
  (Polymarket orders are EIP-712 structs), and any on-chain action.
- **L2** = HMAC-SHA256 of the request with your `(api_key, api_secret,
  api_passphrase)`.
  Required for: authenticated REST endpoints — posting the signed order to
  the CLOB, cancelling, listing open orders, fetching balances.

**Rule of thumb**: the order payload is L1-signed; the HTTP request that
carries it is L2-authenticated. `py-clob-client` hides this — you just call
`client.create_and_post_order(...)`.

---

## 5. Finding the BTC 5-minute market

Slug pattern: **`btc-updown-5m-{window_start_unix_seconds}`**

where `window_start_unix_seconds = now - (now % 300)`. New slug every 300
seconds (the 15-minute variant is `btc-updown-15m-{ts}` with `% 900`).

Query via Gamma — no auth:

```bash
curl "https://gamma-api.polymarket.com/markets?slug=btc-updown-5m-1771168800"
```

The response contains `conditionId`, `clobTokenIds` (JSON array of the
Up/Down ERC-1155 token IDs), `question`, and `endDate`. Feed those IDs to
the CLOB client to get the order book and place orders.

In this repo, `polymarket_bot/bot.py::_get_or_fetch_market` does exactly
this lookup and caches the result until the window rolls over.

---

## 6. Required allowances (one-time)

Before your first order will match, you must approve USDC and the CTF
(Conditional Token Framework) contracts to spend from your proxy. This
is a one-time on-chain transaction. The Polymarket team maintains a
public gist with the exact script:
https://gist.github.com/poly-rodr/44313920481de58d5a3f6d1f8226bd5e

**Symptom of forgetting it**: orders post successfully but never match.

---

## 7. Rate limits and common pitfalls

- **Gamma**: ~4,000 req/10s global; `/markets` ~300/10s. Don't poll faster
  than every ~500ms for the same slug.
- **CLOB**: ~100+ req/min for authenticated endpoints; 5 concurrent
  WebSocket connections per IP.
- **Cloudflare** throttles rather than 429-rejects — unexpected latency is
  usually rate limiting.

Pitfalls you will hit:

1. **Wrong `signature_type`** → "invalid signature" errors. If you signed
   up via email/Magic, you are type 1 — not 0.
2. **No allowances** → orders post but never fill.
3. **Funder mismatch** → you sent USDC to your MetaMask EOA but your
   `signature_type=2` proxy is a Safe at a different address.
4. **Lost API creds nonce** → unrecoverable, generate new ones.
5. **Taker fees on 5-min markets** eat the edge. Always post maker-only
   (`POST_ONLY`). This repo enforces it by default via
   `StrategyConfig.post_only=True`.

---

## 8. Environment variables this bot expects

Put these in `.env` at the repo root — `polymarket_bot/config.py` loads
them automatically via `python-dotenv`.

```bash
# Required for live mode
POLYMARKET_PRIVATE_KEY=0x...        # signer EOA private key
POLYMARKET_FUNDER=0x...             # proxy wallet (= signer for type 0)
POLYMARKET_API_KEY=...              # L2 api key (from create_or_derive_api_creds)
POLYMARKET_API_SECRET=...           # L2 secret
POLYMARKET_API_PASSPHRASE=...       # L2 passphrase

# Optional overrides
PAPER_TRADING=true                  # default; set false for real orders
INITIAL_CAPITAL_USD=100
KELLY_FRACTION=0.25
MIN_EDGE=0.03
MAX_POSITION_USD=25
MAX_PORTFOLIO_EXPOSURE_USD=100
LOG_LEVEL=INFO
```

For `paper` and `backtest` modes, none of the Polymarket keys are
required — `paper` uses a simulated client and `backtest` replays real
Binance klines.

---

## 9. Quick checklist

- [ ] Create or import a Polygon wallet; note its EOA address.
- [ ] Fund the proxy address (EOA itself for signature_type=0) with USDC.e
      on Polygon.
- [ ] Run the allowance-setting gist once.
- [ ] Run the `create_or_derive_api_creds()` snippet in §3.
- [ ] Paste all 5 `POLYMARKET_*` values into `.env`.
- [ ] `python -m polymarket_bot backtest --days 7 --show-trades` to
      confirm the pipeline is green.
- [ ] `python -m polymarket_bot paper` for a few hours to validate
      live plumbing without risking capital.
- [ ] `python -m polymarket_bot live` to go live (start with tiny
      `MAX_POSITION_USD`).

---

## 10. Authoritative references

- Polymarket docs: https://docs.polymarket.com/
- CLOB quickstart: https://docs.polymarket.com/developers/CLOB/quickstart
- Authentication: https://docs.polymarket.com/api-reference/authentication
- Rate limits: https://docs.polymarket.com/api-reference/rate-limits
- Gamma markets: https://docs.polymarket.com/developers/gamma-markets-api/get-markets
- `py-clob-client`: https://github.com/Polymarket/py-clob-client
- Allowance-setting gist: https://gist.github.com/poly-rodr/44313920481de58d5a3f6d1f8226bd5e
