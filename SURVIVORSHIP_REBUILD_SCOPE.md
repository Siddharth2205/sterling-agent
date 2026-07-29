# Scope: Survivorship-Free Rebuild

**Purpose:** decide whether buying point-in-time (survivorship-free) data is worth it.
**The one question this rebuild answers:** does the model earn its keep by *avoiding* stocks
that later collapsed/delisted — the one source of edge our survivor-only test can't see?

Context: on free data, the model's selection alpha vs the equal-weight universe of the same
survivors was **+2.3%/yr, t = 0.18** (indistinguishable from zero). The only untested upside is
"avoids losers," which requires data that includes the losers. See [DIAGNOSIS.md](DIAGNOSIS.md).

---

## 1. What data is actually required

Three things, in priority order. The first is the whole point; the others are quality upgrades.

| # | Requirement | Why it matters | Free data today |
|---|---|---|---|
| A | **Point-in-time universe incl. delisted names** — which tickers were index members / liquid *as of each historical date*, and the delisted ones' price history through their final day | Removes survivorship bias at the source; lets the model be tested on the losers it's supposed to avoid | ❌ yfinance = current survivors only |
| B | **Delisting returns** — the actual return on the delisting event (≈ −100% for bankruptcies; acquisition price for buyouts) | Without this, a delisted name is silently dropped and survivorship bias sneaks back in. This is the make-or-break correctness detail | ❌ absent |
| C | **Deeper point-in-time fundamentals** — as-reported (not restated) statements going back ~10y | Our fundamental features were only ~1/3 populated (yfinance ≈ 4–5y, restated). Deeper as-reported history strengthens the fundamental axis and removes a residual look-ahead | ⚠️ shallow & restated |

### Candidate vendors (approx. — verify current pricing/coverage before buying)

| Vendor | Covers | Delisted + PIT membership | Fundamentals | Approx. cost | Notes |
|---|---|---|---|---|---|
| **Sharadar** (via Nasdaq Data Link) | US | ✅ SEP prices incl. delisted; ACTIONS; TICKERS with date ranges | ✅ SF1 as-reported (ARQ dimension = point-in-time) | ~$50–150/mo, **monthly** | Cleanest fit; monthly billing = cheap trial. **US only.** |
| **Norgate Data** | US, ASX (+some) | ✅ delisted + historical index constituents; handles delisting returns | limited | ~$30–80/mo | Great for survivorship; weaker fundamentals; **thin TSX.** |
| **EODHD** | Global incl. **TSX** | ⚠️ delisted tickers available; PIT membership weaker | ✅ fundamentals add-on | ~$20–100/mo | Best Canadian coverage of the three; verify delisting-return quality. |
| CRSP | US | ✅ gold standard incl. delisting returns | via Compustat | $$$$ / academic | Overkill/expensive unless you have university access. |

**The Canadian problem:** Sterling is TSX-native, but survivorship-free TSX data is the hardest and
priciest to get. Sharadar (the cleanest) is **US-only**. Realistic options: (a) run the decisive
experiment on **US names only** first (cheap, clean), or (b) pay up for EODHD/Norgate to get TSX.

---

## 2. Code changes (grounded in the pipeline already built)

| Module | Change | Effort |
|---|---|---|
| `research/universe.py` | Static list → **time-varying** `members_asof(date)`; add delisted tickers | Rewrite (medium) |
| `research/dataset.py` | Swap yfinance for the vendor API; fetch delisted histories; store delisting date + return | Rewrite adapter (medium) |
| `research/hist_fundamentals.py` | Replace yfinance-derived reconstruction with vendor as-reported PIT feed | Rewrite (medium) |
| `research/labels.py` | **Critical:** forward return must apply the delisting return when a name delists inside the window (not drop it) | Careful edit + tests (high care) |
| `research/features.py` | Reuse feature *definitions*; iterate the time-varying membership instead of a fixed dict | Light edit |
| `research/model.py` | **Reuse as-is** (walk-forward, HistGB, gate) | None |
| `research/portfolio_sim.py` | Reuse; equal-weight-universe baseline now uses real PIT membership; holdings can delist mid-period | Light edit |
| `research/validate.py`, tests | Reuse; extend tests for delisting-return handling | Light |

The modeling and validation layers — the hard intellectual work — are **done and reusable**. The
rebuild is ~80% data-adapter plumbing, ~20% getting delisting-aware labels correct.

---

## 3. The one trap that invalidates everything

**Delisting-return handling (requirement B).** The single most common way survivorship-free
rebuilds silently fail: a stock delists, the code can't find a forward price, and it drops the row.
That drops exactly the disasters you bought the data to measure — reintroducing survivorship bias
while *looking* unbiased. `labels.py` must, for any name that delists inside the forward window,
use the delisting return (bankruptcy ≈ −100%, buyout = deal price). If the chosen vendor doesn't
supply clean delisting returns, the whole exercise is compromised. **Vet this before subscribing.**

---

## 4. Effort estimate

- Vendor integration (universe + prices + fundamentals adapters): ~2–4 focused sessions.
- Delisting-aware labels + tests: ~1 session (high care).
- Re-run full pipeline (features → walk-forward → net-of-cost + survivorship-neutral sim): hours — it's built.
- **Total: ~1–2 weeks part-time, dominated by data plumbing, not modeling.**

---

## 5. Why it might still come back null (set expectations honestly)

- Selection alpha among survivors is already ~0 (t=0.18). Better data only helps if the
  avoid-losers effect is **large**; a small effect confirms the null more rigorously.
- More/cleaner data raises the ceiling on data *quality*, not on whether a weak signal exists.
- Retail-grade daily OHLCV + fundamentals is the same information thousands of quants mine; a
  durable edge from these features alone is the exception, not the rule.

---

## 6. Recommended decision: run the cheap, decisive experiment first

Don't commit to an annual data spend. The highest-information, lowest-cost move:

1. **One month of Sharadar** (US-only, includes delisted + point-in-time fundamentals) — on the
   order of ~$50–150, cancellable.
2. Rebuild the data adapters for that feed (the ~1–2 week effort above), keeping the model/validation
   layers untouched.
3. Re-run the **exact** survivorship-neutral, net-of-cost test we already have.

**Decision rule:**
- If survivorship-free US selection alpha is still ~0 → **stop.** The model has no edge; index/equal-weight for real money. You'll have spent ~$100 to know for certain.
- If a real avoid-losers edge appears (materially positive, t > 2, survives costs) → **then** consider
  paying up for TSX survivorship-free data and a live rebuild.

Buying multi-year / multi-market data up front, before that ~$100 US-only test, is the expensive way
to answer a question you can answer cheaply.
