# Sterling — 60-Day Paper Trading Evaluation Protocol

**Purpose:** Validate signal quality and operational reliability before any live money transitions. All positions are hypothetical ($250 CAD per slot, max 4 simultaneous). No real trades are placed.

**Evaluation window:** 60 calendar days from the date of the first GitHub Actions run.  
**Record start date:** _____________  
**Record end date:** _____________  (start + 60 days)

---

## Part A — Daily Record

Keep a simple log (spreadsheet or notebook). Each market day:

### What to record

| Field | How to get it |
|---|---|
| Date | Calendar date |
| XIC.TO closing price | Yahoo Finance or Wealthsimple quote |
| Paper portfolio value | See calculation below |
| Any new alerts fired | Check Telegram / `data/paper_trades.csv` |
| Alert looked wrong? | See §D |

### Paper portfolio value calculation

Sterling tracks hypothetical positions at $250 CAD each (4 slots = $1,000 total). Use the paper_trades.csv to reconstruct:

1. For each open paper position: look up today's closing price for the ticker
2. Multiply shares hypothetical × current price (shares = $250 / entry_zone_low)
3. Sum all open positions + any remaining "cash" (slots not yet used × $250)
4. Record as today's paper portfolio value

You don't need to do this every day — a weekly snapshot is sufficient for the 60-day pass/fail check. Do record it on any day an alert fires.

### Minimum viable log

```
Date        XIC_close  Paper_value  Notes
2026-05-01  30.45      1000.00      No signals
2026-05-06  30.62      1018.50      BUY RY.TO @ 138.20 → entered $250 hypothetical
2026-05-08  30.71      1024.10      
...
```

---

## Part B — Success Criteria (Day 60 Pass)

**Primary criterion:**
> Paper Sterling total return ≥ XIC.TO total return − 5 percentage points  
> over the 60-day window.

Example: If XIC.TO returns +4.2% over 60 days, Sterling must return ≥ −0.8% to pass.

**Supporting criteria (all should hold):**

| Metric | Pass threshold |
|---|---|
| Win rate on closed paper trades | ≥ 45% |
| No single position loss | > −15% |
| Alerts fire at least 3× per week on average | ≥ 36 total alerts |
| Zero "obviously wrong" alerts (see §D) that would have caused real harm | 0 harmful alerts |
| GitHub Actions ran without failure on ≥ 90% of scheduled runs | ≥ 162 / 180 runs |

**How to compute 60-day returns:**

```
XIC return = (XIC_close_day60 - XIC_close_day0) / XIC_close_day0 × 100

Paper return = (paper_value_day60 - 1000) / 1000 × 100
```

Day 0 is the closing price on the calendar day before the first Actions run.

---

## Part C — Failure Criteria (Do Not Deploy Live)

**Hard stop — do not proceed to live under any of these conditions:**

1. **Return gap > 10pp:** Paper Sterling trails XIC.TO by more than 10 percentage points at day 60.
   > Example: XIC +4.2%, Paper −6.5% → gap = 10.7pp → FAIL

2. **Catastrophic single loss:** Any single paper position loses > 20% before stop-loss fires. This indicates the stop-loss logic is broken or the signal is severely miscalibrated.

3. **Signal silence:** Fewer than 10 total actionable alerts over 60 days. The strategy is too conservative to generate useful signal flow.

4. **Systematic obvious errors:** More than 2 alerts in §D category "clearly wrong direction" that, if acted upon with real money, would have been immediately harmful.

5. **Operational failure rate > 20%:** GitHub Actions failed on more than 36 of 180 scheduled runs, meaning the system is not reliable enough for real deployment.

**If any hard stop is hit:** archive `data/paper_trades.csv`, review DECISIONS.md and the sweep results, and revisit the signal model before re-starting the protocol.

---

## Part D — Logging Obviously Wrong Alerts

An alert is "obviously wrong" if it recommends action that directly contradicts publicly available information you can see at the time. Examples:

- BUY recommended the day after a company announces bankruptcy / trading halt
- SELL recommended on a ticker that has already suspended trading (price = 0)
- Score of 90/100 on a ticker with volume < 1,000 shares/day (liquidity filter failure)
- CDR ticker alerts without the "📝 PAPER TRADE" prefix (means paper mode is broken)

**When an obviously wrong alert fires:**

Log it in a separate `ALERT_EXCEPTIONS.md` with:

```
Date: 2026-05-14
Ticker: XYZ.TO
Signal: BUY @ 92.0 confidence
Why wrong: Company announced insolvency on 2026-05-13; trading halted
Data gap: Fundamental score used stale cached P/E; did not catch halt
Action taken: None (paper mode — no real money at risk)
```

Do NOT manually override or delete the `paper_trades.csv` row. The log should be complete even for bad signals — the purpose of the 60-day period is to catch exactly these cases.

---

## Part E — Post-60-Day Decision Tree

```
Day 60 evaluation:
│
├─ All pass criteria met AND no hard-stop triggers?
│     │
│     └─► Proceed to live — but start with 25% of account ($250 CAD)
│           for the first 30 days, then review before scaling to full size.
│
├─ Pass on primary criterion but ≥1 supporting criterion missed?
│     │
│     └─► Extend paper period by 30 days. Investigate the failing criterion.
│           Do not go live until extended period also passes.
│
└─ Any hard-stop trigger hit?
      │
      └─► Stop. Archive results. Diagnose root cause before any further testing.
            Minimum: review DECISIONS.md, re-run sweep with updated parameters,
            restart 60-day protocol from day 0.
```

---

## Part F — Live Transition Checklist (only after pass)

- [ ] Run `sterling analyze` locally with real API keys — confirm Telegram works without "PAPER TRADE" prefix (remove `--paper` flag)
- [ ] Read Wealthsimple fee schedule — confirm per-trade commission impact on $250 positions
- [ ] Set a hard stop-loss rule: if live account drops > 8% in any 30-day period, halt trading and re-evaluate
- [ ] Set a calendar reminder for 30-day live review
- [ ] Never automate order placement — signals are advisory, execution is always manual
