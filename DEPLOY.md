# Sterling — GitHub Actions Deployment Guide

Paper-trading mode only. No live orders are placed. Live transition is manual after the 60-day evaluation defined in `PAPER_TEST_PROTOCOL.md`.

---

## Prerequisites

- Git installed and a GitHub account
- Your four secrets ready (configured in Step 2):
  - `FINNHUB_API_KEY` — from finnhub.io (free tier)
  - `TELEGRAM_BOT_TOKEN` — from @BotFather on Telegram
  - `TELEGRAM_CHAT_ID` — your personal chat ID (run `python setup_telegram.py` to find it)
  - `STERLING_WATCHLIST` — comma-separated list of tickers to paper-trade

---

## Step 1 — Push the repo to GitHub as private

`data/portfolio.json` stays gitignored — holdings are private. The workflow reads tickers from the `STERLING_WATCHLIST` secret instead.

```bash
# On GitHub.com: click New repository → set visibility to Private → Create
# Then back in your terminal:

git remote add origin https://github.com/YOUR_USERNAME/sterling-agent.git
git branch -M main
git push -u origin main
```

**Where to find "New repository":** GitHub.com → top-right "+" icon → "New repository". On the creation page, select **Private** before clicking Create Repository.

---

## Step 2 — Configure the four secrets

Navigate to: **Your repo → Settings → Secrets and variables → Actions**

The path in the GitHub UI is:
```
https://github.com/YOUR_USERNAME/sterling-agent/settings/secrets/actions
```

Click **"New repository secret"** (green button, top-right of the secrets table) and add each of:

| Secret name | Value |
|---|---|
| `FINNHUB_API_KEY` | Your Finnhub API key |
| `TELEGRAM_BOT_TOKEN` | Your bot token from @BotFather (format: `123456:ABC-DEF...`) |
| `TELEGRAM_CHAT_ID` | Your personal chat ID (a number like `1009988605`) |
| `STERLING_WATCHLIST` | Comma-separated tickers to paper-trade (no spaces around commas) |

**`STERLING_WATCHLIST` format:** comma-separated, no spaces, uppercase. Example starter list:
```
SHOP.TO,ENB.TO,BNS.TO,CNR.TO,RY.TO,TD.TO,CNQ.TO,SU.TO,AAPL.NE,MSFT.NE,NVDA.NE,GOOGL.NE
```
TSX tickers use `.TO` suffix; CDR tickers (NEO Exchange) use `.NE`. You can include as many as you want — the workflow runs all of them each session.

Secrets are write-only after saving — you cannot read them back. If you mistype one, delete it and re-create.

---

## Step 3 — Enable Actions and set write permissions

**Enable Actions (if disabled):**
Navigate to: **Settings → Actions → General**

Under "Actions permissions", select **"Allow all actions and reusable workflows"** and click Save.

**Grant write permission for the commit step:**
On the same page, scroll to **"Workflow permissions"** and select:
> **"Read and write permissions"**

Click Save. This allows the workflow to commit `data/paper_trades.csv` back to the repo after each run.

---

## Step 4 — Verify the first scheduled run

The workflow fires at three times daily (Mon–Fri):
- 09:30 ET (13:30 UTC) — TSX open
- 12:30 ET (16:30 UTC) — midday
- 15:55 ET (19:55 UTC) — TSX close

GitHub Actions schedule triggers can be delayed up to ~15 minutes during high load.

**To trigger immediately without waiting:**

Navigate to: **Actions → Sterling Paper Trading** (left sidebar) → **"Run workflow"** button (top-right of the workflow table) → Select branch `main` → **"Run workflow"**.

**Where to verify it ran:**
- **Actions tab** → click the run → expand "Run sterling analyze --paper --notify" step to see signal output
- **Code tab** → `data/paper_trades.csv` — should appear (or gain a new row) after the first run with at least one actionable signal
- **Telegram** — you should receive a message prefixed "📝 PAPER TRADE —" within a few minutes of the run finishing

**If no Telegram message arrives:**
1. Check the Actions log for `[OK] Notifications sent for N actionable signals.` — if N=0, no tickers had actionable signals that run (HOLD is the neutral state; try again at market open)
2. Check secrets are correct: re-enter `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`
3. Confirm your bot is started: send `/start` to your bot in Telegram before expecting messages

---

## Step 5 — Monitor ongoing runs

- **Actions tab** shows all runs with pass/fail status and timing
- A green `chore: paper trades update [skip ci]` commit appearing in your commit history means the run completed and logged at least one signal
- If a run fails (red X), click it → expand the failed step → the error is usually a missing secret or a yfinance rate-limit (transient; the next scheduled run will retry)

---

## Teardown / Pause

To stop paper trading without deleting the repo:
- **Settings → Actions → General → "Disable Actions"** — stops all future scheduled runs
- Or: delete `.github/workflows/sterling.yml` and push

Your accumulated `data/paper_trades.csv` remains in the repo for evaluation.
