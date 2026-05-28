"""Sterling CLI -- one entry point for all commands."""

import io
import json
import logging
import sys
from pathlib import Path

import click

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("sterling")


def _fix_windows_utf8() -> None:
    # Rewrap stdout/stderr for UTF-8 on Windows cp1252 consoles.
    # Guarded by hasattr(.buffer) so it is a no-op under Click's CliRunner
    # (which supplies io.StringIO without a .buffer attribute).
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


@click.group()
@click.version_option(version="0.1.0", prog_name="sterling")
def cli():
    """Sterling -- Canadian portfolio analysis agent."""
    _fix_windows_utf8()


# -- Portfolio commands --------------------------------------------------------

@cli.command()
@click.argument("ticker")
@click.argument("shares", type=float)
@click.argument("avg_cost_cad", type=float)
@click.option("--currency", default="CAD", help="Purchase currency (CAD or USD)")
def add(ticker, shares, avg_cost_cad, currency):
    """Add or update a holding. Example: sterling add SHOP.TO 5 98.50"""
    from sterling import portfolio
    h = portfolio.add_holding(ticker, shares, avg_cost_cad, currency)
    click.echo(f"[OK] Added {h['ticker']}: {h['shares']} shares @ ${h['avg_cost_cad']:.2f} {h['currency']}")


@cli.command()
@click.argument("ticker")
def remove(ticker):
    """Remove a holding from the portfolio."""
    from sterling import portfolio
    if portfolio.remove_holding(ticker):
        click.echo(f"[OK] Removed {ticker.upper()}")
    else:
        click.echo(f"[ERR] {ticker.upper()} not found in portfolio", err=True)


@cli.command()
@click.argument("tickers", nargs=-1, required=True)
def watch(tickers):
    """Add tickers to the watchlist. Example: sterling watch ENB.TO BNS.TO CNR.TO"""
    from sterling import portfolio
    for ticker in tickers:
        portfolio.add_to_watchlist(ticker)
        click.echo(f"[OK] Watching {ticker.upper()}")


@cli.command()
@click.argument("tickers", nargs=-1, required=True)
def unwatch(tickers):
    """Remove tickers from the watchlist."""
    from sterling import portfolio
    for ticker in tickers:
        portfolio.remove_from_watchlist(ticker)
        click.echo(f"[OK] Removed {ticker.upper()} from watchlist")


@cli.command("portfolio")
def show_portfolio():
    """Show current holdings and P&L."""
    from sterling import portfolio, data_feed

    port = portfolio.get_portfolio()
    tickers = list(port["holdings"].keys())

    current_prices = {}
    if tickers:
        click.echo("Fetching live prices...")
        for t in tickers:
            try:
                q = data_feed.get_quote(t)
                current_prices[t] = q["price"]
            except Exception:
                pass

    summary = portfolio.portfolio_summary(current_prices)

    if not summary["positions"]:
        click.echo("No holdings. Add one with: sterling add SHOP.TO 5 98.50")
        return

    click.echo(f"\n{'TICKER':<12} {'SHARES':>8} {'AVG COST':>10} {'PRICE':>10} {'VALUE':>10} {'P&L':>10} {'P&L%':>8} {'WT%':>6}")
    click.echo("-" * 76)

    for p in summary["positions"]:
        price_str = f"${p['current_price']:.2f}" if p['current_price'] else "-"
        val_str = f"${p['current_value']:.2f}" if p['current_value'] else "-"
        pnl_str = f"${p['unrealized_pnl']:+.2f}" if p['unrealized_pnl'] is not None else "-"
        pnl_pct_str = f"{p['pnl_pct']:+.1f}%" if p['pnl_pct'] is not None else "-"
        click.echo(
            f"{p['ticker']:<12} {p['shares']:>8.2f} ${p['avg_cost_cad']:>9.2f} "
            f"{price_str:>10} {val_str:>10} {pnl_str:>10} {pnl_pct_str:>8} {p['weight_pct']:>5.1f}%"
        )

    click.echo("-" * 76)
    total_val = f"${summary['total_value_cad']:.2f}" if summary['total_value_cad'] else "-"
    total_pnl = f"${summary['total_unrealized_pnl']:+.2f}" if summary['total_unrealized_pnl'] is not None else "-"
    click.echo(f"  Cost basis: ${summary['total_cost_cad']:.2f}  |  Current: {total_val}  |  P&L: {total_pnl}")

    if summary["watchlist"]:
        click.echo(f"\n  Watchlist: {', '.join(summary['watchlist'])}")


# -- Analysis -----------------------------------------------------------------

@cli.command()
@click.option("--ticker", "-t", multiple=True, help="Specific tickers to analyze (default: portfolio + watchlist)")
@click.option("--notify/--no-notify", default=False, help="Send Telegram alerts for actionable signals")
@click.option("--json-output", is_flag=True, help="Output raw JSON")
@click.option("--paper/--no-paper", default=False,
              help="Paper-trade mode: log all actionable signals to data/paper_trades.csv "
                   "and prefix Telegram alerts with 'PAPER TRADE'")
@click.option("--watchlist-from-env", is_flag=True, default=False,
              help="Read tickers from STERLING_WATCHLIST env var (comma-separated). "
                   "Takes precedence over data/portfolio.json. Required in CI.")
def analyze(ticker, notify, json_output, paper, watchlist_from_env):
    """Run multi-factor analysis on portfolio and watchlist."""
    import os
    from sterling import config, analyst, portfolio, notifier

    cfg = config.validate_optional()
    if cfg["missing"]:
        click.echo(f"Warning: missing env vars {cfg['missing']} -- some signals will be degraded", err=True)

    if watchlist_from_env:
        raw = os.environ.get("STERLING_WATCHLIST", "")
        if not raw.strip():
            click.echo(
                "[ERR] --watchlist-from-env set but STERLING_WATCHLIST is empty or not set.\n"
                "      Add STERLING_WATCHLIST as a GitHub Secret (comma-separated tickers,\n"
                "      e.g. SHOP.TO,ENB.TO,NVDA.NE).",
                err=True,
            )
            raise SystemExit(1)
        tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    elif ticker:
        tickers = list(ticker)
    else:
        port = portfolio.get_portfolio()
        tickers = list(port["holdings"].keys()) + port.get("watchlist", [])

    if not tickers:
        click.echo("No tickers. Add holdings with 'sterling add' or watchlist with 'sterling watch'.")
        return

    mode_tag = " [PAPER MODE]" if paper else ""
    click.echo(f"Analyzing {len(tickers)} tickers: {', '.join(tickers)}{mode_tag}\n")

    results = analyst.analyze_portfolio(tickers, config.FINNHUB_API_KEY or "")

    if json_output:
        click.echo(json.dumps(results, indent=2))
        return

    _print_analysis_table(results)

    actionable = [r for r in results if r.get("recommendation") in ("BUY", "ACCUMULATE", "SELL", "TRIM")]

    if paper and actionable:
        from sterling import paper_log
        for signal in actionable:
            logged_path = paper_log.log_signal(signal)
        click.echo(f"\n[PAPER] {len(actionable)} signal(s) logged to {logged_path}")

    if notify and config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
        for signal in actionable:
            notifier.send_signal(
                signal,
                config.TELEGRAM_BOT_TOKEN,
                config.TELEGRAM_CHAT_ID,
                paper=paper,
            )
        click.echo(f"\n[OK] Notifications sent for {len(actionable)} actionable signals.")


def _print_analysis_table(results: list):
    rec_colors = {
        "BUY":        "\033[32m",
        "ACCUMULATE": "\033[36m",
        "HOLD":       "\033[37m",
        "TRIM":       "\033[33m",
        "SELL":       "\033[31m",
        "ERROR":      "\033[35m",
    }
    reset = "\033[0m"

    click.echo(f"{'TICKER':<12} {'REC':<11} {'CONF':>5} {'PRICE':>8} {'ENTRY ZONE':>20} {'STOP':>8} {'TARGET':>8} {'R:R':>5}")
    click.echo("-" * 83)

    for r in results:
        if "error" in r:
            click.echo(f"{r['ticker']:<12} ERROR: {r['error']}")
            continue
        rec = r.get("recommendation", "-")
        col = rec_colors.get(rec, "")
        conf = r.get("confidence", 0)
        price = f"${r['current_price']:.2f}" if r.get("current_price") else "-"
        ez = r.get("entry_zone")
        entry_str = f"${ez[0]:.2f}-${ez[1]:.2f}" if ez else "-"
        stop = f"${r['stop_loss']:.2f}" if r.get("stop_loss") else "-"
        target = f"${r['target']:.2f}" if r.get("target") else "-"
        rr = f"{r['risk_reward']:.1f}x" if r.get("risk_reward") else "-"
        click.echo(f"{r['ticker']:<12} {col}{rec:<11}{reset} {conf:>5.1f} {price:>8} {entry_str:>20} {stop:>8} {target:>8} {rr:>5}")

    click.echo("\n  SIGNAL DETAIL")
    click.echo("-" * 83)
    click.echo(f"  {'TICKER':<12} {'TECH':>6} {'FUND':>6} {'SENT':>6} {'MACRO':>6} {'INSD':>6}   THESIS")
    click.echo(f"  {'------':<12} {'----':>6} {'----':>6} {'----':>6} {'----':>6} {'----':>6}")
    for r in results:
        if "error" in r:
            continue
        s = r.get("signals", {})
        thesis = r.get("thesis", "")[:70]
        click.echo(
            f"  {r['ticker']:<12} {s.get('technical',0):>6.0f} {s.get('fundamental',0):>6.0f} "
            f"{s.get('sentiment',0):>6.0f} {s.get('macro',0):>6.0f} {s.get('insider',0):>6.0f}   {thesis}"
        )
        if r.get("fx_warning"):
            click.echo(f"  {'':12} [!] {r['fx_warning']}")


# -- Scan ---------------------------------------------------------------------

@cli.command()
@click.option("--digest", is_flag=True, help="Send ONE Telegram digest of the top 15 setups")
def scan(digest):
    """Scan ~220 tickers for the top 15 setups (read-only, no paper trades)."""
    from datetime import datetime, timezone
    from sterling import config
    from sterling.scanner import run_scan, top_setups, format_scan_digest, send_scan_digest

    cfg = config.validate_optional()
    if cfg["missing"]:
        click.echo(f"Warning: missing env vars {cfg['missing']} -- some data will be degraded", err=True)

    def progress(done, total):
        click.echo(f"  scanned {done}/{total}")

    click.echo("Sterling Scan -- scanning ~220 tickers...\n")
    results = run_scan(config.FINNHUB_API_KEY or "", progress_cb=progress)

    top = top_setups(results, 15)

    if not top:
        click.echo("No results. Check internet connection and API keys.")
        return

    click.echo(f"\n{'#':>3} {'TICKER':<12} {'SCORE':>6} {'REC':<12} {'TOP AXIS':<14} NOTE")
    click.echo("-" * 85)
    for i, entry in enumerate(top, 1):
        cdr_tag = " [CAD-settled]" if entry["is_cdr"] else ""
        click.echo(
            f"{i:>3} {entry['ticker']:<12} {entry['score']:>6.1f} {entry['recommendation']:<12} "
            f"{entry['top_axis']:<14} {entry['note']}{cdr_tag}"
        )

    click.echo(f"\n  Scanned {len(results)} tickers total. Top 15 by confidence shown above.\n")

    if digest:
        if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
            click.echo("[ERR] Telegram credentials not set. Cannot send digest.", err=True)
            return

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        message = format_scan_digest(top, date_str)
        ok = send_scan_digest(message, config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)

        if ok:
            click.echo("[OK] Scan digest sent to Telegram.")
        else:
            click.echo("[ERR] Failed to send scan digest.", err=True)


# -- Backtest -----------------------------------------------------------------

@cli.command()
@click.option("--years", default=3, help="Years of history to backtest", show_default=True)
@click.option("--capital", default=1000.0, help="Starting capital in CAD", show_default=True)
@click.option("--threshold", default=65.0, help="Signal score threshold to enter", show_default=True)
@click.option("--hold-days", default=20, help="Max hold period in trading days", show_default=True)
@click.option("--min-hold-days", default=10,
              help="Min days held before a time-exit fires (stop-loss still overrides)", show_default=True)
@click.option("--quick-signal", is_flag=True, default=False,
              help="Use technical-only signal instead of full 5-axis (faster, less accurate)")
@click.option("--universe", default="tsx-cdr",
              type=click.Choice(["tsx", "cdr", "tsx-cdr"], case_sensitive=False),
              help="Ticker universe to scan", show_default=True)
def backtest(years, capital, threshold, hold_days, min_hold_days, quick_signal, universe):
    """Walk-forward backtest on TSX / CDR tickers (5-axis signal by default)."""
    from sterling.backtester import run_backtest, save_results

    mode = "quick technical" if quick_signal else "full 5-axis"
    click.echo(f"Running {years}-year backtest on ${capital:.0f} CAD [{mode} signal, universe={universe}]...")
    if not quick_signal:
        click.echo("  NOTE: fundamental scores use current-snapshot values (not point-in-time).")
    click.echo("This may take 3-8 minutes while downloading price history.\n")

    result = run_backtest(
        years=years,
        capital_cad=capital,
        hold_days=hold_days,
        min_hold_days=min_hold_days,
        signal_threshold=threshold,
        use_full_signal=not quick_signal,
        universe=universe,
    )

    out_dir = save_results(result)

    stats = result.stats
    click.echo("\n" + "=" * 56)
    click.echo("  STERLING BACKTEST RESULTS")
    click.echo("=" * 56)
    click.echo(f"  Signal mode:      {stats.get('signal_mode', '?')}")
    click.echo(f"  Period:           {stats.get('start_date')} -> {stats.get('end_date')}")
    click.echo(f"  Starting capital: ${stats.get('capital_cad', 0):,.2f} CAD")
    click.echo(f"  Final value:      ${stats.get('final_value_cad', 0):,.2f} CAD")
    click.echo(f"  Total return:     {stats.get('total_return_pct', 0):+.2f}%")
    click.echo(f"  CAGR:             {stats.get('cagr_pct', 0):+.2f}%")
    click.echo(f"  Sharpe ratio:     {stats.get('sharpe', 0):.3f}")
    click.echo(f"  Sortino ratio:    {stats.get('sortino', 0):.3f}")
    click.echo(f"  Max drawdown:     {stats.get('max_drawdown_pct', 0):.2f}%")
    click.echo(f"  Win rate:         {stats.get('win_rate_pct', 0):.1f}%")
    click.echo(f"  Profit factor:    {stats.get('profit_factor', 0)}")
    click.echo(f"  Avg R:R:          {stats.get('avg_rr', 0):.2f}x")
    click.echo(f"  Total trades:     {stats.get('total_trades', 0)}")
    click.echo(f"  Min hold days:    {stats.get('min_hold_days', '?')}")
    click.echo("-" * 56)
    click.echo(f"  XIC.TO CAGR:      {stats.get('benchmark_xic_cagr_pct', 0):+.2f}% (after 1.5% FX drag)")
    beats = stats.get("beats_benchmark", False)
    verdict = "[OK] Strategy beats benchmark" if beats else "[!!] Strategy underperforms benchmark"
    click.echo(f"  Verdict:          {verdict}")
    click.echo("=" * 56)
    click.echo(f"\n  Charts and trade log saved to: {out_dir}")


# -- Sweep --------------------------------------------------------------------

@cli.command()
@click.option("--years", default=3, help="Years of history", show_default=True)
@click.option("--capital", default=1000.0, help="Starting capital in CAD", show_default=True)
@click.option("--hold-days", default=20, help="Max hold period in trading days", show_default=True)
@click.option("--universe", default="tsx-cdr",
              type=click.Choice(["tsx", "cdr", "tsx-cdr"], case_sensitive=False),
              help="Ticker universe to scan", show_default=True)
def sweep(years, capital, hold_days, universe):
    """Auto-calibrate threshold from score distribution, then run 3x3 parameter sweep.

    Calibration: scans all signals across history, computes p70/p80/p85.
    Sweep: entry_threshold {p70, p80, p85} x min_hold_days {5, 10, 20} = 9 combos.
    Downloads market data once; all runs share the same data bundle.
    Results saved to data/sweep_<timestamp>/.
    """
    import numpy as np
    from sterling.backtester import (
        load_backtest_data, scan_score_distribution, sweep_backtests,
        save_sweep_results,
    )

    click.echo(f"Running calibration + sweep ({years}yr, ${capital:.0f} CAD, universe={universe})...")
    click.echo("  Step 1/2: downloading market data + scanning score distribution...")
    click.echo("  (may take 5-10 minutes)\n")

    bundle = load_backtest_data(years=years, use_full_signal=True, universe=universe)
    records = scan_score_distribution(years=years, _preloaded=bundle)

    if not records:
        click.echo("[ERR] Calibration scan returned no scores. Check internet connection.", err=True)
        return

    post_scores = [r["post_score"] for r in records]
    p70 = round(float(np.percentile(post_scores, 70)), 2)
    p80 = round(float(np.percentile(post_scores, 80)), 2)
    p85 = round(float(np.percentile(post_scores, 85)), 2)

    click.echo(f"  Calibration complete: {len(records):,} observations")
    click.echo(f"  p70={p70:.2f}  p80={p80:.2f}  p85={p85:.2f}")
    click.echo(f"\n  Step 2/2: running 9 backtest combinations...\n")

    results = sweep_backtests(
        years=years,
        capital_cad=capital,
        thresholds=(p70, p80, p85),
        min_holds=(5, 10, 20),
        hold_days=hold_days,
        _preloaded=bundle,
        universe=universe,
    )

    out_dir = save_sweep_results(results)

    xic_cagr = results[0]["result"].stats.get("benchmark_xic_cagr_pct", 0)

    click.echo("\n" + "=" * 82)
    click.echo("  PARAMETER SWEEP — calibrated threshold x min_hold_days  (3-axis model)")
    click.echo("=" * 82)
    click.echo(f"  {'Thresh':>8}  {'Pct':>5}  {'MinHold':>7}  {'CAGR%':>7}  {'Sharpe':>7}  "
               f"{'MaxDD%':>7}  {'Trades':>7}  {'Overlay':>7}  {'Beats?':>6}")
    click.echo(f"  {'-'*8}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}")

    pct_label = {p70: "p70", p80: "p80", p85: "p85"}
    best = max(results, key=lambda x: x["result"].stats.get("cagr_pct", -999))
    for item in results:
        s = item["result"].stats
        cagr = s.get("cagr_pct", 0)
        sharpe = s.get("sharpe", 0)
        dd = s.get("max_drawdown_pct", 0)
        trades = s.get("total_trades", 0)
        overlay = s.get("macro_overlay_fire_count", 0)
        beats = "YES" if s.get("beats_benchmark") else "no"
        marker = " <--" if item is best else ""
        pct = pct_label.get(item["threshold"], "")
        click.echo(
            f"  {item['threshold']:>8.2f}  {pct:>5}  {item['min_hold']:>7}  {cagr:>+7.2f}  "
            f"{sharpe:>7.3f}  {dd:>7.2f}  {trades:>7}  {overlay:>7}  {beats:>6}{marker}"
        )

    click.echo("-" * 72)
    click.echo(f"  XIC.TO benchmark CAGR: {xic_cagr:+.2f}% (after 1.5% FX drag)")

    # Macro overlay report across all runs
    total_overlay = sum(
        item["result"].stats.get("macro_overlay_fire_count", 0) for item in results
    )
    click.echo(f"  Macro overlay fires (all runs combined): {total_overlay}")
    if total_overlay == 0:
        click.echo("  [!!] Overlay fired ZERO times. Check VIX/TSX data fetch.")

    best_s = best["result"].stats
    if best_s.get("beats_benchmark", False):
        click.echo("  Best combo beats XIC.TO. Proceed to expanded validation.")
    else:
        click.echo("  No combo beats XIC.TO. Strategy needs further development before deployment.")
    click.echo("=" * 72)
    click.echo(f"\n  Full results saved to: {out_dir}")


# -- Calibrate ----------------------------------------------------------------

@cli.command()
@click.option("--years", default=3, help="Years of history to scan", show_default=True)
@click.option("--universe", default="tsx-cdr",
              type=click.Choice(["tsx", "cdr", "tsx-cdr"], case_sensitive=False),
              help="Ticker universe to scan", show_default=True)
def calibrate(years, universe):
    """Scan score distribution across history; output percentile table.

    Run this after changing signal weights or universe. Use the printed
    percentile values to set --threshold in 'sterling backtest' and 'sterling sweep'.
    """
    import csv as csv_mod
    from sterling.backtester import scan_score_distribution, load_backtest_data, _DATA_DIR
    from datetime import datetime as dt

    click.echo(f"Scanning {years}-year score distribution (3-axis model, universe={universe})...")
    click.echo("Downloads market data once — may take 3-5 minutes.\n")

    records = scan_score_distribution(years=years, universe=universe)

    if not records:
        click.echo("[ERR] No score records returned. Check internet connection.", err=True)
        return

    scores = [r["post_score"] for r in records]
    import numpy as np
    pcts = [50, 60, 70, 75, 80, 85, 90, 95]
    vals = {p: round(float(np.percentile(scores, p)), 2) for p in pcts}

    # Save distribution CSV
    ts = dt.utcnow().strftime("%Y%m%d_%H%M%S")
    out = _DATA_DIR / f"calibration_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "score_distribution.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=records[0].keys())
        w.writeheader()
        w.writerows(records)

    # Count risk-off days
    risk_off_count = sum(1 for r in records if r["pre_score"] != r["post_score"])

    click.echo(f"  Observations: {len(records):,} (ticker × scan-date)")
    click.echo(f"  Unique tickers: {len(set(r['ticker'] for r in records))}")
    click.echo(f"  Macro overlay fired: {risk_off_count:,} times ({risk_off_count/len(records)*100:.1f}%)")
    click.echo("")
    click.echo("  Percentile   Score threshold")
    click.echo("  ----------   ---------------")
    for p, v in vals.items():
        note = " <-- suggested entry threshold (p80)" if p == 80 else ""
        click.echo(f"  p{p:<9}   {v:.2f}{note}")
    click.echo("")
    click.echo(f"  Score range: min={min(scores):.1f}  mean={np.mean(scores):.1f}  max={max(scores):.1f}")
    click.echo(f"\n  Distribution CSV saved to: {csv_path}")
    click.echo(f"\n  Use these values with: sterling sweep")
    click.echo(f"  Example: sterling backtest --threshold {vals[80]}")


# -- Diagnose -----------------------------------------------------------------

@cli.command()
@click.option("--dir", "backtest_dir", default=None,
              help="Path to backtest directory (default: most recent data/backtest_*/)")
def diagnose(backtest_dir):
    """Diagnose the most recent backtest: cash exposure, signal distribution, cost drag."""
    from sterling.diagnostics import find_latest_backtest, run_diagnostics

    if backtest_dir:
        bt_path = Path(backtest_dir)
    else:
        bt_path = find_latest_backtest()

    if bt_path is None or not bt_path.exists():
        click.echo("[ERR] No backtest directory found. Run 'sterling backtest' first.", err=True)
        return

    click.echo(f"Reading backtest from: {bt_path}\n")
    report = run_diagnostics(bt_path)

    stats = report["stats"]
    cash = report["cash_exposure"]
    scores = report["score_distribution"]
    macro = report["macro_overlay"]
    drag = report["cost_drag"]

    click.echo("=" * 58)
    click.echo("  STERLING BACKTEST DIAGNOSTIC REPORT")
    click.echo("=" * 58)
    click.echo(f"  Period:       {stats.get('start_date', '?')} to {stats.get('end_date', '?')}")
    click.echo(f"  Duration:     {stats.get('years', '?')} years")
    click.echo(f"  Total trades: {stats.get('total_trades', drag.get('total_trades', '?'))}")
    click.echo(f"  CAGR:         {stats.get('cagr_pct', '?'):+}%  (benchmark XIC: {stats.get('benchmark_xic_cagr_pct', '?'):+}%)")
    click.echo("")

    # ── 1. Cash exposure ──────────────────────────────────────────────────────
    click.echo("1. CASH EXPOSURE")
    click.echo("-" * 58)
    if cash:
        click.echo(f"   Trading days analyzed:        {cash['total_trading_days']}")
        click.echo(f"   Days fully in cash:           {cash['days_in_cash']}  ({cash['pct_in_cash']}%)")
        click.echo(f"   Days holding >= 1 position:   {cash['days_with_positions']}  ({cash['pct_with_positions']}%)")
        click.echo(f"   Peak simultaneous positions:  {cash['peak_simultaneous_positions']}")
        if cash["pct_in_cash"] > 50:
            click.echo(f"   [!!] Strategy was in CASH more than half the time.")
            click.echo(f"        During a bull run, this is the largest alpha drag.")
    else:
        click.echo("   [N/A] Could not determine date range from stats.")
    click.echo("")

    # ── 2. Score distribution ─────────────────────────────────────────────────
    click.echo("2. CONFIDENCE SCORE DISTRIBUTION AT ENTRY")
    click.echo("-" * 58)
    if scores:
        click.echo(f"   n={scores['count']}  min={scores['min']}  Q1={scores['q1']}  "
                   f"median={scores['median']}  Q3={scores['q3']}  max={scores['max']}  "
                   f"mean={scores['mean']}")
        click.echo("")
        click.echo("   Score  | Distribution (each # ~ 1 trade)")
        click.echo(scores["histogram"])
    else:
        click.echo("   [N/A] 'score' column not recorded in this backtest run.")
        click.echo("         The backtester has now been updated to save entry scores.")
        click.echo("         Re-run 'sterling backtest' to capture score data.")
        click.echo("")
        click.echo("   NOTE: This backtest used _quick_signal() — a technical-only scorer")
        click.echo("         (RSI + MACD + SMA cross). The full 5-axis confidence score")
        click.echo("         (technical + fundamental + sentiment + macro + insider) is")
        click.echo("         NOT used in backtesting. Entry threshold was 65.0/100.")
    click.echo("")

    # ── 3. Macro overlay demotions ────────────────────────────────────────────
    click.echo("3. MACRO OVERLAY DEMOTIONS (apply_macro_overlay)")
    click.echo("-" * 58)
    if macro["has_overlay_data"]:
        click.echo(f"   Trades where overlay fired:          {macro['overlay_applied_count']}")
        click.echo(f"   BUY signals demoted by overlay:      {macro['demoted_buy_to_lower']}")
    else:
        click.echo(f"   BUY signals demoted to HOLD/lower:   0  (structural zero)")
        click.echo("")
        click.echo("   REASON: The backtester calls _quick_signal() which is a")
        click.echo("           technical-only scorer. apply_macro_overlay() is never")
        click.echo("           invoked during walk-forward simulation.")
        click.echo("")
        click.echo("   IMPLICATION: The live analyst applies a -15pt downgrade when")
        click.echo("           VIX > 25 or TSX intraday < -2%. In a backtest covering")
        click.echo("           volatile periods (e.g. 2022 rate shock, 2024 vol spikes)")
        click.echo("           this overlay would suppress some entries — potentially")
        click.echo("           improving Sharpe by avoiding whipsaw trades, at the")
        click.echo("           cost of missed early-recovery entries.")
    click.echo("")

    # ── 4. Gross vs net return ────────────────────────────────────────────────
    click.echo("4. GROSS vs NET RETURN  (slippage & FX drag)")
    click.echo("-" * 58)
    click.echo(f"   Slippage rate:              {drag['round_trip_drag_pct']:.3f}% round-trip per trade")
    click.echo(f"   Total trades:               {drag['total_trades']}")
    click.echo(f"   Gross PnL (pre-slippage):   ${drag['gross_pnl_cad']:+.2f} CAD")
    click.echo(f"   Net PnL (post-slippage):    ${drag['net_pnl_cad']:+.2f} CAD")
    click.echo(f"   Total slippage cost:        ${drag['total_slippage_cost_cad']:+.2f} CAD  "
               f"({drag['slippage_as_pct_of_gross']:.2f}% of gross)")
    click.echo(f"   Avg slippage per trade:     ${drag['implied_slippage_per_trade']:+.2f} CAD")
    click.echo(f"   FX drag on trades:          ${drag['fx_drag_on_trades_cad']:.2f} CAD")
    click.echo("")
    click.echo(f"   FX NOTE: {drag['fx_drag_note']}")
    click.echo("")
    if abs(drag["total_slippage_cost_cad"]) > 0:
        total_return = stats.get("total_return_pct", 0)
        gross_return = total_return - (drag["total_slippage_cost_cad"] / stats.get("capital_cad", 1000) * 100)
        click.echo(f"   Gross total return (est.):  {gross_return:+.2f}%")
        click.echo(f"   Net total return:           {total_return:+.2f}%")

    click.echo("=" * 58)


# -- Outcomes -----------------------------------------------------------------

@cli.command()
@click.option("--summary", is_flag=True, help="Print aggregate outcome statistics")
@click.option("--by-confidence", is_flag=True, help="Break down outcomes by confidence bucket")
def outcomes(summary, by_confidence):
    """Evaluate paper-trade signals against actual price outcomes."""
    from sterling.outcomes import evaluate_signals, save_outcomes, compute_summary, compute_by_confidence

    click.echo("Fetching price history for paper-trade signals...\n")
    results = evaluate_signals()

    if not results:
        click.echo("No paper trades found. Run 'sterling analyze --paper' first.")
        return

    out_path = save_outcomes(results)

    if summary or by_confidence:
        stats = compute_summary(results)
        click.echo("=" * 60)
        click.echo("  PAPER TRADE OUTCOME SUMMARY")
        click.echo("=" * 60)
        click.echo(f"  Total signals (deduplicated):  {stats['total']}")
        click.echo(f"  Hit target:                    {stats['target']}  ({stats['target']/stats['total']*100:.0f}%)")
        click.echo(f"  Hit stop:                      {stats['stop']}  ({stats['stop']/stats['total']*100:.0f}%)")
        click.echo(f"  Expired (day 30):              {stats['expired']}  ({stats['expired']/stats['total']*100:.0f}%)")
        click.echo(f"  Still open:                    {stats['open']}  ({stats['open']/stats['total']*100:.0f}%)")
        if stats.get("error", 0):
            click.echo(f"  Errors:                        {stats['error']}")
        click.echo("-" * 60)
        if stats["resolved"]:
            click.echo(f"  Win rate (resolved):           {stats['win_rate_pct']:.1f}%")
            click.echo(f"  Avg winning trade:             {stats['avg_win_pct']:+.2f}%")
            click.echo(f"  Avg losing trade:              {stats['avg_loss_pct']:+.2f}%")
            click.echo(f"  Profit factor:                 {stats['profit_factor']}")
            click.echo(f"  Expectancy per trade:          {stats['expectancy_pct']:+.2f}%")
        else:
            click.echo("  No resolved trades yet.")
        click.echo("=" * 60)

        if by_confidence:
            buckets = compute_by_confidence(results)
            if buckets:
                click.echo(f"\n  {'CONFIDENCE':<12} {'SIGNALS':>8} {'WIN%':>7} {'AVG W':>8} {'AVG L':>8} {'PF':>6} {'EXPECT':>8}")
                click.echo(f"  {'-'*12} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")
                for b in buckets:
                    pf = f"{b['profit_factor']:.2f}" if b['profit_factor'] != float("inf") else "inf"
                    click.echo(
                        f"  {b['bucket']:<12} {b['total']:>8} {b['win_rate_pct']:>6.1f}% "
                        f"{b['avg_win_pct']:>+7.2f}% {b['avg_loss_pct']:>+7.2f}% {pf:>6} {b['expectancy_pct']:>+7.2f}%"
                    )
                click.echo("")
    else:
        _print_outcomes_table(results)

    click.echo(f"\n  Outcomes saved to: {out_path}")


def _print_outcomes_table(results: list):
    outcome_colors = {
        "target":  "\033[32m",
        "stop":    "\033[31m",
        "expired": "\033[33m",
        "open":    "\033[36m",
        "error":   "\033[35m",
    }
    reset = "\033[0m"

    click.echo(f"{'DATE':<12} {'TICKER':<10} {'CONF':>5} {'ENTRY':>8} {'STOP':>8} {'TARGET':>8} {'OUTCOME':<10} {'EXIT':>8} {'PNL%':>8}")
    click.echo("-" * 87)

    for r in results:
        col = outcome_colors.get(r["outcome"], "")
        entry = f"${float(r['entry_price']):.2f}" if r.get("entry_price") else "-"
        stop = f"${float(r['stop_loss']):.2f}" if r.get("stop_loss") else "-"
        target = f"${float(r['target']):.2f}" if r.get("target") else "-"
        exit_p = f"${float(r['exit_price']):.2f}" if r.get("exit_price") else "-"
        pnl = f"{float(r['realized_pnl_pct']):+.2f}%" if r.get("realized_pnl_pct") not in ("", None) else "-"
        click.echo(
            f"{r['timestamp'][:10]:<12} {r['ticker']:<10} {float(r.get('confidence',0)):>5.1f} "
            f"{entry:>8} {stop:>8} {target:>8} {col}{r['outcome']:<10}{reset} {exit_p:>8} {pnl:>8}"
        )


# -- Scheduler ----------------------------------------------------------------

@cli.command("run")
@click.option("--once", is_flag=True, help="Run one analysis pass and exit")
@click.option("--daemon", is_flag=True, help="Run as background process")
def run(once, daemon):
    """Start the scheduled analysis worker."""
    from sterling.scheduler import start
    start(daemon=daemon, run_once=once)


# -- Test runner --------------------------------------------------------------

@cli.command("test")
def run_tests():
    """Run the test suite with pytest."""
    import subprocess
    root = Path(__file__).parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(root / "tests"), "-v"],
        cwd=str(root),
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    cli()
