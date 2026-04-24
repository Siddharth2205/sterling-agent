"""Sterling CLI -- one entry point for all commands."""

import io
import json
import logging
import sys
from pathlib import Path

import click

# Ensure UTF-8 output on Windows consoles (cp1252 by default)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("sterling")


@click.group()
@click.version_option(version="0.1.0", prog_name="sterling")
def cli():
    """Sterling -- Canadian portfolio analysis agent."""


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
def analyze(ticker, notify, json_output):
    """Run multi-factor analysis on portfolio and watchlist."""
    from sterling import config, analyst, portfolio, notifier

    cfg = config.validate_optional()
    if cfg["missing"]:
        click.echo(f"Warning: missing env vars {cfg['missing']} -- some signals will be degraded", err=True)

    if ticker:
        tickers = list(ticker)
    else:
        port = portfolio.get_portfolio()
        tickers = list(port["holdings"].keys()) + port.get("watchlist", [])

    if not tickers:
        click.echo("No tickers. Add holdings with 'sterling add' or watchlist with 'sterling watch'.")
        return

    click.echo(f"Analyzing {len(tickers)} tickers: {', '.join(tickers)}\n")

    results = analyst.analyze_portfolio(tickers, config.FINNHUB_API_KEY or "")

    if json_output:
        click.echo(json.dumps(results, indent=2))
        return

    _print_analysis_table(results)

    if notify and config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
        actionable = [r for r in results if r.get("recommendation") in ("BUY", "ACCUMULATE", "SELL", "TRIM")]
        for signal in actionable:
            notifier.send_signal(signal, config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
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


# -- Backtest -----------------------------------------------------------------

@cli.command()
@click.option("--years", default=3, help="Years of history to backtest", show_default=True)
@click.option("--capital", default=1000.0, help="Starting capital in CAD", show_default=True)
@click.option("--threshold", default=65.0, help="Signal score threshold to enter", show_default=True)
@click.option("--hold-days", default=20, help="Max hold period in trading days", show_default=True)
def backtest(years, capital, threshold, hold_days):
    """Walk-forward backtest on TSX 60 constituents."""
    from sterling.backtester import run_backtest, save_results

    click.echo(f"Running {years}-year backtest on ${capital:.0f} CAD capital...")
    click.echo("This may take 2-5 minutes while downloading price history.\n")

    result = run_backtest(
        years=years,
        capital_cad=capital,
        hold_days=hold_days,
        signal_threshold=threshold,
    )

    out_dir = save_results(result)

    stats = result.stats
    click.echo("\n" + "=" * 52)
    click.echo("  STERLING BACKTEST RESULTS")
    click.echo("=" * 52)
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
    click.echo("-" * 52)
    click.echo(f"  XIC.TO CAGR:      {stats.get('benchmark_xic_cagr_pct', 0):+.2f}% (after 1.5% FX drag)")
    beats = stats.get("beats_benchmark", False)
    verdict = "[OK] Strategy beats benchmark" if beats else "[!!] Strategy underperforms benchmark"
    click.echo(f"  Verdict:          {verdict}")
    click.echo("=" * 52)
    click.echo(f"\n  Charts and trade log saved to: {out_dir}")


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
