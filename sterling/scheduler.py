"""APScheduler-based job runner. Fires at TSX open, midday, near-close."""

import logging
import sys
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)

# TSX schedule (Eastern = UTC-4 during EDT, UTC-5 during EST)
# Using ET timezone; APScheduler handles DST transitions.
_JOBS = [
    {"hour": 9,  "minute": 30, "label": "open"},
    {"hour": 12, "minute": 30, "label": "midday"},
    {"hour": 15, "minute": 55, "label": "close"},
]


def _run_analysis():
    """Single analysis pass — called by each scheduled job."""
    from sterling import config, analyst, notifier, portfolio

    logger.info("Scheduled analysis starting...")
    cfg = config.validate_optional()
    if cfg["missing"]:
        logger.error(f"Missing env vars: {cfg['missing']} — skipping run")
        return

    port = portfolio.get_portfolio()
    tickers = list(port["holdings"].keys()) + port.get("watchlist", [])
    if not tickers:
        logger.info("No tickers in portfolio or watchlist — skipping analysis")
        return

    results = analyst.analyze_portfolio(tickers, config.FINNHUB_API_KEY)

    actionable = [r for r in results if r.get("recommendation") in ("BUY", "ACCUMULATE", "SELL", "TRIM")]
    sent_count = 0

    for signal in actionable:
        if config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
            sent = notifier.send_signal(signal, config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
            if sent:
                sent_count += 1

    logger.info(f"Analysis complete. {len(results)} tickers scored, {sent_count} alerts sent.")


def start(daemon: bool = False, run_once: bool = False):
    """
    Start the scheduler.
    - run_once: execute one analysis pass immediately, then exit.
    - daemon: use BackgroundScheduler (non-blocking).
    """
    _setup_logging()

    if run_once:
        logger.info("Running single-pass analysis...")
        _run_analysis()
        logger.info("Single-pass complete.")
        return

    scheduler_cls = BackgroundScheduler if daemon else BlockingScheduler
    scheduler = scheduler_cls(timezone="America/Toronto")

    for job in _JOBS:
        scheduler.add_job(
            _run_analysis,
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour=job["hour"],
                minute=job["minute"],
                timezone="America/Toronto",
            ),
            id=f"sterling_{job['label']}",
            name=f"Sterling {job['label']} scan",
            replace_existing=True,
            misfire_grace_time=300,
        )
        logger.info(f"Scheduled: {job['hour']:02d}:{job['minute']:02d} ET ({job['label']})")

    try:
        logger.info("Sterling scheduler started. Press Ctrl+C to stop.")
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("Scheduler stopped.")


def _setup_logging():
    log_dir = Path(__file__).parent.parent / "data"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "sterling.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.handlers.RotatingFileHandler(
                log_file, maxBytes=5 * 1024 * 1024, backupCount=3
            ),
        ],
    )

# Import at top-level only if needed
import logging.handlers
