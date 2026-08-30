"""CLI entry point:  python -m sterling.research <command>

Commands:
  verify     check the Sharadar API key is responding
  parquet    convert bulk CSVs to Parquet (one-time)
  edgar      download SEC companyfacts bulk + extract fundamentals to Parquet (one-time)
  features   build the survivorship-free feature+label matrix
  analyze    run walk-forward validation + net-of-cost portfolio sim
  all        features + analyze
"""

from __future__ import annotations

import argparse
import json
import logging
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="sterling.research")
    parser.add_argument("command",
                        choices=["verify", "parquet", "features", "analyze", "all",
                                 "book", "evaluate", "notify-test",
                                 "experiment", "experiment-final", "dashboard", "edgar",
                                 "grid-status", "merge-ledger"])
    parser.add_argument("--other", type=str, default=None,
                        help="merge-ledger: path to the other leaderboard CSV to union in")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--no-notify", action="store_true",
                        help="skip the Telegram notification for book/evaluate")
    parser.add_argument("-n", "--batch", type=int, default=20,
                        help="experiment: max candidates to evaluate this run")
    parser.add_argument("--budget-min", type=float, default=None,
                        help="experiment: stop starting new candidates after this many minutes")
    parser.add_argument("--fresh", action="store_true",
                        help="evaluate: pull current prices (yfinance) instead of the local bulk parquet")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(message)s")

    if args.command == "verify":
        from sterling.research import sharadar
        print(json.dumps(sharadar.verify(), indent=2))
    elif args.command == "parquet":
        from sterling.research import store
        store.ensure_parquet()
        print("Parquet ready.")
    elif args.command == "features":
        from sterling.research import pipeline
        print(json.dumps(pipeline.build_features(), indent=2, default=str))
    elif args.command == "analyze":
        from sterling.research import pipeline
        r = pipeline.analyze()
        print(json.dumps(r["walkforward"], indent=2, default=str))
    elif args.command == "all":
        from sterling.research import pipeline
        r = pipeline.run_all()
        print(json.dumps(r["walkforward"], indent=2, default=str))
    elif args.command == "book":
        from sterling.research import live, notify
        asof, book = live.generate_book()
        n = live.log_book(asof, book)
        print(f"Paper book for {asof}: {n} names, logged to {live.LEDGER.name}")
        print(book.head(15).to_string(index=False))
        if not args.no_notify:
            print("Telegram:", "sent" if notify.send(notify.format_book(asof, book)) else "skipped/failed")
    elif args.command == "evaluate":
        from sterling.research import live, notify
        res = live.evaluate(fresh=args.fresh)
        print(json.dumps(res, indent=2, default=str))
        if not args.no_notify and "error" not in res:
            print("Telegram:", "sent" if notify.send(notify.format_evaluation(res)) else "skipped/failed")
    elif args.command == "notify-test":
        from sterling.research import notify
        ok = notify.send("✅ <b>Sterling</b> — Telegram is wired up. You'll get the paper book here.")
        print("Telegram test:", "sent — check your phone" if ok else "failed (check .env creds)")
    elif args.command == "experiment":
        # one autonomous search cycle: evaluate a batch, update the leaderboard, ping progress
        from sterling.research import experiment, notify
        experiment.run_batch(n=args.batch, time_budget_min=args.budget_min)
        rep = experiment.final_report()
        print(json.dumps(rep, indent=2, default=str))
        if not args.no_notify:
            notify.send(notify.format_experiment(rep))
    elif args.command == "edgar":
        # one-time: download the SEC companyfacts bulk + extract our tags to Parquet
        from sterling.research import edgar
        edgar.download_companyfacts()
        ciks = set(edgar.ticker_cik_map().values())
        fp = edgar.convert_companyfacts(ciks=ciks or None)
        print(f"EDGAR facts ready: {fp} ({fp.stat().st_size / 1e6:.0f} MB, "
              f"{len(ciks)} universe CIKs)")
    elif args.command == "dashboard":
        from sterling.research import dashboard
        d = dashboard.build_data()
        print(f"dashboard: {d['trials']} trials -> docs/dashboard_data.json")
    elif args.command == "merge-ledger":
        from sterling.research import experiment
        res = experiment.merge_into_ledger(args.other)
        print(f"merged leaderboard: {res}")
    elif args.command == "grid-status":
        # print coverage; on first reaching 100%, ping Telegram once (sentinel-gated)
        from sterling.research import experiment, notify, config
        st = experiment.grid_status()
        print(json.dumps(st, indent=2))
        sentinel = config.DATA / ".grid_complete"
        if st["complete"] and not sentinel.exists():
            rep = experiment.final_report()
            if not args.no_notify:
                notify.send(
                    "🏁 <b>Sterling — GRID 100% COMPLETE</b>\n"
                    f"All {st['total']} strategy configs settled.\n\n"
                    + notify.format_experiment(rep))
            sentinel.write_text(str(st["total"]))
            print("grid complete — pinged + sentinel written")
    elif args.command == "experiment-final":
        # the Aug-28 wrap-up: full honest verdict to Telegram
        from sterling.research import experiment, notify
        rep = experiment.final_report()
        print(json.dumps(rep, indent=2, default=str))
        if not args.no_notify:
            notify.send("🏁 <b>Sterling — final results</b>\n" + notify.format_experiment(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
