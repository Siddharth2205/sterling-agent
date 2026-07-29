"""CLI entry point:  python -m sterling.research <command>

Commands:
  verify     check the Sharadar API key is responding
  parquet    convert bulk CSVs to Parquet (one-time)
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
                                 "book", "evaluate", "notify-test"])
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--no-notify", action="store_true",
                        help="skip the Telegram notification for book/evaluate")
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
        res = live.evaluate()
        print(json.dumps(res, indent=2, default=str))
        if not args.no_notify and "error" not in res:
            print("Telegram:", "sent" if notify.send(notify.format_evaluation(res)) else "skipped/failed")
    elif args.command == "notify-test":
        from sterling.research import notify
        ok = notify.send("✅ <b>Sterling</b> — Telegram is wired up. You'll get the paper book here.")
        print("Telegram test:", "sent — check your phone" if ok else "failed (check .env creds)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
