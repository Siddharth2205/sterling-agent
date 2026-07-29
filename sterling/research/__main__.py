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
                        choices=["verify", "parquet", "features", "analyze", "all"])
    parser.add_argument("-v", "--verbose", action="store_true")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
