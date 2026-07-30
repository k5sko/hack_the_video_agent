"""Beeline path CLI.

    python path.py "attention" --known "neural network,linear algebra"
    python path.py "how does a network learn" --store neo4j

Prints exactly the JSON that ``build_path`` returns, validated through
``shared.types.PathResult``. P4 imports ``build_path``; this file is only the
hand-crank for it.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from engine import build_path
from store import get_store


def parse_known(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build a Beeline learning path.")
    ap.add_argument("query", help="free-text thing you want to understand")
    ap.add_argument(
        "--known",
        default="",
        help="comma-separated concepts you already know",
    )
    ap.add_argument(
        "--store",
        choices=("memory", "neo4j"),
        default="memory",
        help="memory reads the JSON payload; neo4j reads the loaded Aura graph",
    )
    ap.add_argument("--payload", default=None, help="payload path for --store memory")
    ap.add_argument("--indent", type=int, default=2)
    args = ap.parse_args(argv)

    store = get_store(args.store, args.payload)
    try:
        result = build_path(args.query, parse_known(args.known), store)
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()

    print(result.model_dump_json(indent=args.indent))
    return 0


if __name__ == "__main__":
    sys.exit(main())
