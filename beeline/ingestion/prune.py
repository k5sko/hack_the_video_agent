"""Post-process graph_payload.json into something a path engine can reason over.

The extractor is deliberately permissive: it proposes every concept a chapter
mentions and every prerequisite it implies. That is the right call at extraction
time -- recall is cheap to get and impossible to recover later -- but it means the
raw payload carries ~1600 REQUIRES edges, 93% of which were asserted by exactly
one chapter. Those single-assertion edges are where "attention requires the mnist
dataset" comes from, and the prerequisite closure inherits every one of them.

This step spends that recall back:

  1. merge modifier variants ("basic calculus" / "calculus basics" -> "calculus")
  2. drop REQUIRES edges below MIN_CONFIDENCE (default 2: at least two chapters
     independently asserted it)
  3. drop orphan concepts -- barely explained and required by nothing -- which
     otherwise sit in the graph as string-match decoys for the resolver

Measured on the 7-video corpus, dropping to confidence >= 2 takes the "attention"
path from 23 clips / 38.1 min to 7 clips / 12.6 min and leaves 104 edges. Going
to >= 3 leaves 26 edges and the paths get too thin to teach anything.

Reads graph_payload.json, writes graph_payload.json, and keeps the untouched
extractor output beside it as graph_payload.raw.json so this is always redoable.

    python -m beeline.ingestion.prune            # default thresholds
    python -m beeline.ingestion.prune --min-confidence 3 --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
PAYLOAD = HERE / "graph_payload.json"
RAW = HERE / "graph_payload.raw.json"

MIN_CONFIDENCE = 2

# Words that qualify a concept without changing which concept it is.
MODIFIERS = {
    "basic",
    "basics",
    "simple",
    "general",
    "overall",
    "fundamental",
    "fundamentals",
    "concept",
    "concepts",
    "intro",
    "introduction",
}

# Extractor artifacts that name a piece of the video, not a piece of knowledge.
NON_CONCEPT = re.compile(
    r"\b(chapter|video|lesson|series|recap|outline|preview|episode|part \d+)\b",
    re.I,
)


def core(name: str) -> str:
    """Strip qualifiers so surface variants collapse onto one key."""
    tokens = [t for t in re.split(r"\W+", name.lower()) if t and t not in MODIFIERS]
    return " ".join(tokens) or name.lower()


def build_canonical_map(concepts: List[dict]) -> Dict[str, str]:
    """Map every concept name onto the survivor for its core form.

    The survivor is the shortest surface form, which is reliably the least
    qualified one ("calculus" beats "calculus basics").
    """
    by_core: Dict[str, List[str]] = defaultdict(list)
    for c in concepts:
        by_core[core(c["name"])].append(c["name"])

    canonical: Dict[str, str] = {}
    for names in by_core.values():
        survivor = min(names, key=lambda n: (len(n), n))
        for n in names:
            canonical[n] = survivor
    return canonical


def prune(payload: dict, min_confidence: int = MIN_CONFIDENCE) -> tuple[dict, dict]:
    """Return (pruned payload, stats). Does not mutate the input."""
    p = json.loads(json.dumps(payload))
    before = {k: len(p.get(k, [])) for k in ("concepts", "requires", "explains")}

    # 1. merge surface variants, folding absorbed names in as aliases so the
    #    resolver can still find a concept by the wording the lecture used.
    canonical = build_canonical_map(p["concepts"])
    aliases: Dict[str, set] = defaultdict(set)
    for c in p["concepts"]:
        target = canonical[c["name"]]
        aliases[target].update(c.get("aliases") or [])
        if c["name"] != target:
            aliases[target].add(c["name"])

    for e in p["requires"]:
        e["from"] = canonical.get(e["from"], e["from"])
        e["to"] = canonical.get(e["to"], e["to"])
    for e in p["explains"]:
        e["concept"] = canonical.get(e["concept"], e["concept"])

    # 2. confidence floor. Self-edges can appear once names merge.
    p["requires"] = [
        e
        for e in p["requires"]
        if e["confidence"] >= min_confidence and e["from"] != e["to"]
    ]

    # Merging can produce duplicate edges; keep the strongest of each pair.
    strongest: Dict[tuple, dict] = {}
    for e in p["requires"]:
        key = (e["from"], e["to"])
        if key not in strongest or e["confidence"] > strongest[key]["confidence"]:
            strongest[key] = e
    p["requires"] = sorted(strongest.values(), key=lambda e: (e["from"], e["to"]))

    # Same for EXPLAINS: one clip may now explain a merged concept twice.
    best: Dict[tuple, dict] = {}
    for e in p["explains"]:
        key = (e["clip_id"], e["concept"])
        if key not in best or e["score"] > best[key]["score"]:
            best[key] = e
    p["explains"] = sorted(best.values(), key=lambda e: (e["clip_id"], e["concept"]))

    # 3. drop orphans and non-concepts. An orphan is explained by at most one
    #    clip AND required by nothing -- it can never be a prerequisite, so its
    #    only effect is to sit in the index as a decoy for the resolver. This is
    #    what makes "how does a network learn" match the dead-end leaf "learning"
    #    instead of "gradient descent".
    explained = Counter(e["concept"] for e in p["explains"])
    required_by = Counter(e["to"] for e in p["requires"])
    requires_out = Counter(e["from"] for e in p["requires"])

    survivors = []
    dropped: List[str] = []
    for c in {canonical[c["name"]]: c for c in p["concepts"]}.values():
        name = canonical[c["name"]]
        is_junk = bool(NON_CONCEPT.search(name))
        is_orphan = (
            explained[name] <= 1 and required_by[name] == 0 and requires_out[name] == 0
        )
        if is_junk or is_orphan:
            dropped.append(name)
            continue
        c = dict(c, name=name, aliases=sorted(aliases[name] - {name}))
        survivors.append(c)

    p["concepts"] = sorted(survivors, key=lambda c: c["name"])
    kept = {c["name"] for c in p["concepts"]}
    p["requires"] = [e for e in p["requires"] if e["from"] in kept and e["to"] in kept]
    p["explains"] = [e for e in p["explains"] if e["concept"] in kept]

    stats = {
        "before": before,
        "after": {k: len(p[k]) for k in ("concepts", "requires", "explains")},
        "merged": sum(1 for k, v in canonical.items() if k != v),
        "dropped_concepts": sorted(dropped),
        "min_confidence": min_confidence,
    }
    return p, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min-confidence", type=int, default=MIN_CONFIDENCE)
    ap.add_argument("--payload", default=str(PAYLOAD))
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change without writing",
    )
    args = ap.parse_args()

    path = Path(args.payload)
    raw = path.with_suffix(".raw.json")

    # Always prune from the extractor's original output, never from an
    # already-pruned file, so re-running with a different threshold is safe.
    source = raw if raw.exists() else path
    payload = json.loads(source.read_text())

    pruned, stats = prune(payload, args.min_confidence)

    b, a = stats["before"], stats["after"]
    print(f"min_confidence = {stats['min_confidence']}")
    for k in ("concepts", "requires", "explains"):
        print(f"  {k:9} {b[k]:>5} -> {a[k]:>5}")
    print(f"  merged variants: {stats['merged']}")
    print(f"  dropped concepts: {len(stats['dropped_concepts'])}")

    if args.dry_run:
        print("\n(dry run, nothing written)")
        return 0

    if not raw.exists():
        shutil.copy2(path, raw)
        print(f"\nkept extractor output at {raw.name}")
    path.write_text(json.dumps(pruned, indent=2) + "\n")
    print(f"wrote {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
