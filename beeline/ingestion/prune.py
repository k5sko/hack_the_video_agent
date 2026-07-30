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
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
PAYLOAD = HERE / "graph_payload.json"
RAW = HERE / "graph_payload.raw.json"

MIN_CONFIDENCE = 2

# Cosine above which two concept names are treated as the same concept. Tuned by
# inspecting the merge list: lower starts fusing genuinely distinct ideas
# ("vectors" with "matrices"), higher leaves obvious synonyms apart.
SIMILARITY_THRESHOLD = 0.86

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


def _embedding_cache() -> Path:
    path = HERE.parent / "data" / "cache" / "prune_embeddings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def embed_names(names: List[str]) -> Dict[str, List[float]]:
    """Embed concept names, caching so re-running costs nothing."""
    cache_file = _embedding_cache()
    cache: Dict[str, List[float]] = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
        except json.JSONDecodeError:
            cache = {}

    missing = [n for n in names if n not in cache]
    if missing:
        sys.path.insert(0, str(HERE.parent / "graph"))
        from store import embed, load_env  # noqa: E402

        load_env()  # credentials live in the repo-root .env

        for i in range(0, len(missing), 256):
            chunk = missing[i : i + 256]
            for name, vector in zip(chunk, embed(chunk)):
                cache[name] = vector
        cache_file.write_text(json.dumps(cache))
    return {n: cache[n] for n in names if n in cache}


def _decision_cache() -> Path:
    path = HERE.parent / "data" / "cache" / "prune_merge_decisions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def adjudicate(pairs: List[tuple]) -> List[tuple]:
    """Decide which candidate pairs actually name the same concept.

    Cosine alone cannot make this call. Measured on this corpus, genuine
    synonyms score *below* genuine non-synonyms:

        0.552  'tokens' ~ 'tokenization'          <- same concept
        0.598  'embedding vectors' ~ 'word embeddings'  <- same concept
        0.613  'vectors' ~ 'matrices'             <- different concepts

    There is no cutoff that keeps the first two and rejects the third, so the
    plan's "merge above ~0.85 cosine" is not merely mistuned, it is unachievable.
    Lexical rules fail the same way: requiring a shared stem would fuse 'cost
    function' with 'activation function'.

    So embeddings do what they are good at -- proposing a short candidate list
    out of thousands of pairs -- and a language model does what it is good at:
    judging whether two names denote the same idea. Decisions are cached by pair,
    so this is a one-off cost.
    """
    if not pairs:
        return []

    cache_file = _decision_cache()
    decisions: Dict[str, bool] = {}
    if cache_file.exists():
        try:
            decisions = json.loads(cache_file.read_text())
        except json.JSONDecodeError:
            decisions = {}

    def key(a: str, b: str) -> str:
        return " || ".join(sorted((a, b)))

    unknown = [p for p in pairs if key(*p) not in decisions]
    if unknown:
        sys.path.insert(0, str(HERE.parent / "graph"))
        from store import load_env  # noqa: E402

        load_env()
        from openai import OpenAI

        client = OpenAI()
        for i in range(0, len(unknown), 60):
            batch = unknown[i : i + 60]
            listing = [{"i": j, "a": a, "b": b} for j, (a, b) in enumerate(batch)]
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "These pairs of terms come from transcripts of a "
                                "neural-network lecture series. For each pair, "
                                "decide whether the two terms refer to the SAME "
                                "underlying concept, such that a learner who "
                                "understood one would have nothing left to learn "
                                "from the other.\n\n"
                                "Same: 'tokens'/'tokenization', "
                                "'embedding vectors'/'word embeddings'.\n"
                                "Different: 'vectors'/'matrices', "
                                "'cost function'/'activation function' -- related, "
                                "but each teaches something the other does not.\n\n"
                                f"{json.dumps(listing, indent=1)}\n\n"
                                'Return JSON {"same": [i, ...]} listing only the '
                                "indices that are the same concept."
                            ),
                        }
                    ],
                    response_format={"type": "json_object"},
                    timeout=60,
                )
                same = set(json.loads(response.choices[0].message.content).get("same", []))
            except Exception as exc:
                print(f"  merge adjudication failed ({exc}); keeping apart", file=sys.stderr)
                same = set()
            for j, (a, b) in enumerate(batch):
                decisions[key(a, b)] = j in same
        cache_file.write_text(json.dumps(decisions, indent=1, sort_keys=True))

    return [p for p in pairs if decisions.get(key(*p))]


def merge_by_similarity(
    concepts: List[dict],
    explained: Counter,
    degree: Counter,
    threshold: float,
) -> Dict[str, str]:
    """Fold near-synonymous concept names onto one survivor.

    The extractor names the same idea several ways across chapters -- 'tokens'
    and 'tokenization', 'matrices' and 'matrix operations', 'embedding vectors'
    and 'word embeddings'. Stopword stripping cannot catch these, and left alone
    they are worse than clutter: the corpus teaches one spelling and not the
    other, so the untaught spelling surfaces as a *gap*. Beeline then reports
    that it never teaches "tokens" three clips after teaching tokenization.

    The survivor of each group is the name the corpus actually teaches most --
    most explaining clips, then most edges, then shortest. Merging toward what is
    taught is what turns these phantom gaps back into covered concepts.
    """
    names = [c["name"] for c in concepts]
    vectors = embed_names(names)
    usable = [n for n in names if n in vectors]

    import math

    norms = {}
    for name in usable:
        v = vectors[name]
        norms[name] = math.sqrt(sum(x * x for x in v)) or 1.0

    def strength(name: str) -> tuple:
        return (explained[name], degree[name], -len(name), name)

    candidates: List[tuple] = []
    for i, a in enumerate(usable):
        va, na = vectors[a], norms[a]
        for b in usable[i + 1 :]:
            vb = vectors[b]
            dot = sum(x * y for x, y in zip(va, vb))
            if dot / (na * norms[b]) >= threshold:
                candidates.append((a, b))

    same = adjudicate(candidates)

    # Deliberately NOT a union-find. Transitive merging is catastrophic here:
    # 'A is B' and 'B is C' and 'C is D' chained together once collapsed
    # 'neural network', 'linear algebra', 'matrix multiplication', 'vectors' and
    # 'embedding' into a single node called 'gradient descent'. Similarity is not
    # transitive, and every pair in that chain was individually plausible.
    #
    # Instead each name may only fold into a survivor it was *directly* judged
    # identical to. Ambiguous middles stay put, which is the safe failure: an
    # unmerged duplicate is untidy, a wrongly merged concept is a wrong path.
    partners: Dict[str, set] = defaultdict(set)
    for a, b in same:
        partners[a].add(b)
        partners[b].add(a)

    # A name folds into a survivor only when it was directly judged identical to
    # it AND shares that survivor's whole neighbourhood -- effectively requiring
    # the little cluster to agree, not just one pair.
    #
    # A looser rule ("merge into your strongest partner") was tried and reverted.
    # It merged a few more synonyms but took REQUIRES from 181 edges to 230,
    # because summing confidence across newly merged names lifted junk like
    # 'derivative requires neural network' over the floor -- and that promptly
    # produced ordering violations. Fewer, safer merges make a better graph.
    canonical: Dict[str, str] = {name: name for name in names}
    for name in sorted(partners, key=strength, reverse=True):
        if canonical[name] != name:  # already absorbed; cannot also be a survivor
            continue
        for other in sorted(partners[name], key=strength, reverse=True):
            if other == name or canonical[other] != other:
                continue
            if strength(other) < strength(name) and not partners[other] - {name} - partners[name]:
                canonical[other] = name
    return canonical


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


def prune(
    payload: dict,
    min_confidence: int = MIN_CONFIDENCE,
    use_embeddings: bool = True,
    threshold: float = SIMILARITY_THRESHOLD,
) -> tuple[dict, dict]:
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

    def remap() -> None:
        for edge in p["requires"]:
            edge["from"] = canonical.get(edge["from"], edge["from"])
            edge["to"] = canonical.get(edge["to"], edge["to"])
        for edge in p["explains"]:
            edge["concept"] = canonical.get(edge["concept"], edge["concept"])

    remap()

    # 1b. merge near-synonyms by embedding similarity. This is the step that
    #     stops the corpus reporting "tokens" as a concept it never teaches
    #     three clips after it teaches tokenization.
    merged_by_similarity = 0
    if use_embeddings:
        explained_now = Counter(e["concept"] for e in p["explains"])
        degree_now: Counter = Counter()
        for e in p["requires"]:
            degree_now[e["from"]] += 1
            degree_now[e["to"]] += 1

        survivors_now = sorted({canonical[c["name"]] for c in p["concepts"]})
        similar = merge_by_similarity(
            [{"name": n} for n in survivors_now],
            explained_now,
            degree_now,
            threshold,
        )
        merged_by_similarity = sum(1 for k, v in similar.items() if k != v)
        for name, target in similar.items():
            if name != target:
                aliases[target].update(aliases.pop(name, set()) | {name})
        for key in list(canonical):
            canonical[key] = similar.get(canonical[key], canonical[key])
        remap()

    # 2. Collapse duplicate edges. Confidence counts how many chapters asserted
    #    a prerequisite, so once two names become one concept their assertions
    #    are assertions about the same edge and add up. Summing before the floor
    #    is what lets genuine edges, split across synonyms, survive it.
    summed: Dict[tuple, dict] = {}
    for e in p["requires"]:
        if e["from"] == e["to"]:  # self-edges appear once names merge
            continue
        key = (e["from"], e["to"])
        if key in summed:
            summed[key]["confidence"] += e["confidence"]
        else:
            summed[key] = dict(e)
    p["requires"] = [
        e
        for e in sorted(summed.values(), key=lambda e: (e["from"], e["to"]))
        if e["confidence"] >= min_confidence
    ]

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
        "merged_by_similarity": merged_by_similarity,
        "dropped_concepts": sorted(dropped),
        "min_confidence": min_confidence,
    }
    return p, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min-confidence", type=int, default=MIN_CONFIDENCE)
    ap.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD)
    ap.add_argument(
        "--no-embeddings",
        action="store_true",
        help="skip similarity merging (no OpenAI calls)",
    )
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

    pruned, stats = prune(
        payload,
        args.min_confidence,
        use_embeddings=not args.no_embeddings,
        threshold=args.threshold,
    )

    b, a = stats["before"], stats["after"]
    print(f"min_confidence = {stats['min_confidence']}")
    for k in ("concepts", "requires", "explains"):
        print(f"  {k:9} {b[k]:>5} -> {a[k]:>5}")
    print(f"  merged variants: {stats['merged']} "
          f"({stats['merged_by_similarity']} by similarity)")
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
