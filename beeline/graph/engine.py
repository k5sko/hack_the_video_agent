"""Beeline path engine — pure Python, no Neo4j, no OpenAI, no network.

Everything in here operates on plain in-memory structures handed over by a store
(see ``store.py``). That separation is deliberate: the algorithm is the part that
has to be provably correct, so it must be unit-testable with zero infrastructure.

Pipeline (see graph/README.md):

    resolve -> closure -> prune known -> greedy weighted set cover
            -> topological ordering -> PathResult

The only third-party imports are ``networkx`` (cycle detection, a pure graph
library) and ``shared.types`` (the Pydantic response contract, which we import
rather than re-declare).
"""

from __future__ import annotations

import heapq
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Set, Tuple

import networkx as nx

# beeline/ on sys.path so that `shared.types` resolves regardless of cwd.
_BEELINE_ROOT = Path(__file__).resolve().parents[1]
if str(_BEELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_BEELINE_ROOT))

from shared.types import ClipSegment, PathResult  # noqa: E402

MAX_DEPTH = 5

__all__ = [
    "Video",
    "Clip",
    "Concept",
    "Explains",
    "RequiresEdge",
    "GraphStore",
    "normalize",
    "break_cycles",
    "requires_adjacency",
    "closure",
    "prune_known",
    "select_clips",
    "topo_order",
    "order_clips",
    "build_path",
    "MAX_DEPTH",
]


# --------------------------------------------------------------------------- #
# Plain data                                                                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Video:
    id: str
    title: str
    youtube_url: str
    duration: float


@dataclass(frozen=True)
class Clip:
    id: str
    video_id: str
    start: float
    end: float
    summary: str = ""

    @property
    def duration(self) -> float:
        return max(float(self.end) - float(self.start), 1.0)


@dataclass(frozen=True)
class Concept:
    name: str
    aliases: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Explains:
    clip_id: str
    concept: str
    score: float


@dataclass(frozen=True)
class RequiresEdge:
    """``source`` REQUIRES ``target``: you need ``target`` before ``source``."""

    source: str
    target: str
    confidence: float = 1.0


@dataclass
class RemovedEdge:
    source: str
    target: str
    confidence: float
    cycle: Tuple[str, ...] = field(default_factory=tuple)

    def __str__(self) -> str:  # pragma: no cover - logging sugar
        loop = " -> ".join(self.cycle) if self.cycle else "?"
        return (
            f"REQUIRES {self.source!r} -> {self.target!r} "
            f"(confidence={self.confidence}) broke cycle [{loop}]"
        )


class GraphStore(Protocol):
    """Structural contract the engine needs. ``store.py`` supplies two of these."""

    def resolve(self, query: str) -> List[str]: ...
    def requires_edges(self) -> List[RequiresEdge]: ...
    def clips(self) -> List[Clip]: ...
    def explains(self) -> List[Explains]: ...
    def concepts(self) -> List[Concept]: ...
    def videos(self) -> List[Video]: ...


# --------------------------------------------------------------------------- #
# Normalisation                                                                 #
# --------------------------------------------------------------------------- #


def normalize(name: str) -> str:
    """Concept keys are compared case- and whitespace-insensitively."""
    return " ".join(str(name).strip().lower().split())


# --------------------------------------------------------------------------- #
# 0. Cycle breaking                                                             #
# --------------------------------------------------------------------------- #


def break_cycles(
    edges: Iterable[RequiresEdge],
) -> Tuple[List[RequiresEdge], List[RemovedEdge]]:
    """Make the REQUIRES graph a DAG by deleting the weakest edge of each cycle.

    LLM-extracted prerequisite edges are routinely contradictory ("A requires B"
    *and* "B requires A"). Both the depth-capped closure and the topological sort
    are meaningless on a cyclic graph, so we repair it before anything else runs.

    Deterministic: within a cycle we drop the lowest-confidence edge, ties broken
    alphabetically by (source, target). Returns (surviving_edges, removed).
    """
    edges = list(edges)
    conf: Dict[Tuple[str, str], float] = {}
    for e in edges:
        key = (e.source, e.target)
        # keep the strongest assertion if P2 emitted the same edge twice
        conf[key] = max(conf.get(key, float("-inf")), float(e.confidence))

    graph = nx.DiGraph()
    for source, target in sorted(conf):
        graph.add_edge(source, target)

    removed: List[RemovedEdge] = []
    while True:
        try:
            cycle = nx.find_cycle(graph, orientation="original")
        except nx.NetworkXNoCycle:
            break
        cycle_edges = [(u, v) for u, v, *_ in cycle]
        worst = min(cycle_edges, key=lambda e: (conf[e], e[0], e[1]))
        graph.remove_edge(*worst)
        removed.append(
            RemovedEdge(
                source=worst[0],
                target=worst[1],
                confidence=conf[worst],
                cycle=tuple(u for u, _ in cycle_edges) + (cycle_edges[0][0],),
            )
        )

    survivors = [
        RequiresEdge(source=u, target=v, confidence=conf[(u, v)])
        for u, v in sorted(graph.edges())
    ]
    return survivors, removed


def requires_adjacency(edges: Iterable[RequiresEdge]) -> Dict[str, List[str]]:
    """source -> sorted list of its direct prerequisites."""
    adj: Dict[str, Set[str]] = {}
    for e in edges:
        adj.setdefault(e.source, set()).add(e.target)
        adj.setdefault(e.target, set())
    return {k: sorted(v) for k, v in adj.items()}


# --------------------------------------------------------------------------- #
# 1./2. Closure and prune-known (one BFS, known set is the only difference)     #
# --------------------------------------------------------------------------- #


def prune_known(
    targets: Sequence[str],
    adjacency: Dict[str, List[str]],
    known: Iterable[str] = (),
    max_depth: int = MAX_DEPTH,
) -> Set[str]:
    """BFS down REQUIRES from ``targets``, stopping at known nodes.

    Known nodes are dropped *and never expanded*, so anything reachable only
    through them disappears too. A node still reachable by an unknown route
    survives — that is the whole point: ticking one box collapses a subtree
    without amputating shared prerequisites.

    Targets are always kept even if the caller lists them as known; the target is
    what was asked for, and returning an empty path for it is not useful.
    """
    known_set = {normalize(k) for k in known}
    targets = [normalize(t) for t in targets]

    result: Set[str] = set(targets)
    seen: Set[str] = set(targets)
    # a target you already know is kept (you asked for it) but not expanded
    frontier: List[str] = sorted(t for t in set(targets) if t not in known_set)

    depth = 0
    while frontier and depth < max_depth:
        depth += 1
        nxt: List[str] = []
        for node in frontier:
            for prereq in adjacency.get(node, []):
                if prereq in seen:
                    continue
                seen.add(prereq)
                if prereq in known_set:
                    continue  # stop here: not included, not expanded
                result.add(prereq)
                nxt.append(prereq)
        frontier = sorted(nxt)
    return result


def closure(
    targets: Sequence[str],
    adjacency: Dict[str, List[str]],
    max_depth: int = MAX_DEPTH,
) -> Set[str]:
    """Full depth-capped prerequisite closure — prune_known with nothing known."""
    return prune_known(targets, adjacency, known=(), max_depth=max_depth)


# --------------------------------------------------------------------------- #
# 3. Greedy weighted set cover                                                  #
# --------------------------------------------------------------------------- #


# A clip whose EXPLAINS score is below this does not *teach* the concept, it
# merely mentions it. Letting a mention count as coverage is how a playlist ends
# up made of title cards.
MIN_COVERAGE_SCORE = 0.75

# Even above the floor, only clips close to the best available explanation of a
# concept may cover it. Without this, a 0.80 aside beats the 0.95 chapter that
# actually explains the thing, purely by being shorter.
SCORE_TOLERANCE = 0.10


def qualified_coverage(
    needed: Iterable[str],
    explains: Iterable[Explains],
    min_score: float = MIN_COVERAGE_SCORE,
    tolerance: float = SCORE_TOLERANCE,
) -> Dict[str, Dict[str, float]]:
    """{clip_id: {concept: score}} restricted to clips that genuinely teach.

    Two filters, and the second matters more than the first: an absolute floor
    removes passing mentions, then a *relative* floor keeps only clips within
    ``tolerance`` of the best explanation that exists for each concept. The
    relative one is what stops a weak-but-short clip from winning a
    time-normalised contest against the chapter that does the real work.
    """
    needed_set = {normalize(c) for c in needed}

    best_for: Dict[str, float] = {}
    rows: List[Tuple[str, str, float]] = []
    for e in explains:
        concept = normalize(e.concept)
        if concept not in needed_set:
            continue
        score = float(e.score)
        rows.append((e.clip_id, concept, score))
        if score > best_for.get(concept, 0.0):
            best_for[concept] = score

    by_clip: Dict[str, Dict[str, float]] = {}
    for clip_id, concept, score in rows:
        cutoff = max(min_score, best_for[concept] - tolerance)
        if score >= cutoff:
            by_clip.setdefault(clip_id, {})[concept] = score
    return by_clip


def select_clips(
    needed: Iterable[str],
    clips: Sequence[Clip],
    explains: Iterable[Explains],
    targets: Iterable[str] = (),
    min_score: float = MIN_COVERAGE_SCORE,
    tolerance: float = SCORE_TOLERANCE,
) -> Tuple[List[str], Dict[str, List[str]], List[str]]:
    """Cover every needed concept with clips that actually teach it.

    Prerequisites and targets want opposite things. For a prerequisite you want
    the cheapest clip that unblocks you -- it is scaffolding, not the point. For
    the concept the learner actually asked about, the shortest qualifying clip is
    the worst possible answer; they came here to understand that one thing, so it
    gets the most substantial explanation available.

    So targets are covered first, by depth (highest score, then longest), and the
    remainder is a time-normalised greedy set cover over qualified clips only.

    Returns (selected_clip_ids, {clip_id: concepts it newly covered}, gaps).
    ``gaps`` are needed concepts no clip teaches -- surfaced, never dropped.
    """
    needed_set = {normalize(c) for c in needed}
    target_set = {normalize(t) for t in targets} & needed_set
    by_clip = qualified_coverage(needed_set, explains, min_score, tolerance)

    durations = {c.id: c.duration for c in clips}
    remaining = set(needed_set)
    selected: List[str] = []
    covers: Dict[str, List[str]] = {}

    def take(clip_id: str) -> None:
        newly = sorted(c for c in by_clip[clip_id] if c in remaining)
        covers[clip_id] = newly
        selected.append(clip_id)
        remaining.difference_update(newly)

    # Phase 1 -- depth for what was asked about.
    for target in sorted(target_set):
        if target not in remaining:
            continue
        candidates = [
            cid
            for cid, hits in by_clip.items()
            if target in hits and cid in durations and cid not in covers
        ]
        if not candidates:
            continue
        # best explanation; among equals, the one that spends the most time on it
        take(max(candidates, key=lambda cid: (by_clip[cid][target], durations[cid], cid)))

    # Phase 2 -- efficiency for the scaffolding.
    while remaining:
        best_id: Optional[str] = None
        best_key: Optional[Tuple[float, float, str]] = None
        for clip_id in sorted(by_clip):
            if clip_id in covers or clip_id not in durations:
                continue
            hits = {c: s for c, s in by_clip[clip_id].items() if c in remaining}
            if not hits:
                continue
            gain = sum(hits.values())
            ratio = gain / durations[clip_id]
            # max ratio; ties -> larger gain; ties -> lexicographic clip id
            key = (-ratio, -gain, clip_id)
            if best_key is None or key < best_key:
                best_key, best_id = key, clip_id
        if best_id is None:
            break
        take(best_id)

    return selected, covers, sorted(remaining)


# --------------------------------------------------------------------------- #
# 4. Ordering                                                                   #
# --------------------------------------------------------------------------- #


def topo_order(nodes: Iterable[str], adjacency: Dict[str, List[str]]) -> List[str]:
    """Kahn over the induced subgraph, prerequisites first.

    REQUIRES points dependent -> prerequisite, so we walk the reversed edges.
    A min-heap on the concept name makes the order deterministic.
    """
    nodes = sorted({normalize(n) for n in nodes})
    node_set = set(nodes)

    # edge prereq -> dependent, restricted to the induced subgraph
    out: Dict[str, List[str]] = {n: [] for n in nodes}
    indeg: Dict[str, int] = {n: 0 for n in nodes}
    for dependent in nodes:
        for prereq in adjacency.get(dependent, []):
            if prereq in node_set:
                out[prereq].append(dependent)
                indeg[dependent] += 1

    heap = [n for n in nodes if indeg[n] == 0]
    heapq.heapify(heap)
    order: List[str] = []
    while heap:
        node = heapq.heappop(heap)
        order.append(node)
        for dependent in sorted(out[node]):
            indeg[dependent] -= 1
            if indeg[dependent] == 0:
                heapq.heappush(heap, dependent)

    if len(order) < len(nodes):  # pragma: no cover - defensive, graph is a DAG
        order.extend(sorted(node_set - set(order)))
    return order


def order_clips(
    selected: Sequence[str],
    covers: Dict[str, List[str]],
    order: Sequence[str],
    clips_by_id: Dict[str, Clip],
    adjacency: Optional[Dict[str, List[str]]] = None,
    targets: Iterable[str] = (),
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Topologically sort the clips themselves, prerequisites first.

    The original plan ordered each clip by the earliest topo position among the
    concepts it covers. That silently breaks whenever a clip teaches more than
    one thing: a chapter covering both 'word embeddings' (early) and
    'tokenization' (late) gets placed by the early one and drags the late one in
    ahead of its own prerequisites. Using the *latest* position instead just
    moves the violation somewhere else.

    Both fail because ordering concepts is the wrong problem -- we are ordering
    clips, and a clip is an indivisible bundle of concepts. Once you build the
    clip-level graph the real structure shows up, including cycles that no
    concept-position sort can represent:

        clip A teaches {word embeddings, tokenization}
        clip B teaches {high-dimensional vectors}
        tokenization requires high-dimensional vectors  -> A after B
        high-dimensional vectors requires word embeddings -> B after A

    That cycle is genuine, not a data error: the corpus really does teach those
    concepts in bundles that interlock. So we Kahn over the clip graph and, when
    it stalls, break the cycle at the most defensible point -- the clip whose own
    concepts are furthest upstream -- rather than pretending the conflict is not
    there. Deterministic throughout: ties break by (video, start).
    """
    adjacency = adjacency or {}
    pos = {name: i for i, name in enumerate(order)}
    selected = list(selected)

    def depth(clip_id: str) -> int:
        return max((pos[c] for c in covers.get(clip_id, []) if c in pos), default=len(pos))

    def tiebreak(clip_id: str) -> Tuple[int, str, float, str]:
        clip = clips_by_id[clip_id]
        return (depth(clip_id), clip.video_id, float(clip.start), clip_id)

    # Which clip teaches each concept (first selected wins; covers are disjoint).
    teacher: Dict[str, str] = {}
    for clip_id in selected:
        for concept in covers.get(clip_id, []):
            teacher.setdefault(concept, clip_id)

    # Edge prereq_clip -> dependent_clip.
    out: Dict[str, Set[str]] = {c: set() for c in selected}
    indeg: Dict[str, int] = {c: 0 for c in selected}
    for clip_id in selected:
        for concept in covers.get(clip_id, []):
            for prereq in adjacency.get(concept, ()):
                source = teacher.get(prereq)
                if source is None or source == clip_id:
                    continue
                if clip_id not in out[source]:
                    out[source].add(clip_id)
                    indeg[clip_id] += 1

    def corpus_position(clip_id: str) -> Tuple[str, float]:
        clip = clips_by_id[clip_id]
        return (clip.video_id, float(clip.start))

    ordered: List[str] = []
    conflicts: List[Tuple[str, str]] = []
    pending = set(selected)
    while pending:
        ready = [c for c in pending if indeg[c] == 0]
        if not ready:
            # A genuine cycle: some ordering constraint must be sacrificed. Break
            # it in corpus order -- release whichever clip the lecturer presented
            # first. A coherent series is already authored in a teachable order,
            # so when our extracted edges contradict themselves, the author's own
            # sequencing is the better authority. Deterministic, so the demo is
            # reproducible.
            # Release a clip that is genuinely *inside* a cycle. Picking the
            # lowest-depth pending clip instead sacrifices a bystander: a clip
            # merely waiting on the cycle gets emitted early, and its own
            # prerequisites then play after it. Only clips in a non-trivial
            # strongly connected component are actually unorderable.
            stuck = nx.DiGraph()
            stuck.add_nodes_from(pending)
            for src in pending:
                for dst in out.get(src, ()):
                    if dst in pending:
                        stuck.add_edge(src, dst)
            cyclic = [
                node
                for component in nx.strongly_connected_components(stuck)
                if len(component) > 1
                for node in component
            ]
            victim = min(cyclic or pending, key=tiebreak)
            conflicts.extend(
                (src, victim) for src in pending if victim in out.get(src, ())
            )
            ready = [victim]
        chosen = min(ready, key=tiebreak)
        ordered.append(chosen)
        pending.discard(chosen)
        for nxt in out[chosen]:
            if nxt in pending:
                indeg[nxt] -= 1
        indeg[chosen] = 0

    # The concept you asked about plays last. Everything else on the path exists
    # to make it comprehensible, so ending anywhere else means the payoff lands
    # before the setup. This also settles most conflicts above: extracted edges
    # like 'attention REQUIRES value vectors' point at sub-parts of the target
    # taught inside the very explanation you are building toward, and treating
    # those as things to watch first inverts the lesson.
    if targets:
        target_set = {normalize(t) for t in targets}
        finale = [
            c for c in ordered if target_set & {normalize(x) for x in covers.get(c, [])}
        ]
        if finale and len(finale) < len(ordered):
            ordered = [c for c in ordered if c not in set(finale)] + finale

    return ordered, conflicts


# --------------------------------------------------------------------------- #
# 5. Prose (deterministic; P4 overwrites these with an LLM pass)                #
# --------------------------------------------------------------------------- #


def _human_list(items: Sequence[str]) -> str:
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def _minutes(seconds: float) -> str:
    mins = int(round(seconds / 60.0))
    return "under a minute" if mins <= 0 else f"{mins} min"


def _why(
    clip_covers: Sequence[str],
    targets: Set[str],
    dependents: Dict[str, List[str]],
    needed: Set[str],
    index: int,
    total: int,
) -> str:
    covered = _human_list(list(clip_covers))
    if not covered:
        return "Included to round out the path."
    if any(c in targets for c in clip_covers):
        if index == total - 1:
            return (
                f"The core explanation of {covered}. It plays last because everything "
                f"it assumes is already covered above."
            )
        return f"The core explanation of {covered}."
    here = set(clip_covers)
    downstream = sorted(
        {
            d
            for c in clip_covers
            for d in dependents.get(c, [])
            if d in needed and d not in here
        }
    )
    if downstream:
        verb = "depends" if len(downstream[:3]) == 1 else "depend"
        return (
            f"Teaches {covered}, which {_human_list(downstream[:3])} {verb} on "
            f"directly, so it has to come first."
        )
    return (
        f"Teaches {covered}, a prerequisite on the way to "
        f"{_human_list(sorted(targets))}."
    )


def _narration(
    targets: Sequence[str],
    known: Sequence[str],
    ordered_covers: Sequence[Sequence[str]],
    gaps: Sequence[str],
    watch_seconds: float,
    total_corpus_seconds: float,
    n_clips: int,
) -> str:
    goal = _human_list(sorted(targets))
    parts: List[str] = []
    if n_clips == 0:
        parts.append(f"Nothing left to watch for {goal}.")
    else:
        steps = [_human_list(list(c)) for c in ordered_covers if c]
        route = " then ".join(steps[:4])
        if len(steps) > 4:
            route += f" then {len(steps) - 4} more"
        parts.append(
            f"{n_clips} clips, {_minutes(watch_seconds)} out of "
            f"{_minutes(total_corpus_seconds)} of source video, to get you to {goal}."
        )
        if route:
            parts.append(f"You go {route}, in prerequisite order.")
    if known:
        them = "them" if len(known) > 1 else "it"
        parts.append(
            f"You marked {_human_list(sorted(known))} known, so {them} and "
            f"everything reachable only through {them} was pruned."
        )
    if gaps:
        parts.append(
            f"{_human_list(sorted(gaps))} "
            f"{'is' if len(gaps) == 1 else 'are'} required here but never taught "
            f"anywhere in this corpus."
        )
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# 6. The whole thing                                                            #
# --------------------------------------------------------------------------- #


def build_path(
    query: str,
    known: Optional[Sequence[str]] = None,
    store: Optional[GraphStore] = None,
    max_depth: int = MAX_DEPTH,
) -> PathResult:
    """Free-text query + known concepts -> an ordered, minimal playlist.

    ``store`` is any object satisfying :class:`GraphStore`. Switching between
    ``MemoryStore`` and ``Neo4jStore`` is that one argument.
    """
    if store is None:  # pragma: no cover - convenience for the CLI/REPL
        from store import MemoryStore  # local import keeps engine.py dependency-free

        store = MemoryStore()

    known = list(known or [])
    known_norm = [normalize(k) for k in known]

    videos = {v.id: v for v in store.videos()}
    clips_by_id = {c.id: c for c in store.clips()}
    all_clips = list(clips_by_id.values())
    explains = list(store.explains())
    total_corpus_seconds = float(sum(v.duration for v in videos.values()))

    # Defensive: a store should hand us a DAG, but never trust that.
    edges, _removed = break_cycles(store.requires_edges())
    adjacency = requires_adjacency(edges)
    dependents: Dict[str, List[str]] = {}
    for e in edges:
        dependents.setdefault(e.target, []).append(e.source)

    # 1. resolve
    targets = [normalize(t) for t in store.resolve(query)]
    if not targets:
        return PathResult(
            query=query,
            mode="graph",
            target_concepts=[],
            known=known,
            needed_concepts=[],
            playlist=[],
            gaps=[],
            total_corpus_seconds=total_corpus_seconds,
            watch_seconds=0.0,
            narration=(
                f"No concept in this corpus matched {query!r}, so there is no "
                f"prerequisite path to build."
            ),
        )

    # 2./3. closure, then prune what the learner already knows
    needed = prune_known(targets, adjacency, known_norm, max_depth=max_depth)

    # 4. greedy weighted set cover; uncoverable concepts become gaps
    selected, covers, gaps = select_clips(needed, all_clips, explains, targets=targets)

    # 5. topological order, prerequisites first
    teachable = needed - set(gaps)
    order = topo_order(teachable, adjacency)
    ordered, ordering_conflicts = order_clips(
        selected, covers, order, clips_by_id, adjacency, targets
    )

    # 6. emit
    target_set = set(targets)
    playlist: List[ClipSegment] = []
    for i, clip_id in enumerate(ordered):
        clip = clips_by_id[clip_id]
        video = videos.get(clip.video_id)
        clip_covers = sorted(
            covers.get(clip_id, []),
            key=lambda c: (order.index(c) if c in order else len(order), c),
        )
        playlist.append(
            ClipSegment(
                clip_id=clip.id,
                video_id=clip.video_id,
                video_title=video.title if video else clip.video_id,
                youtube_url=video.youtube_url if video else "",
                # served by the API from data/clips/; the player never touches
                # YouTube, so the demo works with no network
                media_url=f"/media/{clip.id}.mp4",
                start_seconds=float(clip.start),
                end_seconds=float(clip.end),
                covers=clip_covers,
                why=_why(
                    clip_covers, target_set, dependents, teachable, i, len(ordered)
                ),
            )
        )

    watch_seconds = float(sum(clips_by_id[c].duration for c in ordered))
    return PathResult(
        query=query,
        mode="graph",
        target_concepts=sorted(targets),
        known=known,
        needed_concepts=order,
        playlist=playlist,
        gaps=sorted(gaps),
        total_corpus_seconds=total_corpus_seconds,
        watch_seconds=watch_seconds,
        narration=_narration(
            targets,
            known,
            [p.covers for p in playlist],
            gaps,
            watch_seconds,
            total_corpus_seconds,
            len(playlist),
        ),
    )
