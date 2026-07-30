"""Path-quality invariants, checked against the real extracted corpus.

The project plan specified acceptance criteria as fixed numbers -- "under 25% of
corpus", "4 clips, 12 minutes", "50-80 concepts". Those were written before any
data existed, and optimising for them is actively harmful: watch time was both
the objective of the set cover and the measure of success, so the engine was
rewarded for building playlists out of 31-second title cards. It scored 6.5% of
corpus and taught nothing.

These tests assert what we actually want instead: that the path teaches, that it
teaches in a possible order, and that nothing is quietly hidden. Watch time is
reported by test_reports_watch_time rather than bounded -- it is a result, not a
target. The one time-related property worth asserting is monotonicity: telling
Beeline you already know something must never cost you more video.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Set

import pytest

HERE = Path(__file__).resolve().parent
GRAPH = HERE.parent
sys.path.insert(0, str(GRAPH))

from engine import MIN_COVERAGE_SCORE, build_path, normalize  # noqa: E402
from store import MemoryStore  # noqa: E402

PAYLOAD = GRAPH.parent / "ingestion" / "graph_payload.json"

# The queries the demo actually runs. If these degrade, the demo degrades.
QUERIES = [
    ("attention", ["linear algebra", "neural network"]),
    ("attention", []),
    ("backpropagation", ["linear algebra", "neural network"]),
    ("backpropagation", []),
]


pytestmark = pytest.mark.skipif(
    not PAYLOAD.exists(),
    reason="graph_payload.json not built yet (run the ingestion agent)",
)


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(PAYLOAD.read_text())


@pytest.fixture(scope="module")
def store() -> MemoryStore:
    return MemoryStore(str(PAYLOAD))


@pytest.fixture(scope="module")
def prereqs(payload) -> Dict[str, Set[str]]:
    """concept -> set of concepts it directly REQUIRES."""
    out: Dict[str, Set[str]] = {}
    for e in payload["requires"]:
        out.setdefault(normalize(e["from"]), set()).add(normalize(e["to"]))
    return out


@pytest.fixture(scope="module")
def explains_index(payload) -> Dict[str, Dict[str, float]]:
    """clip_id -> {concept: score}."""
    out: Dict[str, Dict[str, float]] = {}
    for e in payload["explains"]:
        out.setdefault(e["clip_id"], {})[normalize(e["concept"])] = float(e["score"])
    return out


def _paths(store):
    return [(q, k, build_path(q, k, store)) for q, k in QUERIES]


# --------------------------------------------------------------------------- #
# It teaches                                                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("query,known", QUERIES)
def test_target_is_covered_by_the_playlist(query, known, store):
    """You must be shown the thing you asked about."""
    result = build_path(query, known, store)
    covered = {normalize(c) for seg in result.playlist for c in seg.covers}
    missing = {normalize(t) for t in result.target_concepts} - covered
    assert not missing, f"{query!r} never actually covers {sorted(missing)}"


@pytest.mark.parametrize("query,known", QUERIES)
def test_every_covering_clip_actually_teaches(query, known, store, explains_index):
    """No concept may be 'covered' by a clip that merely mentions it.

    This is the invariant that stops the playlist collapsing into title cards.
    """
    result = build_path(query, known, store)
    weak = []
    for seg in result.playlist:
        for concept in seg.covers:
            score = explains_index.get(seg.clip_id, {}).get(normalize(concept))
            if score is None or score < MIN_COVERAGE_SCORE:
                weak.append((seg.clip_id, concept, score))
    assert not weak, f"{query!r} covers concepts with weak/absent EXPLAINS: {weak}"


@pytest.mark.parametrize("query,known", QUERIES)
def test_target_gets_the_best_available_explanation(query, known, store, explains_index):
    """The target is not scaffolding -- it must get the strongest clip, not the
    shortest one that name-drops it."""
    result = build_path(query, known, store)
    for target in (normalize(t) for t in result.target_concepts):
        chosen = [
            seg for seg in result.playlist if target in {normalize(c) for c in seg.covers}
        ]
        if not chosen:
            continue
        got = explains_index[chosen[0].clip_id][target]
        best = max(
            (scores[target] for scores in explains_index.values() if target in scores),
            default=got,
        )
        assert got >= best - 1e-9, (
            f"{query!r} explains {target!r} with a {got} clip when a {best} one exists"
        )


# --------------------------------------------------------------------------- #
# In a possible order                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("query,known", QUERIES)
def test_prerequisites_play_before_dependents(query, known, store, prereqs):
    """For any two concepts on the path, if A requires B then B is taught first.

    Concepts not on the path are irrelevant here -- they were either pruned as
    known or surfaced as gaps, and both are legitimate reasons to be absent.
    """
    result = build_path(query, known, store)
    first_seen: Dict[str, int] = {}
    for i, seg in enumerate(result.playlist):
        for concept in seg.covers:
            first_seen.setdefault(normalize(concept), i)

    # Which clip teaches each concept, and what each clip teaches in total.
    clip_of = {}
    taught_by_clip: Dict[str, Set[str]] = {}
    for seg in result.playlist:
        names = {normalize(c) for c in seg.covers}
        taught_by_clip[seg.clip_id] = names
        for name in names:
            clip_of.setdefault(name, seg.clip_id)

    def mutually_constrained(a: str, b: str) -> bool:
        """True when the clips teaching a and b each depend on the other.

        A clip is an indivisible bundle. If clip X teaches something that
        requires a concept from clip Y, *and* Y teaches something requiring a
        concept from X, no ordering of the two can satisfy both. That conflict is
        a property of how the corpus bundles concepts into chapters, not a bug in
        the sort, so it is the one violation we tolerate.
        """
        ca, cb = clip_of.get(a), clip_of.get(b)
        if ca is None or cb is None or ca == cb:
            return True

        def depends(x: str, y: str) -> bool:
            return any(
                prereq in taught_by_clip[y]
                for concept in taught_by_clip[x]
                for prereq in prereqs.get(concept, ())
            )

        return depends(ca, cb) and depends(cb, ca)

    violations: List[str] = []
    for concept, position in first_seen.items():
        for prereq in prereqs.get(concept, ()):
            if prereq in first_seen and first_seen[prereq] > position:
                if mutually_constrained(concept, prereq):
                    continue
                violations.append(
                    f"{concept!r}@{position} plays before its prerequisite "
                    f"{prereq!r}@{first_seen[prereq]}"
                )
    assert not violations, f"{query!r}: " + "; ".join(violations)


@pytest.mark.parametrize("query,known", QUERIES)
def test_no_clip_plays_twice(query, known, store):
    result = build_path(query, known, store)
    ids = [seg.clip_id for seg in result.playlist]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("query,known", QUERIES)
def test_clip_boundaries_are_sane(query, known, store):
    for seg in build_path(query, known, store).playlist:
        assert seg.end_seconds > seg.start_seconds
        assert seg.start_seconds >= 0


# --------------------------------------------------------------------------- #
# Nothing is hidden                                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("query,known", QUERIES)
def test_gaps_are_disjoint_from_what_is_covered(query, known, store):
    """A gap means 'no clip teaches this'. If a clip covers it, it is not a gap."""
    result = build_path(query, known, store)
    covered = {normalize(c) for seg in result.playlist for c in seg.covers}
    assert not (set(map(normalize, result.gaps)) & covered)


@pytest.mark.parametrize("query,known", QUERIES)
def test_gaps_are_genuinely_untaught(query, known, store, explains_index):
    """Every gap must be unteachable by the whole corpus, not merely unselected.

    A concept some clip could have taught is a selection bug wearing an honesty
    costume.
    """
    result = build_path(query, known, store)
    teachable = {
        concept
        for scores in explains_index.values()
        for concept, score in scores.items()
        if score >= MIN_COVERAGE_SCORE
    }
    fake = sorted(set(map(normalize, result.gaps)) & teachable)
    assert not fake, f"{query!r} reports gaps the corpus can actually teach: {fake}"


@pytest.mark.parametrize("query,known", QUERIES)
def test_known_concepts_never_appear_on_the_path(query, known, store):
    result = build_path(query, known, store)
    covered = {normalize(c) for seg in result.playlist for c in seg.covers}
    assert not (covered & {normalize(k) for k in known})


# --------------------------------------------------------------------------- #
# Time is a result, not a target                                                #
# --------------------------------------------------------------------------- #


def test_knowing_more_never_costs_more_video(store):
    """The one monotonicity that must hold: marking a concept known can only
    shrink the path. If it grows, pruning is wrong somewhere."""
    escalating = [
        [],
        ["linear algebra"],
        ["linear algebra", "neural network"],
        ["linear algebra", "neural network", "vectors"],
    ]
    for query in ("attention", "backpropagation"):
        previous = None
        for known in escalating:
            watch = build_path(query, known, store).watch_seconds
            if previous is not None:
                assert watch <= previous + 1e-9, (
                    f"{query!r}: knowing {known} raised watch time "
                    f"from {previous}s to {watch}s"
                )
            previous = watch


def test_path_is_much_shorter_than_the_corpus(store):
    """Deliberately loose. The product promise is 'meaningfully less than
    watching everything', not a specific percentage -- a tight bound here is what
    pushed the engine toward title cards in the first place."""
    for query, known in QUERIES:
        result = build_path(query, known, store)
        assert result.watch_seconds < 0.5 * result.total_corpus_seconds


def test_reports_watch_time(store, capsys):
    """Not an assertion -- a record, so regressions are visible in test output."""
    with capsys.disabled():
        print()
        for query, known in QUERIES:
            r = build_path(query, known, store)
            pct = 100 * r.watch_seconds / r.total_corpus_seconds
            print(
                f"    {query!r:18} known={len(known)}  "
                f"{len(r.playlist):>2} clips  {r.watch_seconds/60:>5.1f}m  "
                f"{pct:>4.1f}%  gaps={len(r.gaps)}"
            )
