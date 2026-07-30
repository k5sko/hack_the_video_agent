"""Unit tests for the pure engine. Zero network, zero database.

Two layers:
  * hand-built micro-graphs that pin down each algorithm exactly
  * the real fixture (fixtures/mini_graph.json) driven through MemoryStore, which
    is what the demo actually runs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

GRAPH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GRAPH_DIR))

from engine import (  # noqa: E402
    Clip,
    Explains,
    RequiresEdge,
    break_cycles,
    build_path,
    closure,
    order_clips,
    prune_known,
    requires_adjacency,
    select_clips,
    topo_order,
)
from shared.types import PathResult  # noqa: E402
from store import MemoryStore  # noqa: E402

FIXTURE = GRAPH_DIR / "fixtures" / "mini_graph.json"
CORPUS_SECONDS = 8400.0
BUDGET = 0.25 * CORPUS_SECONDS


@pytest.fixture(scope="module")
def store() -> MemoryStore:
    return MemoryStore(FIXTURE)


@pytest.fixture(scope="module")
def adjacency(store: MemoryStore):
    return requires_adjacency(store.requires_edges())


# --------------------------------------------------------------------------- #
# cycle breaking                                                                #
# --------------------------------------------------------------------------- #


def test_break_cycles_drops_the_weakest_edge_of_a_two_cycle():
    edges = [RequiresEdge("a", "b", 3), RequiresEdge("b", "a", 1)]
    survivors, removed = break_cycles(edges)
    assert [(r.source, r.target) for r in removed] == [("b", "a")]
    assert [(s.source, s.target) for s in survivors] == [("a", "b")]


def test_break_cycles_handles_a_three_cycle():
    edges = [
        RequiresEdge("a", "b", 3),
        RequiresEdge("b", "c", 1),
        RequiresEdge("c", "a", 2),
    ]
    survivors, removed = break_cycles(edges)
    assert [(r.source, r.target) for r in removed] == [("b", "c")]
    assert len(survivors) == 2


def test_break_cycles_is_a_noop_on_a_dag():
    edges = [RequiresEdge("a", "b", 1), RequiresEdge("a", "c", 1),
             RequiresEdge("b", "c", 1)]
    survivors, removed = break_cycles(edges)
    assert removed == []
    assert len(survivors) == 3


def test_break_cycles_is_deterministic_under_input_order():
    edges = [
        RequiresEdge("a", "b", 3),
        RequiresEdge("b", "c", 1),
        RequiresEdge("c", "a", 1),
    ]
    first = break_cycles(edges)[1]
    second = break_cycles(list(reversed(edges)))[1]
    assert [(r.source, r.target) for r in first] == [(r.source, r.target) for r in second]


def test_fixture_cycles_are_broken_at_load_time(store: MemoryStore):
    removed = {(r.source, r.target) for r in store.removed_edges}
    assert removed == {
        ("cost function", "gradient descent"),
        ("softmax", "logits"),
        ("embedding", "attention"),
    }


def test_surviving_fixture_graph_is_a_dag(store: MemoryStore):
    import networkx as nx

    g = nx.DiGraph((e.source, e.target) for e in store.requires_edges())
    assert nx.is_directed_acyclic_graph(g)


# --------------------------------------------------------------------------- #
# closure                                                                       #
# --------------------------------------------------------------------------- #


def test_closure_follows_requires_transitively():
    adj = requires_adjacency(
        [RequiresEdge("a", "b"), RequiresEdge("b", "c"), RequiresEdge("x", "y")]
    )
    assert closure(["a"], adj) == {"a", "b", "c"}


def test_closure_is_depth_capped_at_five():
    chain = [RequiresEdge(f"n{i}", f"n{i + 1}") for i in range(10)]
    adj = requires_adjacency(chain)
    reached = closure(["n0"], adj)
    assert reached == {f"n{i}" for i in range(6)}  # n0 plus five hops
    assert "n6" not in reached


def test_closure_unions_multiple_targets():
    adj = requires_adjacency([RequiresEdge("a", "shared"), RequiresEdge("b", "shared")])
    assert closure(["a", "b"], adj) == {"a", "b", "shared"}


def test_attention_closure_on_the_fixture(adjacency):
    reached = closure(["attention"], adjacency)
    assert {"embedding", "softmax", "dot product", "positional encoding",
            "query key value", "vectors", "neural network"} <= reached
    # the removed cycle edge embedding->attention must not have dragged in logits
    assert "logits" not in reached
    assert "backpropagation" not in reached


# --------------------------------------------------------------------------- #
# prune known                                                                   #
# --------------------------------------------------------------------------- #


def test_known_node_and_its_exclusive_subtree_disappear():
    adj = requires_adjacency(
        [
            RequiresEdge("target", "known"),
            RequiresEdge("known", "only_under_known"),
            RequiresEdge("only_under_known", "deeper"),
        ]
    )
    assert prune_known(["target"], adj, ["known"]) == {"target"}


def test_a_node_reachable_by_an_unknown_route_survives():
    adj = requires_adjacency(
        [
            RequiresEdge("target", "known"),
            RequiresEdge("target", "other"),
            RequiresEdge("known", "shared"),
            RequiresEdge("other", "shared"),
        ]
    )
    assert prune_known(["target"], adj, ["known"]) == {"target", "other", "shared"}


def test_pruning_is_case_and_space_insensitive():
    adj = requires_adjacency([RequiresEdge("a", "neural network")])
    assert prune_known(["a"], adj, ["  Neural   Network "]) == {"a"}


def test_target_survives_even_if_marked_known():
    adj = requires_adjacency([RequiresEdge("a", "b")])
    assert prune_known(["a"], adj, ["a"]) == {"a"}


def test_known_neural_network_collapses_activations_but_keeps_exp(adjacency):
    needed = prune_known(["attention"], adjacency, ["neural network"])
    # activation function / sigmoid / relu hang off neural network alone
    assert "activation function" not in needed
    assert "sigmoid" not in needed
    assert "relu" not in needed
    # exponential function is also required by softmax, so it stays
    assert "exponential function" in needed
    # matrix multiplication is also required by query key value, so it stays
    assert "matrix multiplication" in needed


# --------------------------------------------------------------------------- #
# greedy weighted set cover                                                     #
# --------------------------------------------------------------------------- #


def _clip(cid, start, end):
    return Clip(id=cid, video_id="V", start=start, end=end)


def test_set_cover_prefers_score_per_second():
    clips = [_clip("cheap", 0, 100), _clip("dear", 0, 1000)]
    explains = [
        Explains("cheap", "a", 0.9),
        Explains("dear", "a", 0.95),
        Explains("dear", "b", 0.95),
    ]
    selected, covers, gaps = select_clips(["a"], clips, explains)
    assert selected == ["cheap"]
    assert covers == {"cheap": ["a"]}
    assert gaps == []


def test_set_cover_takes_the_multi_concept_clip_when_it_is_denser():
    clips = [_clip("wide", 0, 100), _clip("narrow_a", 0, 90), _clip("narrow_b", 0, 90)]
    explains = [
        Explains("wide", "a", 0.9),
        Explains("wide", "b", 0.9),
        Explains("narrow_a", "a", 0.9),
        Explains("narrow_b", "b", 0.9),
    ]
    selected, _covers, _gaps = select_clips(["a", "b"], clips, explains)
    assert selected == ["wide"]


def test_set_cover_ignores_scores_for_concepts_that_are_not_needed():
    clips = [_clip("c1", 0, 100)]
    explains = [Explains("c1", "needed", 0.5), Explains("c1", "irrelevant", 9.0)]
    selected, covers, _gaps = select_clips(["needed"], clips, explains)
    assert selected == ["c1"]
    assert covers["c1"] == ["needed"]


def test_set_cover_is_deterministic_on_exact_ties():
    clips = [_clip("zeta", 0, 100), _clip("alpha", 0, 100)]
    explains = [Explains("zeta", "a", 0.9), Explains("alpha", "a", 0.9)]
    assert select_clips(["a"], clips, explains)[0] == ["alpha"]
    assert select_clips(["a"], list(reversed(clips)), explains)[0] == ["alpha"]


def test_uncovered_concepts_become_gaps_not_silence():
    clips = [_clip("c1", 0, 100)]
    explains = [Explains("c1", "covered", 0.9)]
    selected, covers, gaps = select_clips(["covered", "orphan"], clips, explains)
    assert selected == ["c1"]
    assert gaps == ["orphan"]
    assert "orphan" not in covers.get("c1", [])


# --------------------------------------------------------------------------- #
# ordering                                                                      #
# --------------------------------------------------------------------------- #


def test_topo_order_puts_prerequisites_first():
    adj = requires_adjacency([RequiresEdge("top", "mid"), RequiresEdge("mid", "base")])
    order = topo_order(["top", "mid", "base"], adj)
    assert order == ["base", "mid", "top"]


def test_topo_order_is_deterministic_across_independent_nodes():
    adj = requires_adjacency([RequiresEdge("top", "b"), RequiresEdge("top", "a")])
    assert topo_order(["top", "a", "b"], adj) == ["a", "b", "top"]


def test_topo_order_ignores_edges_leaving_the_pruned_set():
    adj = requires_adjacency([RequiresEdge("top", "mid"), RequiresEdge("mid", "gone")])
    assert topo_order(["top", "mid"], adj) == ["mid", "top"]


def test_clips_sort_by_earliest_concept_then_video_then_start():
    clips = {
        "late": Clip("late", "V1", 0, 10),
        "early": Clip("early", "V2", 0, 10),
        "tie_a": Clip("tie_a", "V1", 500, 510),
        "tie_b": Clip("tie_b", "V1", 100, 110),
    }
    covers = {
        "late": ["top"],
        "early": ["base"],
        "tie_a": ["mid"],
        "tie_b": ["mid"],
    }
    order = ["base", "mid", "top"]
    assert order_clips(list(clips), covers, order, clips) == [
        "early",
        "tie_b",
        "tie_a",
        "late",
    ]


def test_fixture_playlist_is_ordered_by_earliest_covered_concept(store: MemoryStore):
    """The spec'd rule: a clip sorts by the *earliest* topo position it covers.

    (A clip that bundles a deep prerequisite with a shallow one therefore lands
    early — that is the documented behaviour, not a bug.)
    """
    result = build_path("attention", [], store)
    pos = {name: i for i, name in enumerate(result.needed_concepts)}
    keys = [min(pos[c] for c in seg.covers) for seg in result.playlist]
    assert keys == sorted(keys)


def test_target_explanation_plays_last(store: MemoryStore):
    for query in ("attention", "backpropagation"):
        result = build_path(query, [], store)
        assert result.playlist[-1].covers[-1] == result.target_concepts[0]
        assert result.needed_concepts[-1] == result.target_concepts[0]


# --------------------------------------------------------------------------- #
# resolution                                                                    #
# --------------------------------------------------------------------------- #


def test_memory_store_resolves_exact_alias_and_fuzzy(store: MemoryStore):
    assert store.resolve("attention") == ["attention"]
    assert store.resolve("  ATTENTION ") == ["attention"]
    assert store.resolve("backprop") == ["backpropagation"]          # alias
    assert "gradient descent" in store.resolve("how does a network learn")


def test_memory_store_returns_nothing_for_gibberish(store: MemoryStore):
    assert store.resolve("zzzzqqq") == []


def test_unresolvable_query_yields_an_empty_but_valid_path(store: MemoryStore):
    result = build_path("zzzzqqq", [], store)
    assert isinstance(result, PathResult)
    assert result.playlist == []
    assert result.target_concepts == []
    assert result.watch_seconds == 0.0


# --------------------------------------------------------------------------- #
# end to end on the fixture                                                     #
# --------------------------------------------------------------------------- #

CANNED = ["attention", "backpropagation", "how does a network learn"]


@pytest.mark.parametrize("query", CANNED)
def test_canned_queries_validate_and_fit_the_budget(store: MemoryStore, query: str):
    result = build_path(query, [], store)
    PathResult.model_validate(result.model_dump())
    assert result.mode == "graph"
    assert result.target_concepts
    assert result.playlist
    assert result.total_corpus_seconds == CORPUS_SECONDS
    assert result.watch_seconds < BUDGET, (
        f"{query}: {result.watch_seconds}s exceeds 25% of the corpus"
    )
    assert result.narration


@pytest.mark.parametrize("query", CANNED)
def test_canned_queries_are_deterministic(store: MemoryStore, query: str):
    a = build_path(query, ["neural network"], store)
    b = build_path(query, ["neural network"], store)
    assert a.model_dump() == b.model_dump()


def test_how_does_a_network_learn_resolves_to_gradient_descent(store: MemoryStore):
    result = build_path("how does a network learn", [], store)
    assert "gradient descent" in result.target_concepts


KNOWN_SETS = [
    [],
    ["linear algebra"],
    ["neural network"],
    ["vectors"],
    ["linear algebra", "neural network"],
    ["linear algebra", "neural network", "vectors"],
    ["linear algebra", "neural network", "vectors", "softmax"],
    ["linear algebra", "neural network", "vectors", "softmax", "embedding"],
    ["derivative"],
    ["derivative", "gradient descent"],
    ["derivative", "gradient descent", "neural network"],
]


@pytest.mark.parametrize("query", CANNED)
def test_more_known_never_increases_watch_time(store: MemoryStore, query: str):
    previous = None
    for known in KNOWN_SETS:
        result = build_path(query, known, store)
        PathResult.model_validate(result.model_dump())
        if previous is not None and set(previous[0]) <= set(known):
            assert result.watch_seconds <= previous[1], (
                f"{query}: knowing {known} watches more than knowing {previous[0]}"
            )
        previous = (known, result.watch_seconds)


def test_positional_encoding_is_a_gap_and_never_in_the_playlist(store: MemoryStore):
    for known in ([], ["neural network"], ["linear algebra", "neural network"]):
        result = build_path("attention", known, store)
        assert "positional encoding" in result.gaps
        assert "positional encoding" not in result.needed_concepts
        for segment in result.playlist:
            assert "positional encoding" not in segment.covers


def test_gaps_and_needed_concepts_are_disjoint(store: MemoryStore):
    for query in CANNED:
        result = build_path(query, [], store)
        assert not (set(result.gaps) & set(result.needed_concepts))


def test_every_needed_concept_is_covered_by_some_selected_clip(store: MemoryStore):
    for query in CANNED:
        result = build_path(query, [], store)
        covered = {c for segment in result.playlist for c in segment.covers}
        assert covered == set(result.needed_concepts)


def test_known_concepts_never_appear_in_the_path(store: MemoryStore):
    known = ["linear algebra", "neural network"]
    result = build_path("attention", known, store)
    for k in known:
        assert k not in result.needed_concepts


def test_watch_seconds_matches_the_playlist(store: MemoryStore):
    result = build_path("attention", ["linear algebra"], store)
    assert result.watch_seconds == sum(
        s.end_seconds - s.start_seconds for s in result.playlist
    )


def test_playlist_clips_carry_real_video_metadata(store: MemoryStore):
    result = build_path("attention", [], store)
    for segment in result.playlist:
        assert segment.youtube_url.startswith("https://www.youtube.com/watch?v=")
        assert segment.video_title
        assert segment.end_seconds > segment.start_seconds
        assert segment.why


def test_fixture_matches_the_documented_corpus_shape():
    payload = json.loads(FIXTURE.read_text())
    assert len(payload["videos"]) == 7
    assert sum(v["duration"] for v in payload["videos"]) == CORPUS_SECONDS
    explained = {e["concept"] for e in payload["explains"]}
    assert "positional encoding" not in explained
