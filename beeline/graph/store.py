"""Two interchangeable graph stores behind one interface.

``MemoryStore``  — reads ``graph_payload.json`` straight off disk. Zero network,
                   used by the tests and as the offline fallback for the demo.
``Neo4jStore``   — reads the loaded Aura graph over Cypher, resolves queries with
                   an OpenAI embedding against the ``concept_embedding`` vector
                   index. Read-only: nothing here writes at query time.

Both satisfy :class:`engine.GraphStore`, so switching is one line:

    store = MemoryStore("fixtures/mini_graph.json")   # or Neo4jStore()
    result = build_path("attention", ["neural network"], store)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from engine import (
    Clip,
    Concept,
    Explains,
    RemovedEdge,
    RequiresEdge,
    Video,
    break_cycles,
    normalize,
)

HERE = Path(__file__).resolve().parent          # beeline/graph
REPO_ROOT = HERE.parents[1]                     # repo root (holds the .env)
DEFAULT_PAYLOAD = HERE / "fixtures" / "mini_graph.json"
P2_PAYLOAD = HERE.parent / "ingestion" / "graph_payload.json"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
VECTOR_INDEX = "concept_embedding"

# Question scaffolding, not content. Kept deliberately small: dropping a real
# noun here would break resolution far more often than it helps.
_STOPWORDS = {
    "a", "an", "the", "of", "is", "are", "was", "were", "be", "does", "do",
    "did", "how", "what", "why", "when", "which", "who", "to", "in", "on",
    "for", "and", "or", "it", "its", "with", "that", "this", "these", "i",
    "me", "my", "you", "your", "about", "explain", "explained", "understand",
    "work", "works", "really", "actually", "even",
}


# --------------------------------------------------------------------------- #
# env + OpenAI helpers (shared with load.py)                                    #
# --------------------------------------------------------------------------- #


def load_env() -> None:
    """Credentials live in the repo-root .env, not in beeline/."""
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")


def embed_text_for(
    name: str,
    aliases: Sequence[str] = (),
    context: Sequence[str] = (),
) -> str:
    """What we actually send to the embedding model for a concept.

    A bare concept name is a terrible retrieval anchor, because it gives the
    index nothing but the words themselves. Asking "how does a network learn"
    against bare names matches 'feedforward networks' -- it shares the token
    'network' -- rather than 'gradient descent', which is the actual answer.

    So a concept is embedded alongside how the corpus talks about it: its
    aliases, plus the summaries of the chapters that teach it. 'gradient descent'
    then carries text like "the video explains how the network adjusts its
    weights to reduce cost", which sits close to the question a learner would
    actually type.
    """
    aliases = [a for a in aliases if normalize(a) != normalize(name)]
    parts = [name]
    if aliases:
        parts.append(f"also called {', '.join(aliases)}")
    for summary in context:
        summary = " ".join(str(summary).split())
        if summary:
            parts.append(summary)
    # Keep it well inside the model's window; the first summaries are the
    # highest-scoring ones and carry most of the signal.
    return ". ".join(parts)[:1500]


def embed(texts: Sequence[str]) -> List[List[float]]:
    """OpenAI embeddings, batched. Used at load time and at resolve time."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    out: List[List[float]] = []
    for i in range(0, len(texts), 128):
        chunk = list(texts[i : i + 128])
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL, input=chunk, dimensions=EMBEDDING_DIMENSIONS
        )
        out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
    return out


# --------------------------------------------------------------------------- #
# payload parsing                                                               #
# --------------------------------------------------------------------------- #


def _first(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def parse_payload(payload: dict) -> Tuple[
    List[Video], List[Clip], List[Concept], List[Explains], List[RequiresEdge]
]:
    """Payload dict -> engine dataclasses. Tolerant of a couple of key spellings
    so that P2's real ``graph_payload.json`` drops in without edits."""
    videos = [
        Video(
            id=str(v["id"]),
            title=str(_first(v, "title", "name", default=v["id"])),
            youtube_url=str(_first(v, "youtube_url", "url", default="")),
            duration=float(_first(v, "duration", "duration_seconds", default=0.0)),
        )
        for v in payload.get("videos", [])
    ]
    clips = [
        Clip(
            id=str(c["id"]),
            video_id=str(_first(c, "video_id", "video", default="")),
            start=float(_first(c, "start", "start_seconds", default=0.0)),
            end=float(_first(c, "end", "end_seconds", default=0.0)),
            summary=str(_first(c, "summary", "text", default="")),
        )
        for c in payload.get("clips", [])
    ]
    concepts = [
        Concept(
            name=normalize(_first(c, "name", "concept")),
            aliases=tuple(
                dict.fromkeys(
                    normalize(a) for a in (c.get("aliases") or []) if str(a).strip()
                )
            ),
        )
        for c in payload.get("concepts", [])
    ]
    explains = [
        Explains(
            clip_id=str(_first(e, "clip_id", "clip")),
            concept=normalize(_first(e, "concept", "concept_name")),
            score=float(_first(e, "score", "confidence", default=1.0)),
        )
        for e in payload.get("explains", [])
    ]
    requires = [
        RequiresEdge(
            source=normalize(_first(r, "from", "source", "from_concept")),
            target=normalize(_first(r, "to", "target", "to_concept")),
            confidence=float(_first(r, "confidence", "score", default=1.0)),
        )
        for r in payload.get("requires", [])
    ]
    requires = [r for r in requires if r.source and r.target and r.source != r.target]
    return videos, clips, concepts, explains, requires


# --------------------------------------------------------------------------- #
# MemoryStore                                                                   #
# --------------------------------------------------------------------------- #


def _stem(token: str) -> str:
    """Crudest possible stemmer, enough to make learn/learns/learning agree."""
    for suffix, keep in (("ing", 5), ("es", 5), ("s", 4)):
        if len(token) >= keep and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _tokens(text: str) -> List[str]:
    return [
        _stem(t)
        for t in re.split(r"[^a-z0-9]+", normalize(text))
        if t and t not in _STOPWORDS
    ]


class MemoryStore:
    """graph_payload.json in memory. Cycles are broken at load time, defensively."""

    def __init__(self, payload_path: Optional[os.PathLike | str] = None):
        self.path = Path(payload_path) if payload_path else default_payload_path()
        payload = json.loads(Path(self.path).read_text())
        (
            self._videos,
            self._clips,
            self._concepts,
            self._explains,
            raw_requires,
        ) = parse_payload(payload)
        self._requires, self.removed_edges = break_cycles(raw_requires)

        self._by_name: Dict[str, str] = {c.name: c.name for c in self._concepts}
        self._by_alias: Dict[str, str] = {}
        for c in self._concepts:
            for alias in c.aliases:
                self._by_alias.setdefault(alias, c.name)

    # -- engine.GraphStore -------------------------------------------------- #

    def videos(self) -> List[Video]:
        return list(self._videos)

    def clips(self) -> List[Clip]:
        return list(self._clips)

    def concepts(self) -> List[Concept]:
        return list(self._concepts)

    def explains(self) -> List[Explains]:
        return list(self._explains)

    def requires_edges(self) -> List[RequiresEdge]:
        return list(self._requires)

    def resolve(self, query: str) -> List[str]:
        """exact -> alias -> substring -> token overlap. No network, no model."""
        q = normalize(query)
        if not q:
            return []
        if q in self._by_name:
            return [q]
        if q in self._by_alias:
            return [self._by_alias[q]]

        scored: List[Tuple[float, str]] = []
        q_tokens = set(_tokens(q))
        for concept in self._concepts:
            surfaces = [concept.name, *concept.aliases]
            best = 0.0
            for surface in surfaces:
                if surface == q:
                    best = max(best, 1.0)
                elif surface in q or q in surface:
                    # substring: longer overlaps win, cap below an exact match
                    best = max(best, 0.9 * len(surface) / max(len(q), len(surface)))
                s_tokens = set(_tokens(surface))
                if q_tokens and s_tokens:
                    overlap = len(q_tokens & s_tokens) / len(q_tokens | s_tokens)
                    best = max(best, 0.8 * overlap)
            if best > 0.15:
                scored.append((best, concept.name))

        if not scored:
            return []
        scored.sort(key=lambda x: (-x[0], x[1]))
        top = [scored[0][1]]
        if len(scored) > 1 and scored[0][0] - scored[1][0] <= 0.05:
            top.append(scored[1][1])
        return top


def default_payload_path() -> Path:
    """Prefer P2's real payload when it exists, otherwise the hand-built fixture."""
    return P2_PAYLOAD if P2_PAYLOAD.exists() else DEFAULT_PAYLOAD


# --------------------------------------------------------------------------- #
# Neo4jStore                                                                    #
# --------------------------------------------------------------------------- #


class Neo4jStore:
    """Read-only view of the loaded Aura graph. Never writes."""

    def __init__(self, driver=None, database: Optional[str] = None, top_k: int = 5):
        load_env()
        self.top_k = top_k
        # Aura names the database after the instance id, not "neo4j" — leaving
        # this as None makes the driver use the connection's home database.
        self.database = database or os.getenv("NEO4J_DATABASE") or None
        self._owns_driver = driver is None
        if driver is None:
            from neo4j import GraphDatabase

            kwargs = {
                "auth": (
                    os.environ["NEO4J_USERNAME"],
                    os.environ["NEO4J_PASSWORD"],
                )
            }
            try:
                # db.index.vector.queryNodes is flagged deprecated on 5.27-aura in
                # favour of Cypher 25's SEARCH; the procedure is what actually
                # works today, so mute the notification rather than the feature.
                from neo4j import NotificationDisabledClassification

                kwargs["notifications_disabled_classifications"] = [
                    NotificationDisabledClassification.DEPRECATION
                ]
            except ImportError:  # pragma: no cover - older driver
                pass
            driver = GraphDatabase.driver(os.environ["NEO4J_URI"], **kwargs)
        self.driver = driver

    # -- plumbing ----------------------------------------------------------- #

    def _read(self, cypher: str, **params) -> List[dict]:
        with self.driver.session(database=self.database) as session:
            return [r.data() for r in session.run(cypher, **params)]

    def close(self) -> None:
        if self._owns_driver:
            self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- engine.GraphStore -------------------------------------------------- #

    def videos(self) -> List[Video]:
        rows = self._read(
            "MATCH (v:Video) RETURN v.id AS id, v.title AS title, "
            "v.youtube_url AS url, v.duration AS duration ORDER BY v.id"
        )
        return [
            Video(
                id=r["id"],
                title=r["title"] or r["id"],
                youtube_url=r["url"] or "",
                duration=float(r["duration"] or 0.0),
            )
            for r in rows
        ]

    def clips(self) -> List[Clip]:
        rows = self._read(
            "MATCH (c:Clip)-[:PART_OF]->(v:Video) "
            "RETURN c.id AS id, v.id AS video_id, c.start AS start, c.end AS end, "
            "coalesce(c.summary, '') AS summary ORDER BY c.id"
        )
        return [
            Clip(
                id=r["id"],
                video_id=r["video_id"],
                start=float(r["start"]),
                end=float(r["end"]),
                summary=r["summary"],
            )
            for r in rows
        ]

    def concepts(self) -> List[Concept]:
        rows = self._read(
            "MATCH (c:Concept) RETURN c.name AS name, "
            "coalesce(c.aliases, []) AS aliases ORDER BY c.name"
        )
        return [Concept(name=r["name"], aliases=tuple(r["aliases"])) for r in rows]

    def explains(self) -> List[Explains]:
        rows = self._read(
            "MATCH (c:Clip)-[e:EXPLAINS]->(k:Concept) "
            "RETURN c.id AS clip_id, k.name AS concept, e.score AS score "
            "ORDER BY c.id, k.name"
        )
        return [
            Explains(
                clip_id=r["clip_id"],
                concept=r["concept"],
                score=float(r["score"] if r["score"] is not None else 1.0),
            )
            for r in rows
        ]

    def requires_edges(self) -> List[RequiresEdge]:
        rows = self._read(
            "MATCH (a:Concept)-[r:REQUIRES]->(b:Concept) "
            "RETURN a.name AS source, b.name AS target, r.confidence AS confidence "
            "ORDER BY a.name, b.name"
        )
        return [
            RequiresEdge(
                source=r["source"],
                target=r["target"],
                confidence=float(
                    r["confidence"] if r["confidence"] is not None else 1.0
                ),
            )
            for r in rows
        ]

    def resolve(self, query: str) -> List[str]:
        """Embed the query, hit the vector index, take the top concept.

        Returns two concepts when the top-2 similarities are within 0.02 — the
        question genuinely straddles both, and the closures get unioned.
        Exact name/alias hits short-circuit the model call.
        """
        q = normalize(query)
        if not q:
            return []

        exact = self._read(
            "MATCH (c:Concept) WHERE c.name = $q OR $q IN coalesce(c.aliases, []) "
            "RETURN c.name AS name ORDER BY c.name LIMIT 1",
            q=q,
        )
        if exact:
            return [exact[0]["name"]]

        vector = embed([query])[0]
        rows = self._read(
            "CALL db.index.vector.queryNodes($index, $k, $vec) "
            "YIELD node, score RETURN node.name AS name, score ORDER BY score DESC",
            index=VECTOR_INDEX,
            k=self.top_k,
            vec=vector,
        )
        if not rows:
            return []
        out = [rows[0]["name"]]
        if len(rows) > 1 and (rows[0]["score"] - rows[1]["score"]) <= 0.02:
            out.append(rows[1]["name"])
        return out

    # -- diagnostics -------------------------------------------------------- #

    def counts(self) -> Dict[str, int]:
        row = self._read(
            "MATCH (v:Video) WITH count(v) AS videos "
            "MATCH (c:Clip) WITH videos, count(c) AS clips "
            "MATCH (k:Concept) WITH videos, clips, count(k) AS concepts "
            "OPTIONAL MATCH ()-[e:EXPLAINS]->() "
            "WITH videos, clips, concepts, count(e) AS explains "
            "OPTIONAL MATCH ()-[r:REQUIRES]->() "
            "RETURN videos, clips, concepts, explains, count(r) AS requires"
        )
        return {k: int(v) for k, v in row[0].items()} if row else {}


def get_store(kind: str = "memory", payload: Optional[str] = None):
    """One line to switch backends."""
    if kind == "neo4j":
        return Neo4jStore()
    if kind == "memory":
        return MemoryStore(payload)
    raise ValueError(f"unknown store {kind!r} (expected 'memory' or 'neo4j')")
