"""Load graph_payload.json into Neo4j Aura.

    python load.py                                  # loads fixtures/mini_graph.json
    python load.py ../ingestion/graph_payload.json  # P2's real payload
    python load.py --skip-embeddings                # no OpenAI calls

Everything is a MERGE, so running it twice is a no-op rather than a duplication.
After the MERGEs we pull the REQUIRES edges into networkx, break every cycle by
deleting the lowest-confidence edge in it, and write those deletions back — the
graph has to be a DAG before any query runs or both the depth-capped closure and
the topological sort are meaningless.

Schema:
    (:Video   {id, title, youtube_url, duration})
    (:Clip    {id, start, end, summary})-[:PART_OF]->(:Video)
    (:Concept {name, aliases, embedding})
    (:Clip)-[:EXPLAINS {score}]->(:Concept)
    (:Concept)-[:REQUIRES {confidence}]->(:Concept)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

from engine import RequiresEdge, break_cycles, normalize
from store import (
    EMBEDDING_DIMENSIONS,
    VECTOR_INDEX,
    default_payload_path,
    embed,
    embed_text_for,
    load_env,
    parse_payload,
)

CONSTRAINTS = [
    "CREATE CONSTRAINT concept_name IF NOT EXISTS "
    "FOR (c:Concept) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT clip_id IF NOT EXISTS FOR (c:Clip) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT video_id IF NOT EXISTS FOR (v:Video) REQUIRE v.id IS UNIQUE",
]

VECTOR_INDEX_CYPHER = f"""
CREATE VECTOR INDEX {VECTOR_INDEX} IF NOT EXISTS
FOR (c:Concept) ON (c.embedding)
OPTIONS {{indexConfig: {{
  `vector.dimensions`: {EMBEDDING_DIMENSIONS},
  `vector.similarity_function`: 'cosine'
}}}}
"""


def log(msg: str) -> None:
    print(msg, flush=True)


def load(
    payload_path: Path,
    skip_embeddings: bool = False,
    database: str | None = None,
) -> dict:
    from neo4j import GraphDatabase

    load_env()
    payload = json.loads(Path(payload_path).read_text())
    videos, clips, concepts, explains, requires = parse_payload(payload)
    log(
        f"payload {payload_path}: {len(videos)} videos, {len(clips)} clips, "
        f"{len(concepts)} concepts, {len(explains)} explains, "
        f"{len(requires)} requires"
    )

    # None => the connection's home database (Aura does not call it "neo4j")
    database = database or os.getenv("NEO4J_DATABASE") or None
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        driver.verify_connectivity()
        with driver.session(database=database) as s:
            for cypher in CONSTRAINTS:
                s.run(cypher)
            s.run(VECTOR_INDEX_CYPHER)
            log("constraints + vector index ensured")

            # ---- nodes ---------------------------------------------------- #
            s.run(
                """
                UNWIND $rows AS row
                MERGE (v:Video {id: row.id})
                SET v.title = row.title,
                    v.youtube_url = row.youtube_url,
                    v.duration = row.duration
                """,
                rows=[v.__dict__ for v in videos],
            )
            s.run(
                """
                UNWIND $rows AS row
                MERGE (c:Clip {id: row.id})
                SET c.start = row.start, c.end = row.end, c.summary = row.summary
                WITH c, row
                MATCH (v:Video {id: row.video_id})
                MERGE (c)-[:PART_OF]->(v)
                """,
                rows=[
                    {
                        "id": c.id,
                        "video_id": c.video_id,
                        "start": c.start,
                        "end": c.end,
                        "summary": c.summary,
                    }
                    for c in clips
                ],
            )
            s.run(
                """
                UNWIND $rows AS row
                MERGE (c:Concept {name: row.name})
                SET c.aliases = row.aliases
                """,
                rows=[{"name": c.name, "aliases": list(c.aliases)} for c in concepts],
            )
            log(f"merged {len(videos)} videos, {len(clips)} clips, "
                f"{len(concepts)} concepts")

            # ---- relationships -------------------------------------------- #
            s.run(
                """
                UNWIND $rows AS row
                MATCH (c:Clip {id: row.clip_id})
                MERGE (k:Concept {name: row.concept})
                MERGE (c)-[e:EXPLAINS]->(k)
                SET e.score = row.score
                """,
                rows=[e.__dict__ for e in explains],
            )
            s.run(
                """
                UNWIND $rows AS row
                MERGE (a:Concept {name: row.source})
                MERGE (b:Concept {name: row.target})
                MERGE (a)-[r:REQUIRES]->(b)
                SET r.confidence = row.confidence
                """,
                rows=[e.__dict__ for e in requires],
            )
            log(f"merged {len(explains)} EXPLAINS, {len(requires)} REQUIRES")

            # ---- cycle break, in the database ----------------------------- #
            rows = s.run(
                "MATCH (a:Concept)-[r:REQUIRES]->(b:Concept) "
                "RETURN a.name AS s, b.name AS t, r.confidence AS c "
                "ORDER BY a.name, b.name"
            ).data()
            live: List[RequiresEdge] = [
                RequiresEdge(r["s"], r["t"], float(r["c"] if r["c"] is not None else 1))
                for r in rows
            ]
            _survivors, removed = break_cycles(live)
            for r in removed:
                s.run(
                    "MATCH (:Concept {name: $s})-[r:REQUIRES]->(:Concept {name: $t}) "
                    "DELETE r",
                    s=r.source,
                    t=r.target,
                )
                log(f"  cycle-break: deleted {r}")
            log(
                f"cycle-break: removed {len(removed)} REQUIRES edge(s); "
                f"{len(live) - len(removed)} remain, graph is a DAG"
            )

            # ---- embeddings ----------------------------------------------- #
            if skip_embeddings:
                log("embeddings skipped (--skip-embeddings)")
            else:
                # Give each concept the words the corpus actually uses to teach
                # it, taken from the highest-scoring chapters that explain it.
                # Without this the index only has bare names to match against,
                # and a question like "how does a network learn" retrieves
                # whatever shares a token rather than what answers it.
                summaries_by_id = {c.id: (c.summary or "") for c in clips}
                context: Dict[str, List[str]] = {}
                for e in sorted(explains, key=lambda e: -float(e.score)):
                    summary = summaries_by_id.get(e.clip_id, "")
                    if not summary:
                        continue
                    bucket = context.setdefault(normalize(e.concept), [])
                    if len(bucket) < 3:
                        bucket.append(summary)

                names = [c.name for c in concepts]
                texts = [
                    embed_text_for(c.name, c.aliases, context.get(normalize(c.name), []))
                    for c in concepts
                ]
                vectors = embed(texts)
                s.run(
                    """
                    UNWIND $rows AS row
                    MATCH (c:Concept {name: row.name})
                    CALL db.create.setNodeVectorProperty(c, 'embedding', row.vec)
                    """,
                    rows=[
                        {"name": n, "vec": v} for n, v in zip(names, vectors)
                    ],
                )
                log(f"embedded {len(names)} concept names "
                    f"({EMBEDDING_DIMENSIONS}d, cosine)")

            counts = s.run(
                """
                MATCH (v:Video) WITH count(v) AS videos
                MATCH (c:Clip) WITH videos, count(c) AS clips
                MATCH (k:Concept) WITH videos, clips, count(k) AS concepts
                OPTIONAL MATCH ()-[e:EXPLAINS]->()
                WITH videos, clips, concepts, count(e) AS explains
                OPTIONAL MATCH ()-[r:REQUIRES]->()
                RETURN videos, clips, concepts, explains, count(r) AS requires
                """
            ).single()
            counts = dict(counts)
            log(f"graph now: {counts}")
            return counts
    finally:
        driver.close()


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Load a graph payload into Neo4j.")
    ap.add_argument(
        "payload",
        nargs="?",
        default=None,
        help="path to graph_payload.json (default: P2's file if present, "
        "else fixtures/mini_graph.json)",
    )
    ap.add_argument("--skip-embeddings", action="store_true")
    ap.add_argument("--database", default=None)
    args = ap.parse_args(argv)
    path = Path(args.payload) if args.payload else default_payload_path()
    load(path, skip_embeddings=args.skip_embeddings, database=args.database)
    return 0


if __name__ == "__main__":
    sys.exit(main())
