# graph/ — P3

**Owner:** Person 3

**Responsibility:** the Neo4j schema and loader, and the path engine that turns a
query into an ordered, minimal playlist.

**Expected input:** `graph_payload.json` from P2, plus a Neo4j Aura Free
instance and `OPENAI_API_KEY` (concept-name embeddings at load time).

**Expected output:** one importable function `build_path(query, known) ->
PathResult` (validated against `shared/types.py`), plus a CLI that prints the
same JSON.

**How to run:**

```bash
cd graph
python load.py ../ingestion/graph_payload.json   # idempotent MERGEs + cycle-break
python path.py "attention" --known "neural network,linear algebra"
```

**Pipeline:**

1. **Resolve** — embed the query, hit the Neo4j vector index, take the top concept
   (top-2 if scores are close) as `target_concepts`
2. **Closure** — `MATCH (t:Concept {name:$t})-[:REQUIRES*0..5]->(p) RETURN DISTINCT p`
3. **Prune known** — BFS from the target that stops at known nodes; drop known
   nodes and anything reachable only through them
4. **Select clips** — greedy weighted set cover, repeatedly taking the clip that
   maximizes (sum of EXPLAINS scores over still-needed concepts ÷ duration).
   Any concept no clip covers goes to `gaps`
5. **Order** — topological sort (Kahn) of the pruned concept DAG; each clip sorts
   by the earliest topo position among the concepts it covers, ties by (video, start)
6. **Emit** — a `PathResult` including `watch_seconds` and `total_corpus_seconds`

**Constraints:** the graph must be a DAG before any query runs — pull REQUIRES
edges into networkx at load time, find cycles, delete the lowest-confidence edge
in each, write deletions back. Uniqueness constraints on `Concept.name`,
`Clip.id`, `Video.id`. No GDS plugins, no writes at query time, no rerankers, no
API — P4 wraps this.

**Done when:** the three canned queries (`"attention"`, `"backpropagation"`,
`"how does a network learn"`) return valid deterministic playlists;
`watch_seconds` < 25% of corpus on each; adding a known concept never increases
watch time; the positional-encoding gap shows up in `gaps`, not `playlist`.
