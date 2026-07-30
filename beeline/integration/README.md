# integration/ — P4

**Owner:** Person 4

**Responsibility:** full-system integration — the FastAPI service, the Strands
Path Agent that wraps P3's functions, and the 90-second demo.

**Expected input:** P3's path engine (importable), P2's `graph_payload.json`,
P1's frontend build.

**Expected output:** one endpoint —

```
POST /api/path   { "query": str, "known": [str], "mode": "graph" | "search_only" }
```

returning JSON matching `shared/types.ts` exactly.

- `mode: "graph"` — run a Strands `Agent` whose tools are P3's `resolve_concept`,
  `prereq_closure`, `select_clips`, plus a narration step: one OpenAI call that
  writes `narration` and the per-clip `why` strings **from the computed path**.
  The agent never invents clips and never reorders them.
- `mode: "search_only"` — call TwelveLabs search with the raw query, return the
  top moment as a one-clip `PathResult` with empty `needed_concepts`, the degraded
  narration, and no gaps analysis.

**How to run:**

```bash
cd integration
uvicorn app:app --reload --port 8000
uvicorn app:app --port 8000 --canned    # serve the three rehearsed queries from disk, zero live calls
```

**Constraints:** cache every response. The demo must run start-to-finish from
cache with the network cable pulled. Do not redesign the UI, add features, or
change the shared JSON contract.

**Done when:** one button runs the live investigation end-to-end; a checkbox
shrinks the path without a reload; Search-only visibly degrades the answer; the
whole demo runs offline.
