# Beeline

Beeline turns hours of course video into a prerequisite graph, then plays you the
shortest sequence of clips that takes you from what you already know to the one
concept you want to understand.

Corpus: 3Blue1Brown *Neural Networks* series — 7 videos, ~2h20m.

## Folder ownership

| Folder | Owner | Responsibility |
|---|---|---|
| `frontend/` | P1 | Player, graph visualization, known-concepts checklist, counters, kill-shot toggle |
| `ingestion/` | P2 | TwelveLabs indexing + Strands ingestion agent → `graph_payload.json` |
| `graph/` | P3 | Neo4j schema, loader, path engine (closure, set cover, ordering) |
| `integration/` | P4 | FastAPI `/api/path`, Strands Path Agent, demo |
| `shared/` | everyone | The response contract everyone codes against |
| `data/cache/` | everyone | Every external API response, cached by request hash. Committed cache = offline demo |

## The contract

`shared/types.ts` is the single source of truth for the API response shape.
`shared/types.py` is its Pydantic mirror — Python layers build responses through
it so contract drift fails loudly instead of silently.

`shared/sample_path.json` is one valid `PathResult` (query `"attention"`, two
concepts marked known). The frontend must fully work against this file with no
backend running.

Do not change the contract without telling everyone.

## Setup

```bash
cp .env.example .env   # fill in real keys; .env is gitignored, never commit it
```

## What Beeline claims

Exactly one thing: every prerequisite of your target is covered by a clip in your
path, in a valid order — and where the corpus can't cover one, you are told.
It does not certify that you understood anything.
