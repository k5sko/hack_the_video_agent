# Beeline

Beeline turns hours of course video into a prerequisite graph, then plays you the
shortest sequence of clips that takes you from what you already know to the one
concept you want to understand.

Corpus: 3Blue1Brown *Neural Networks* series — 7 videos, ~2h20m.

## Running it

Credentials live in the **repo-root** `.env` (gitignored), not in `beeline/`.
Everything runs out of the repo-root `.venv`.

```bash
cp beeline/.env.example .env      # then fill in real keys
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
# 1. API + media server
.venv/bin/uvicorn app:app --app-dir beeline/integration --port 8000
#    BEELINE_STORE=neo4j   use the Aura graph (needed for fuzzy queries)
#    BEELINE_CANNED=1      serve only cached responses; proves the demo is offline

# 2. frontend
cd beeline/frontend && npm install && npm run dev     # http://localhost:5173
```

The frontend falls back to bundled fixtures if the API is unreachable and says so
on screen, so a dead backend degrades the demo instead of blanking it.

### Rebuilding the data

```bash
.venv/bin/python beeline/ingestion/fetch_media.py     # sources + captions (~440MB)
.venv/bin/python -m beeline.ingestion.ingest          # TwelveLabs + OpenAI -> payload
.venv/bin/python beeline/ingestion/prune.py           # merge + confidence floor
.venv/bin/python beeline/ingestion/cut.py             # 82 clips -> data/clips (~420MB)
.venv/bin/python beeline/graph/load.py beeline/ingestion/graph_payload.json
```

`ingest` is a no-op re-run from `data/cache/`. `prune` always works from
`graph_payload.raw.json`, so re-running with different thresholds is safe.

**Wipe Aura before loading a different payload.** The loader uses `MERGE`, so a
previously loaded graph is folded into the new one rather than replaced.

## Folder ownership

| Folder | Responsibility |
|---|---|
| `frontend/` | Player, graph visualization, known-concepts checklist, counters, kill-shot toggle |
| `ingestion/` | TwelveLabs indexing + Strands ingestion agent → `graph_payload.json`, pruning, clip cutting |
| `graph/` | Neo4j schema, loader, path engine (closure, set cover, ordering) |
| `integration/` | FastAPI `/api/path` + `/media`, Strands Path Agent, gap filling |
| `shared/` | The response contract everyone codes against |
| `data/cache/` | Every external API response, keyed by request hash. Committed — this is what makes the demo work offline |
| `data/media`, `data/clips` | Source video and cut clips. Not committed; regenerate with the commands above |

## The contract

`shared/types.ts` is the single source of truth for the API response shape.
`shared/types.py` is its Pydantic mirror — Python layers build responses through
it so contract drift fails loudly instead of silently.

Clips carry `media_url` (the cut file we actually play) as well as `youtube_url`
(attribution), and a `source` of `corpus` or `external`.

## Why the player uses cut files

Clips are cut into their own MP4s and served locally rather than seeked inside an
embedded YouTube player. An IFrame drags YouTube's whole interface along with it,
and — the part that matters on stage — it cannot work offline. Everything else is
already cached; streaming from YouTube would have been the one live dependency.

## What Beeline claims

Exactly one thing: every prerequisite of your target is covered by a clip that
teaches it, in a valid order — and where the corpus can't cover one, you are
told. It does not certify that you understood anything.

## Acceptance criteria

Deliberately behavioural rather than numeric — see `graph/tests/test_quality.py`,
which runs against the real corpus. Watch time is **reported, not targeted**:
when it was the goal, the set cover learned to assemble playlists out of
31-second title cards and scored 6.5% of corpus while teaching nothing.

What is asserted instead: the target is covered; every covering clip genuinely
teaches its concept rather than mentioning it; prerequisites precede dependents
except where the corpus bundles concepts into interlocking chapters; gaps are
things no clip could teach rather than things we failed to pick; and knowing more
never costs you more video.
