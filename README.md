# Beeline

**Educational video is linear. Knowledge is a graph.**

Beeline turns a lecture series into a prerequisite graph, then plays you the
shortest ordered sequence of clips that takes you from what you already know to
the one concept you actually want to understand — and when the corpus never
teaches a prerequisite, it goes and finds video that does.

```
Live app   https://beeline-633626894967-us-west-2.s3.us-west-2.amazonaws.com/app/index.html
Live API   https://siy3v64bjw.us-west-2.awsapprunner.com
```

Corpus: the 3Blue1Brown *Neural Networks* series — 7 videos, 2h 20m.

---

## The problem

You want to understand **attention**. It is explained somewhere inside 2h20m of
lectures. Your options today are both bad:

- Watch everything — hours of material for minutes of relevance.
- Search or skip to a chapter — and land mid-explanation in something that
  silently assumes three concepts you have not seen.

Nobody can tell you: *where is this actually taught (not just mentioned)? what
does that explanation assume? which of those can I skip? what order? and is
anything it depends on never explained at all?*

Search answers "where is the word said." Only a graph answers "what does it take
to understand this, and what am I allowed to skip."

## What it does

Type a concept. Beeline resolves it to a node in the graph, computes its full
prerequisite closure, prunes everything you tick as already known, selects the
minimum set of clips that genuinely teaches what remains, orders them so
prerequisites always play first, and plays them as one continuous session.

```
'attention', knowing nothing            13 clips  26.9m   of 2h 20m
'attention', knowing linalg + networks  10 clips  19.4m
```

Three of those clips are **not from the corpus**. Marking two concepts known
removes two of them — that is the point: the fetched material is the price of
not knowing something.

Every concept is coloured by what it means *for you*: on your path, pruned
because you know it, irrelevant to this question, or a gap the corpus never
teaches.

## What it claims

Exactly one thing: **every prerequisite of your target is covered by a clip that
teaches it, in a valid order — and where the corpus cannot cover one, you are
told, and we go looking.**

It does not grade you, test you, or certify that you understood anything.

---

## How it works

```
   7 videos ──► TwelveLabs ──► chapters ──► OpenAI ──► concepts + prerequisite
                (Marengo +                  (extract)   edges
                 Pegasus)                                    │
                                                             ▼
                                              prune: merge synonyms, drop
                                              single-assertion edges, verify
                                              each "teaches" claim
                                                             │
                                                             ▼
        query ──► Neo4j vector index ──► closure ──► prune known ──► set cover
                                                             │
                                                    ┌────────┴────────┐
                                                    ▼                 ▼
                                            corpus clips      gaps → search
                                                    │           YouTube, cut
                                                    └────────┬────────┘
                                                             ▼
                                              topologically ordered playlist
```

**TwelveLabs** finds where each concept is genuinely taught across speech,
visuals and on-screen maths, and supplies the chapter boundaries that become
clips. It also powers the Search-only comparison.

**OpenAI** decides what each chapter teaches versus assumes, judges which
concept names mean the same thing, verifies whether a chapter actually teaches a
concept or merely uses it, and writes the narration.

**Neo4j** holds the graph: `Clip -[:EXPLAINS]-> Concept`,
`Concept -[:REQUIRES]-> Concept`, plus the vector index that turns free text
into a concept.

**Strands Agents** orchestrates all three, across three agents:

| Agent | Tools |
|---|---|
| Ingestion | `upload_video`, `await_ready`, `get_chapters`, `extract_concepts`, `canonicalize`, `append_payload` |
| Path | `resolve_concept`, `build_learning_path` |
| Gap filling | `classify`, `find_candidates`, `fetch_best` |

Each agent falls back to deterministic code, so a misbehaving model costs a
retry, never a wrong clip.

## The hard parts

Most of the engineering went into problems that only appear once real data
arrives. A few worth naming, because each one produced a plausible-looking
system that was quietly wrong:

**Optimising watch time built playlists of title cards.** Watch time was both
the set cover's objective and the success metric, so the engine learned to cover
"attention" with a 31-second intro scoring 0.80 rather than the 162-second
chapter scoring 0.95 that actually explains it. It hit 6.5% of corpus and taught
nothing. Selection now treats the target and its prerequisites differently: the
thing you asked about gets the best explanation available, prerequisites get the
cheapest sufficient one, and nothing below a quality floor counts as coverage.

**"Uses X" is not "teaches X".** One clip claimed to teach linear algebra; its
own summary described *formalising a network **using** linear algebra*. No score
separates the two — a real explanation and a competent application both land
around 0.75-0.85. Every EXPLAINS claim is now checked against the chapter
summary: *would someone who did not know this learn it here?* 114 of 352 claims
were demoted.

**Ordering clips is not ordering concepts.** A clip is an indivisible bundle. A
chapter teaching both "word embeddings" and "tokenization" cannot be placed by
either one alone, and the clip-level graph turns out to contain genuine cycles.
Clips are topologically sorted directly, cycles are broken from within a
strongly connected component (breaking them on a bystander was a real bug), and
the target is pinned last so the payoff never precedes the setup.

**Similarity is not transitive.** Merging near-duplicate concept names by cosine
is impossible — `'tokens'~'tokenization'` scores 0.552 while `'vectors'~'matrices'`
scores 0.613. Embeddings now propose candidates and a model adjudicates them.
And merging must not chain: union-find over pairwise judgements once collapsed
*neural network*, *linear algebra*, *matrix multiplication*, *vectors* and
*embedding* into a single node called *gradient descent*.

**Recall is cheap at extraction, expensive later.** The raw payload has 375
concepts and 1588 prerequisite edges, 93% of them asserted by exactly one
chapter — which is where "attention requires the mnist dataset" came from.
Pruning to a confidence floor leaves 97 concepts and 184 edges.

Acceptance criteria are behavioural rather than numeric, asserted against the
real corpus in `beeline/graph/tests/test_quality.py` (85 tests). Watch time is
**reported, not targeted** — targeting it is what caused the title-card problem.

## Repository

```
beeline/
├── ingestion/     TwelveLabs indexing, extraction, pruning, clip cutting
├── graph/         Neo4j schema + loader, path engine, tests
├── integration/   FastAPI, Strands Path Agent, gap filling, TwelveLabs search
├── frontend/      React + Vite single page
├── shared/        The response contract (types.ts + Pydantic mirror)
└── data/cache/    Every external API response, committed — this is what makes
                   the demo work offline
deploy/            AWS deployment scripts and notes
```

## Running it

Credentials live in a repo-root `.env` (gitignored). See `beeline/.env.example`.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/uvicorn app:app --app-dir beeline/integration --port 8000
cd beeline/frontend && npm install && npm run dev
```

Rebuilding the data from scratch:

```bash
.venv/bin/python beeline/ingestion/fetch_media.py   # sources + captions
.venv/bin/python -m beeline.ingestion.ingest        # TwelveLabs + OpenAI
.venv/bin/python beeline/ingestion/prune.py         # merge, verify, prune
.venv/bin/python beeline/ingestion/cut.py           # 82 clips -> data/clips
.venv/bin/python beeline/graph/load.py beeline/ingestion/graph_payload.json
```

`ingest` re-runs from cache at zero API cost. **After re-pruning, reload Neo4j
and clear `data/cache/api/`** — otherwise you are looking at history, which
cost us an hour of confusion.

Deployment: see [`deploy/README.md`](deploy/README.md).

## Known limits

- Gap filling is disabled on the deployed service (`BEELINE_ALLOW_GAP_FILL=0`).
  It shells out to yt-dlp and ffmpeg — 30-90s per gap and throttled from
  datacentre IPs. The three rehearsed queries have their fills cached; others
  show gaps unfilled.
- Clips are cut from copyrighted lectures and served from a public S3 prefix.
  Fine for a hackathon demo, not a distribution model.
- One corpus, indexed ahead of time. There is no live ingestion of arbitrary
  URLs.
