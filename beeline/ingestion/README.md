# ingestion/ — P2

**Owner:** Person 2

**Responsibility:** TwelveLabs indexing (Marengo search + Pegasus chapters) and
the Strands ingestion agent that drives
`index → chapter → extract → canonicalize → load` for each video.

**Expected input:** the 7 corpus videos (3Blue1Brown *Neural Networks*), a
TwelveLabs index, `OPENAI_API_KEY`, and `aliases.json` for demo-critical concept
merges.

**Expected output:** one file, `graph_payload.json`, covering at least 5 of the 7
videos:

```json
{
  "videos":   [{"id": "V6", "title": "...", "youtube_url": "...", "duration": 1560}],
  "clips":    [{"id": "v6_c2", "video_id": "V6", "start": 220, "end": 535, "summary": "..."}],
  "concepts": [{"name": "attention", "aliases": ["self-attention"]}],
  "explains": [{"clip_id": "v6_c2", "concept": "attention", "score": 0.94}],
  "requires": [{"from": "attention", "to": "softmax", "confidence": 3}]
}
```

**How to run:** (all commands from the REPO ROOT, using the repo venv)

```bash
# 0. one-time: fetch the corpus + captions into data/media/ (not committed)
.venv/bin/python -m beeline.ingestion.fetch_media

# 1. optional but recommended: index every chunk concurrently (wall-clock bound)
.venv/bin/python -m beeline.ingestion.upload_stage

# 2. full agent run
.venv/bin/python -m beeline.ingestion.ingest

# 3. rebuild the payload from cache with ZERO API calls (this is the demo path)
.venv/bin/python -m beeline.ingestion.ingest --from-cache
```

Credentials load from the **repo root** `.env` (gitignored): `TWELVELABS_API_KEY`,
`TWELVELABS_INDEX_ID`, `OPENAI_API_KEY`. The index must carry **both** a Marengo
model (search) and a Pegasus model (chapter generation); `upload_stage` assumes
`TWELVELABS_INDEX_ID` already points at such an index.

**Constraints:** every tool caches its API response to `data/cache/` keyed by
request hash — a re-run must cost zero API calls. Videos over ~20 min are split
into ~10-min chunks with `ffmpeg -c copy` and uploaded concurrently; every emitted
timestamp is corrected back to the original video's timeline. One Strands `Agent`
with `@tool` functions — no swarms, no A2A, no deployment infra. No UI, no vector
DB, no fine-tuning.

**Done when:** ≥5 videos, ≥50 concepts, both edge types; `attention`, `softmax`,
`embedding`, `backpropagation` each exist as a single canonical node; a re-run
costs zero API calls; P3 can load the file without asking any questions.
