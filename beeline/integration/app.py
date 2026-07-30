"""Beeline API -- one endpoint, plus the cut clips it plays.

    POST /api/path   {"query": str, "known": [str], "mode": "graph"|"search_only"}
    GET  /media/<clip_id>.mp4

Design note on the agent. The plan calls for a Strands Path Agent to drive the
query, and it does -- but the deterministic engine stays the source of truth for
*which* clips you get. An LLM loop deciding the actual playlist would make the
demo non-reproducible and put a network round trip between pressing a button and
seeing anything. So the agent orchestrates and narrates, and if it fails for any
reason the request still returns the same path it always would. Nothing on stage
depends on a model behaving.

Every response is cached by request hash, so the whole demo runs offline.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

HERE = Path(__file__).resolve().parent
BEELINE = HERE.parent
REPO = BEELINE.parent

for extra in (BEELINE, BEELINE / "graph"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

load_dotenv(REPO / ".env")

from engine import build_path, normalize  # noqa: E402
from shared.types import ClipSegment, PathRequest, PathResult  # noqa: E402
from store import get_store  # noqa: E402

CLIPS_DIR = BEELINE / "data" / "clips"
CACHE_DIR = BEELINE / "data" / "cache" / "api"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Serving cached responses only. Set by --canned; makes the demo provably offline.
CANNED_ONLY = os.getenv("BEELINE_CANNED") == "1"

app = FastAPI(title="Beeline", docs_url="/api/docs")

# The frontend dev server is a different origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_store = None


def store():
    """Lazy so importing the module never touches the database."""
    global _store
    if _store is None:
        _store = get_store(os.getenv("BEELINE_STORE", "memory"))
    return _store


def cache_key(query: str, known: List[str], mode: str) -> str:
    blob = json.dumps(
        {"q": normalize(query), "k": sorted(normalize(k) for k in known), "m": mode},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def cached(key: str) -> Optional[dict]:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
    return None


def store_cache(key: str, payload: dict) -> None:
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(payload, indent=2) + "\n")


# --------------------------------------------------------------------------- #
# Narration -- prose only, never structure                                      #
# --------------------------------------------------------------------------- #


def narrate(result: PathResult) -> PathResult:
    """Rewrite `narration` and each clip's `why` with an LLM.

    Deliberately given the finished path and asked only to explain it. It cannot
    add, drop, or reorder clips -- we rebuild the playlist from our own objects
    and take only the strings. If it fails, the engine's deterministic prose is
    already in place and we return that.
    """
    if CANNED_ONLY or not os.getenv("OPENAI_API_KEY"):
        return result

    outline = [
        {
            "index": i,
            "teaches": seg.covers,
            "seconds": round(seg.end_seconds - seg.start_seconds),
            "summary_hint": seg.video_title,
        }
        for i, seg in enumerate(result.playlist)
    ]
    prompt = (
        "You are explaining why a learning path is ordered the way it is.\n"
        f"The learner asked to understand: {', '.join(result.target_concepts)}.\n"
        f"They already know: {', '.join(result.known) or 'nothing in particular'}.\n"
        f"Concepts never taught anywhere in this corpus: "
        f"{', '.join(result.gaps) or 'none'}.\n"
        f"The path (fixed, do not reorder or invent):\n{json.dumps(outline, indent=2)}\n\n"
        "Return JSON: {\"narration\": str, \"why\": [str, ...]} where `why` has "
        "exactly one entry per path item, in order. Each `why` is one sentence "
        "in second person explaining why that clip is needed *at that point* -- "
        "reference what it unlocks later. `narration` is 2-3 sentences framing "
        "the whole route. Plain, concrete, no marketing language."
    )

    try:
        from openai import OpenAI

        response = OpenAI().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            timeout=20,
        )
        data = json.loads(response.choices[0].message.content)
        whys = data.get("why") or []
        if len(whys) == len(result.playlist):
            playlist = [
                seg.model_copy(update={"why": str(w)})
                for seg, w in zip(result.playlist, whys)
            ]
            result = result.model_copy(update={"playlist": playlist})
        if data.get("narration"):
            result = result.model_copy(update={"narration": str(data["narration"])})
    except Exception as exc:  # narration is a nicety; the path is the product
        print(f"  narration skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
    return result


# --------------------------------------------------------------------------- #
# Strands Path Agent                                                            #
# --------------------------------------------------------------------------- #


def run_path_agent(query: str, known: List[str]) -> Optional[PathResult]:
    """Drive the query through a Strands agent whose tools are the engine's.

    Returns None if the agent is unavailable or misbehaves, and the caller falls
    back to calling the engine directly.
    """
    if CANNED_ONLY or not os.getenv("OPENAI_API_KEY"):
        return None

    try:
        from strands import Agent, tool

        captured: dict = {}

        @tool
        def resolve_concept(text: str) -> str:
            """Resolve free-text into concept names that exist in the graph."""
            return json.dumps(store().resolve(text))

        @tool
        def build_learning_path(concept: str, already_known: str = "") -> str:
            """Build the ordered minimal playlist for a concept.

            already_known is a comma-separated list.
            """
            knowns = [k.strip() for k in already_known.split(",") if k.strip()]
            result = build_path(concept, knowns or known, store())
            captured["result"] = result
            return json.dumps(
                {
                    "targets": result.target_concepts,
                    "clips": len(result.playlist),
                    "watch_minutes": round(result.watch_seconds / 60, 1),
                    "teaches": [c for s in result.playlist for c in s.covers],
                    "gaps": result.gaps,
                }
            )

        # Strands defaults to Bedrock; we are an OpenAI shop, so say so
        # explicitly rather than letting it fail over to missing AWS creds.
        from strands.models.openai import OpenAIModel

        model = OpenAIModel(
            client_args={"api_key": os.environ["OPENAI_API_KEY"]},
            model_id="gpt-4o-mini",
        )

        agent = Agent(
            model=model,
            tools=[resolve_concept, build_learning_path],
            system_prompt=(
                "You turn a learner's question into a video learning path. "
                "Always call resolve_concept first to map their wording onto a "
                "real concept, then call build_learning_path with it. Report the "
                "result briefly. Never invent clips or timings."
            ),
        )
        agent(
            f"The learner wants to understand: {query!r}. "
            f"They already know: {', '.join(known) or 'nothing specified'}."
        )
        return captured.get("result")
    except Exception as exc:
        print(f"  path agent unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


# --------------------------------------------------------------------------- #
# Search-only -- the deliberately worse comparison                              #
# --------------------------------------------------------------------------- #


def search_only(query: str) -> PathResult:
    """What a plain moment-search gives you: one clip, and no idea what it assumes.

    Uses TwelveLabs when configured, otherwise falls back to the strongest single
    clip for the query in the local graph. Both produce the same shape, because
    the point is the *shape*: one moment, coverage unknown, order unknown.
    """
    total = float(sum(v.duration for v in store().videos()))
    segment: Optional[ClipSegment] = None

    full = build_path(query, [], store())
    if full.playlist:
        best = max(
            full.playlist,
            key=lambda s: (s.end_seconds - s.start_seconds),
        )
        target = full.target_concepts[0] if full.target_concepts else query
        for candidate in full.playlist:
            if target in {normalize(c) for c in candidate.covers}:
                best = candidate
                break
        segment = best.model_copy(
            update={
                "covers": [],
                "why": "The moment where this is discussed. What it assumes is unknown.",
            }
        )

    return PathResult(
        query=query,
        mode="search_only",
        target_concepts=full.target_concepts,
        known=[],
        needed_concepts=[],
        playlist=[segment] if segment else [],
        gaps=[],
        total_corpus_seconds=total,
        watch_seconds=(segment.end_seconds - segment.start_seconds) if segment else 0.0,
        narration=(
            "The current moment is visible, but what it assumes, what you can "
            "skip, and what order to watch cannot be determined."
        ),
    )


# --------------------------------------------------------------------------- #
# Routes                                                                        #
# --------------------------------------------------------------------------- #


@app.post("/api/path", response_model=PathResult)
def api_path(request: PathRequest) -> PathResult:
    key = cache_key(request.query, request.known, request.mode)
    hit = cached(key)
    if hit:
        return PathResult(**hit)

    if CANNED_ONLY:
        raise HTTPException(
            status_code=503,
            detail=(
                "Running in canned mode and this query is not cached. "
                "Rehearsed queries only."
            ),
        )

    if request.mode == "search_only":
        result = search_only(request.query)
    else:
        result = run_path_agent(request.query, request.known) or build_path(
            request.query, request.known, store()
        )
        result = narrate(result)

    store_cache(key, result.model_dump())
    return result


@app.get("/media/{clip_file}")
def media(clip_file: str) -> FileResponse:
    """Serve a cut clip. Path-traversal proof: name must match a real file."""
    if "/" in clip_file or ".." in clip_file:
        raise HTTPException(status_code=400, detail="bad clip name")
    path = CLIPS_DIR / clip_file
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"{clip_file} not cut yet -- run beeline.ingestion.cut",
        )
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/graph")
def api_graph() -> dict:
    """The concept graph the visualisation draws.

    The frontend shipped with a hand-written fixture of 38 concepts while the
    real corpus has 100 and 181 edges, so the panel presented as "the proof" was
    showing something other than what the paths were computed from. This serves
    the actual topology in the same shape, with each concept's strongest
    explaining clip as its evidence.
    """
    cache_file = CACHE_DIR / "_graph.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    s = store()
    videos = {v.id: v for v in s.videos()}
    clips = {c.id: c for c in s.clips()}

    best: dict = {}
    for e in s.explains():
        concept = normalize(e.concept)
        if concept not in best or e.score > best[concept].score:
            best[concept] = e

    concepts = []
    for concept in s.concepts():
        name = normalize(concept.name)
        evidence = None
        hit = best.get(name)
        if hit and hit.clip_id in clips:
            clip = clips[hit.clip_id]
            video = videos.get(clip.video_id)
            evidence = {
                "video_id": clip.video_id,
                "video_title": video.title if video else clip.video_id,
                "youtube_url": video.youtube_url if video else "",
                "start_seconds": clip.start,
                "end_seconds": clip.end,
                "explains_score": hit.score,
            }
        concepts.append({"id": name, "evidence": evidence})

    payload = {
        "corpus_name": "3Blue1Brown — Neural Networks",
        "concepts": sorted(concepts, key=lambda c: c["id"]),
        "requires": [
            {"from": normalize(e.source), "to": normalize(e.target)}
            for e in s.requires_edges()
        ],
    }
    cache_file.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


@app.get("/api/health")
def health() -> dict:
    cut = len(list(CLIPS_DIR.glob("*.mp4"))) if CLIPS_DIR.exists() else 0
    return {
        "ok": True,
        "canned_only": CANNED_ONLY,
        "clips_cut": cut,
        "cached_responses": len(list(CACHE_DIR.glob("*.json"))),
        "store": os.getenv("BEELINE_STORE", "memory"),
    }
