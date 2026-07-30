"""Strands @tool functions wrapping each ingestion step.

Tools keep bulk data in module-level RUN_STATE and return short human-readable
strings, so the agent orchestrates the pipeline without dragging chapter text
and embeddings through its context window.

Every tool's external calls go through cache.cached(), so re-running the agent
costs zero API calls.
"""

from __future__ import annotations

import json

from strands import tool

from . import build, extract, tl
from .config import CORPUS_BY_ID, PAYLOAD_PATH
from .media import have_media, probe_duration, split_video, video_path

RUN_STATE: dict = {
    "tasks": {},        # chunk_key -> task_id
    "ready": {},        # chunk_key -> {task_id, tl_video_id}
    "clips": {},        # video_id -> [clip dicts, ORIGINAL timeline]
    "extractions": [],  # [{clip_id, teaches, assumes}]
    "canon": {},        # raw -> canonical
}


def reset_state() -> None:
    RUN_STATE.update({"tasks": {}, "ready": {}, "clips": {},
                      "extractions": [], "canon": {}})


@tool
def upload_video(video_id: str) -> str:
    """Split one corpus video into ~10 minute chunks and start TwelveLabs indexing
    for every chunk concurrently. Cached: re-running costs no API calls.

    Args:
        video_id: Corpus id such as "V6" or "V5".
    """
    video_id = video_id.strip().upper()
    if video_id not in CORPUS_BY_ID:
        return f"ERROR unknown video_id {video_id}"
    if not have_media(video_id):
        return f"ERROR no downloaded media for {video_id}; skip this video"

    chunks = split_video(video_id)
    for c in chunks:
        try:
            RUN_STATE["tasks"][c.key] = tl.upload_chunk(c)
        except Exception as exc:
            return f"ERROR upload failed for {c.key}: {exc}"
    return (f"{video_id}: {len(chunks)} chunk(s) submitted for indexing "
            f"(offsets {[round(c.offset) for c in chunks]}s)")


@tool
def await_ready(video_id: str) -> str:
    """Poll TwelveLabs until every chunk of this video is indexed and ready.

    Args:
        video_id: Corpus id such as "V6".
    """
    video_id = video_id.strip().upper()
    if not have_media(video_id):
        return f"ERROR no media for {video_id}"
    chunks = split_video(video_id)
    ok = 0
    for c in chunks:
        task_id = RUN_STATE["tasks"].get(c.key)
        if not task_id:
            try:
                task_id = tl.upload_chunk(c)
                RUN_STATE["tasks"][c.key] = task_id
            except Exception as exc:
                return f"ERROR no task for {c.key}: {exc}"
        try:
            RUN_STATE["ready"][c.key] = tl.await_ready(c, task_id)
            ok += 1
        except Exception as exc:
            print(f"[await_ready] {c.key}: {exc}", flush=True)
    return f"{video_id}: {ok}/{len(chunks)} chunk(s) indexed and ready"


@tool
def get_chapters(video_id: str) -> str:
    """Get Pegasus chapters for each indexed chunk of this video and correct every
    timestamp back onto the ORIGINAL video's timeline. Falls back to ~90s
    caption-aligned windows for any chunk Pegasus cannot chapter.

    Args:
        video_id: Corpus id such as "V6".
    """
    video_id = video_id.strip().upper()
    if not have_media(video_id):
        return f"ERROR no media for {video_id}"
    try:
        clips = build.chapters_for_video(video_id, RUN_STATE["ready"])
    except AssertionError as exc:
        return f"ERROR timeline assertion failed for {video_id}: {exc}"
    RUN_STATE["clips"][video_id] = clips
    if not clips:
        return f"{video_id}: NO chapters produced"
    src = build.CHAPTER_SOURCE.get(video_id, "?")
    span = f"{clips[0]['start']:.0f}-{clips[-1]['end']:.0f}s"
    return (f"{video_id}: {len(clips)} chapters via {src}, "
            f"covering {span} of {probe_duration(video_path(video_id)):.0f}s")


@tool
def extract_concepts(video_id: str) -> str:
    """Run one structured-output OpenAI call per chapter of this video, returning
    the concepts each chapter teaches (with scores) and the concepts it assumes.

    Args:
        video_id: Corpus id such as "V6".
    """
    video_id = video_id.strip().upper()
    clips = RUN_STATE["clips"].get(video_id)
    if not clips:
        return f"ERROR call get_chapters({video_id}) first"

    done = {e["clip_id"] for e in RUN_STATE["extractions"]}
    n_t = n_a = 0
    for i, c in enumerate(clips, start=1):
        clip_id = f"{video_id.lower()}_c{i}"
        if clip_id in done:
            continue
        try:
            e = extract.extract_chapter(clip_id, c["title"], c["summary"] or c["title"])
        except Exception as exc:
            print(f"[extract] {clip_id}: {exc}", flush=True)
            continue
        RUN_STATE["extractions"].append(e)
        n_t += len(e["teaches"])
        n_a += len(e["assumes"])
    return f"{video_id}: extracted {n_t} taught and {n_a} assumed concept mentions"


@tool
def canonicalize() -> str:
    """Merge every extracted concept name into canonical nodes: embed all names,
    merge pairs above 0.85 cosine, then apply the hand-written aliases.json
    overrides last so demo-critical merges always win. Call once, after all
    videos have been extracted.
    """
    surface = set()
    for e in RUN_STATE["extractions"]:
        surface.update(t["concept"] for t in e["teaches"])
        surface.update(e["assumes"])
    if not surface:
        return "ERROR nothing extracted yet"
    try:
        RUN_STATE["canon"] = extract.canonicalize(surface)
    except Exception as exc:
        return f"ERROR canonicalize failed: {exc}"
    n_canon = len(set(RUN_STATE["canon"].values()))
    checks = {k: RUN_STATE["canon"].get(k, "MISSING")
              for k in ("attention", "softmax", "embedding", "backpropagation")}
    return (f"merged {len(surface)} raw names into {n_canon} canonical concepts; "
            f"key nodes: {json.dumps(checks)}")


@tool
def append_payload() -> str:
    """Write graph_payload.json from everything gathered so far: videos, clips,
    concepts, explains edges and requires edges. Call once at the very end.
    """
    canon = RUN_STATE["canon"]
    if not canon:
        return "ERROR call canonicalize() first"

    payload = build.assemble(RUN_STATE["clips"], RUN_STATE["extractions"], canon)
    build.write_payload(payload)
    return (f"wrote {PAYLOAD_PATH.name}: {len(payload['videos'])} videos, "
            f"{len(payload['clips'])} clips, {len(payload['concepts'])} concepts, "
            f"{len(payload['explains'])} explains, {len(payload['requires'])} requires")


ALL_TOOLS = [upload_video, await_ready, get_chapters, extract_concepts,
             canonicalize, append_payload]
