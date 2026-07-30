"""Assemble graph_payload.json.

Chapters come from Pegasus per chunk; if a chunk has no usable Pegasus chapters
we fall back to ~90s caption-aligned windows for that chunk only. Either way
every timestamp is translated back to the ORIGINAL video timeline and asserted.
"""

from __future__ import annotations

import json
from collections import defaultdict

from . import extract, tl
from .config import (CORPUS, CORPUS_BY_ID, PAYLOAD_PATH, youtube_url)
from .media import Chunk, caption_windows, have_media, probe_duration, split_video, video_path

# Filled in by the run so the report can state Pegasus vs fallback honestly.
CHAPTER_SOURCE: dict[str, str] = {}


MIN_CLIP_SECONDS = 30.0


def _valid_chapters(raw: list[dict], chunk: Chunk) -> list[dict]:
    """Normalize Pegasus chapters within the chunk's own local timeline.

    Pegasus returns finer-grained chapters than asked (often 10-30s). Dropping the
    short ones would punch holes in coverage, so we MERGE consecutive short
    chapters until each clip is a useful length for a playlist.
    """
    parsed = []
    for ch in raw or []:
        try:
            s = float(ch["start_time"])
            e = float(ch["end_time"])
        except (KeyError, TypeError, ValueError):
            continue
        # Pegasus occasionally emits times past the clip end; clamp rather than drop.
        s = max(0.0, min(s, chunk.duration))
        e = max(0.0, min(e, chunk.duration))
        if e <= s:
            continue
        parsed.append({
            "start": s, "end": e,
            "title": (ch.get("title") or "").strip(),
            "summary": (ch.get("summary") or "").strip(),
        })
    parsed.sort(key=lambda c: c["start"])

    merged: list[dict] = []
    for ch in parsed:
        if merged and (merged[-1]["end"] - merged[-1]["start"]) < MIN_CLIP_SECONDS:
            prev = merged[-1]
            prev["end"] = max(prev["end"], ch["end"])
            prev["title"] = prev["title"] or ch["title"]
            prev["summary"] = (prev["summary"] + " " + ch["summary"]).strip()
        else:
            merged.append(dict(ch))

    # A trailing stub still shorter than the minimum folds back into its predecessor.
    if len(merged) > 1 and (merged[-1]["end"] - merged[-1]["start"]) < MIN_CLIP_SECONDS:
        tail = merged.pop()
        merged[-1]["end"] = max(merged[-1]["end"], tail["end"])
        merged[-1]["summary"] = (merged[-1]["summary"] + " " + tail["summary"]).strip()

    return [c for c in merged if c["end"] > c["start"]]


def _fallback_chapters(chunk: Chunk) -> list[dict]:
    """Caption-aligned ~90s windows restricted to this chunk's span."""
    lo, hi = chunk.offset, chunk.offset + chunk.duration
    wins = []
    for w in caption_windows(chunk.video_id):
        # caption_windows are already in the ORIGINAL timeline
        if w["end"] <= lo or w["start"] >= hi:
            continue
        s = max(w["start"], lo) - chunk.offset
        e = min(w["end"], hi) - chunk.offset
        if e - s < 15.0:
            continue
        body = w["text"]
        wins.append({"start": s, "end": e, "title": "",
                     "summary": body[:900]})
    return wins


def chapters_for_video(video_id: str, ready: dict[str, dict]) -> list[dict]:
    """Return clips for one video in the ORIGINAL timeline."""
    chunks = split_video(video_id)
    clips: list[dict] = []
    total = probe_duration(video_path(video_id))
    used_pegasus = used_fallback = 0

    for chunk in chunks:
        info = ready.get(chunk.key) or {}
        local: list[dict] = []
        if info.get("tl_video_id"):
            try:
                local = _valid_chapters(tl.get_chapters(chunk, info["tl_video_id"]), chunk)
            except Exception as exc:
                print(f"[chapters] {chunk.key} pegasus failed: {exc}", flush=True)
                local = []
        if local:
            used_pegasus += 1
        else:
            local = _fallback_chapters(chunk)
            if local:
                used_fallback += 1

        for c in local:
            # ---- TIMELINE CORRECTION (the fatal-if-wrong step) ----
            start = chunk.offset + c["start"]
            end = chunk.offset + c["end"]
            assert start >= -0.01, f"{chunk.key}: negative start {start}"
            assert end <= total + 5.0, (
                f"{chunk.key}: end {end:.1f}s exceeds video duration {total:.1f}s "
                f"(offset={chunk.offset:.1f}, local_end={c['end']:.1f})"
            )
            assert end > start, f"{chunk.key}: end <= start"
            clips.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "title": c["title"],
                "summary": c["summary"],
            })

    clips.sort(key=lambda c: c["start"])
    if used_pegasus and not used_fallback:
        CHAPTER_SOURCE[video_id] = "pegasus"
    elif used_pegasus and used_fallback:
        CHAPTER_SOURCE[video_id] = "mixed"
    elif used_fallback:
        CHAPTER_SOURCE[video_id] = "captions"
    else:
        CHAPTER_SOURCE[video_id] = "none"
    return clips


def assemble(clips_by_video: dict[str, list[dict]],
             extractions: list[dict],
             canon: dict[str, str]) -> dict:
    """Single source of truth for payload shape.

    Both the agent path (tools.append_payload) and the offline rebuild path
    (build_payload) call this, so the two can never drift and P3 gets an
    identical file either way.
    """
    videos: list[dict] = []
    clips: list[dict] = []

    # Emit videos in CORPUS (demo) order regardless of processing order.
    for v in CORPUS:
        vid = v["id"]
        vclips = clips_by_video.get(vid)
        if not vclips:
            continue
        videos.append({
            "id": vid,
            "title": v["title"],
            "youtube_url": youtube_url(v["youtube_id"]),
            "duration": int(round(probe_duration(video_path(vid)))),
        })
        for i, c in enumerate(vclips, start=1):
            clips.append({
                "id": f"{vid.lower()}_c{i}",
                "video_id": vid,
                "start": int(round(c["start"])),
                "end": int(round(c["end"])),
                "summary": c["summary"] or c["title"],
            })

    alias_sets: dict[str, set[str]] = defaultdict(set)
    for raw, c in canon.items():
        if raw != c:
            alias_sets[c].add(raw)

    explains: dict[tuple[str, str], float] = {}
    requires_count: dict[tuple[str, str], int] = defaultdict(int)

    for e in extractions:
        taught: dict[str, float] = {}
        for t in e["teaches"]:
            c = canon.get(t["concept"], t["concept"])
            taught[c] = max(taught.get(c, 0.0), float(t["score"]))
        for c, score in taught.items():
            key = (e["clip_id"], c)
            explains[key] = max(explains.get(key, 0.0), score)

        # A chapter cannot both teach and assume the same thing.
        assumed = {canon.get(a, a) for a in e["assumes"]} - set(taught)
        for c in taught:
            for p in assumed:
                if c != p:
                    requires_count[(c, p)] += 1

    names = set(canon.values()) | {c for _, c in explains}
    for a, b in requires_count:
        names.update((a, b))

    return {
        "videos": videos,
        "clips": clips,
        "concepts": [{"name": n, "aliases": sorted(alias_sets.get(n, set()))}
                     for n in sorted(names)],
        "explains": [{"clip_id": cid, "concept": c, "score": round(s, 2)}
                     for (cid, c), s in sorted(explains.items())],
        "requires": [{"from": a, "to": b, "confidence": n}
                     for (a, b), n in sorted(requires_count.items())],
    }


def build_payload(ready: dict[str, dict]) -> dict:
    """Non-agent rebuild: chapters -> extraction -> canonicalize -> assemble."""
    clips_by_video: dict[str, list[dict]] = {}
    for v in CORPUS:
        vid = v["id"]
        if not have_media(vid):
            continue
        raw_clips = chapters_for_video(vid, ready)
        if raw_clips:
            clips_by_video[vid] = raw_clips

    extractions = []
    for vid, vclips in clips_by_video.items():
        for i, c in enumerate(vclips, start=1):
            clip_id = f"{vid.lower()}_c{i}"
            try:
                extractions.append(
                    extract.extract_chapter(clip_id, c["title"], c["summary"] or c["title"]))
            except Exception as exc:
                print(f"[extract] {clip_id} failed: {exc}", flush=True)

    surface = set()
    for e in extractions:
        surface.update(t["concept"] for t in e["teaches"])
        surface.update(e["assumes"])
    canon = extract.canonicalize(surface)

    return assemble(clips_by_video, extractions, canon)


def write_payload(payload: dict) -> None:
    PAYLOAD_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
