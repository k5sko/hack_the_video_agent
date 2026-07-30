"""Find outside material for concepts this corpus never teaches.

A gap is a concept your target genuinely requires that no clip in the indexed
lectures explains. Surfacing it honestly is already better than the usual
silence, but it still leaves the learner stuck -- the whole promise is a route
from where you are to what you want, and a route with a hole in it is not a
route.

So: search YouTube for the missing concept, read the candidates' own chapter
markers, and let a model pick the chapter that teaches it. yt-dlp does both the
search and the chapter extraction, so this needs no additional API key.

The result is downloaded and cut exactly like a corpus clip, which matters more
than it sounds: a patched gap plays in the same player, at the same size, with
the same controls. It reads as part of the path rather than as a link that
ejects you into a browser tab.

Everything is cached by concept name, so a rehearsed demo makes no live calls.

    python -m beeline.integration.gapfill "positional encoding"
    python -m beeline.integration.gapfill --from-path attention
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
BEELINE = HERE.parent
REPO = BEELINE.parent

for extra in (BEELINE, BEELINE / "graph"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

load_dotenv(REPO / ".env")

from shared.types import ClipSegment  # noqa: E402

CACHE = BEELINE / "data" / "cache" / "gapfill"
CLIPS = BEELINE / "data" / "clips"
CACHE.mkdir(parents=True, exist_ok=True)
CLIPS.mkdir(parents=True, exist_ok=True)

# The corpus author. Material from the same source is the best possible fill:
# same voice, same notation, same visual language as everything around it.
CORPUS_AUTHOR = "3blue1brown"

SEARCH_COUNT = 5
# Long enough to teach something, short enough to stay a detour rather than a
# second lecture.
MIN_SECONDS = 60
MAX_SECONDS = 420


def _yt_dlp() -> str:
    """Locate yt-dlp.

    It is a venv-installed console script, so it is only on PATH when the venv
    is activated -- and this module is routinely imported by a server that was
    started some other way. Look next to the running interpreter first.
    """
    beside = Path(sys.executable).parent / "yt-dlp"
    if beside.exists():
        return str(beside)
    found = shutil.which("yt-dlp")
    if found:
        return found
    raise FileNotFoundError(
        "yt-dlp not found; install it into the venv (pip install yt-dlp)"
    )


def _cache_path(concept: str) -> Path:
    slug = "".join(c if c.isalnum() else "_" for c in concept.lower()).strip("_")
    return CACHE / f"{slug}.json"


def _kind_cache() -> Path:
    return CACHE / "_gap_kinds.json"


def classify_gap(concept: str) -> str:
    """Is this gap a concept a clip can teach, or a subject the series assumes?

    Not every hole is patchable, and treating them alike produces nonsense. Given
    the gap 'linear algebra', a search happily returns a good 3.5-minute chapter
    on linear combinations and span -- but that is *a* linear algebra topic, not
    linear algebra. Dropping it into the path pretends a subject has been covered
    in three minutes, which is the same category error as a clip that merely
    mentions the thing it claims to teach.

    So:
      "concept"    -- a specific idea one clip can genuinely teach
                      (positional encoding, dot product, softmax) -> fill it
      "background" -- a subject the corpus assumes wholesale
                      (linear algebra, calculus, machine learning) -> say so

    3Blue1Brown states outright that this series assumes linear algebra. Telling
    a learner that, and pointing at the whole series that covers it, is honest
    and useful. Splicing in one chapter of it is neither.
    """
    cache_file = _kind_cache()
    kinds: Dict[str, str] = {}
    if cache_file.exists():
        try:
            kinds = json.loads(cache_file.read_text())
        except json.JSONDecodeError:
            kinds = {}
    if concept in kinds:
        return kinds[concept]
    if not os.getenv("OPENAI_API_KEY"):
        return "concept"

    try:
        from openai import OpenAI

        response = OpenAI().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Term: {concept!r}, appearing as a prerequisite in a "
                        f"neural-network lecture series.\n\n"
                        'Answer "concept" if a single focused video segment of a '
                        "few minutes could genuinely teach it to someone who does "
                        "not know it (e.g. positional encoding, dot product, "
                        "softmax, tokenization).\n"
                        'Answer "background" if it names a whole field or course '
                        "that no single segment could teach, and which a series "
                        "would simply assume (e.g. linear algebra, calculus, "
                        "probability, machine learning).\n\n"
                        'Return JSON {"kind": "concept"|"background"}.'
                    ),
                }
            ],
            response_format={"type": "json_object"},
            timeout=30,
        )
        kind = json.loads(response.choices[0].message.content).get("kind")
        kind = kind if kind in ("concept", "background") else "concept"
    except Exception as exc:
        print(f"  gap classification failed ({exc}); assuming concept", file=sys.stderr)
        kind = "concept"

    kinds[concept] = kind
    cache_file.write_text(json.dumps(kinds, indent=1, sort_keys=True))
    return kind


def search_candidates(concept: str) -> List[dict]:
    """Search YouTube, returning each hit with whatever chapters it declares.

    Two queries, not one. A bare "<concept> explained" search returns whatever is
    optimised for search -- the first run for 'linear algebra' produced "Give Me
    30 min, I will make Linear Algebra Click Forever", which is a poor neighbour
    for a 3Blue1Brown path. But the corpus author very often covers the gap
    elsewhere: 3Blue1Brown has a whole Essence of Linear Algebra series. Material
    from the same author is the best possible fill, because the voice, notation
    and visual language already match everything around it.
    """
    seen: Dict[str, dict] = {}
    for query in (
        f"ytsearch{SEARCH_COUNT}:{CORPUS_AUTHOR} {concept}",
        f"ytsearch{SEARCH_COUNT}:{concept} explained",
    ):
        for video in _search_one(query):
            if video["id"] and video["id"] not in seen:
                seen[video["id"]] = video
    return list(seen.values())


def _search_one(query: str) -> List[dict]:
    proc = subprocess.run(
        [_yt_dlp(), query, "--dump-json", "--no-warnings", "--skip-download"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        print(f"  search failed: {proc.stderr.strip()[:200]}", file=sys.stderr)
        return []

    out = []
    for line in proc.stdout.splitlines():
        try:
            video = json.loads(line)
        except json.JSONDecodeError:
            continue
        out.append(
            {
                "id": video.get("id"),
                "title": video.get("title", ""),
                "duration": float(video.get("duration") or 0),
                "channel": video.get("channel") or video.get("uploader") or "",
                "chapters": [
                    {
                        "start": float(c.get("start_time", 0)),
                        "end": float(c.get("end_time", 0)),
                        "title": c.get("title", ""),
                    }
                    for c in (video.get("chapters") or [])
                ],
            }
        )
    return out


def choose_segment(concept: str, candidates: List[dict]) -> Optional[dict]:
    """Pick the single best (video, time range) that teaches ``concept``.

    Prefers a declared chapter, because a chapter boundary is the author's own
    statement about where a topic starts and ends -- far better than guessing an
    offset. Falls back to the opening minutes of the most plausible short video.
    """
    options = []
    for video in candidates:
        if not video["id"]:
            continue
        for chapter in video["chapters"]:
            length = chapter["end"] - chapter["start"]
            if MIN_SECONDS <= length <= MAX_SECONDS:
                options.append(
                    {
                        "video_id": video["id"],
                        "video_title": video["title"],
                        "channel": video["channel"],
                        "start": chapter["start"],
                        "end": chapter["end"],
                        "label": chapter["title"],
                    }
                )
        if not video["chapters"] and MIN_SECONDS <= video["duration"] <= MAX_SECONDS:
            options.append(
                {
                    "video_id": video["id"],
                    "video_title": video["title"],
                    "channel": video["channel"],
                    "start": 0.0,
                    "end": video["duration"],
                    "label": video["title"],
                }
            )

    if not options:
        return None
    if not os.getenv("OPENAI_API_KEY"):
        return options[0]

    try:
        from openai import OpenAI

        listing = [
            {
                "index": i,
                "video_title": o["video_title"],
                "channel": o["channel"],
                "section": o["label"],
                "seconds": round(o["end"] - o["start"]),
            }
            for i, o in enumerate(options)
        ]
        response = OpenAI().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"A learner needs to understand {concept!r}. Which one "
                        f"of these video sections best *teaches* it (not merely "
                        f"mentions it)?\n\n"
                        f"This fills a gap in a {CORPUS_AUTHOR} lecture path, so "
                        f"strongly prefer {CORPUS_AUTHOR}'s own material when it "
                        f"covers the concept -- the notation and visual language "
                        f"will already match. Otherwise prefer a focused "
                        f"explanation from a credible educational channel over "
                        f"anything with a clickbait title.\n"
                        f"{json.dumps(listing, indent=2)}\n\n"
                        'Return JSON: {"index": int, "why": str} where `why` is one '
                        "sentence, second person, saying what it gives the learner."
                    ),
                }
            ],
            response_format={"type": "json_object"},
            timeout=25,
        )
        data = json.loads(response.choices[0].message.content)
        choice = options[int(data["index"]) % len(options)]
        choice["why"] = str(data.get("why") or "")
        return choice
    except Exception as exc:
        print(f"  chooser failed ({type(exc).__name__}); taking first", file=sys.stderr)
        return options[0]


def fetch_and_cut(concept: str, choice: dict) -> Optional[str]:
    """Download the chosen video and cut the chosen range. Returns the clip id."""
    clip_id = f"gap_{_cache_path(concept).stem}"
    dest = CLIPS / f"{clip_id}.mp4"
    if dest.exists() and dest.stat().st_size > 0:
        return clip_id

    url = f"https://www.youtube.com/watch?v={choice['video_id']}"
    scratch = CLIPS / f"{clip_id}.src.mp4"
    download = subprocess.run(
        [
            _yt_dlp(),
            url,
            "-f",
            "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b",
            "--merge-output-format",
            "mp4",
            "-o",
            str(scratch),
            "--no-warnings",
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if download.returncode != 0 or not scratch.exists():
        print(f"  download failed: {download.stderr.strip()[:200]}", file=sys.stderr)
        return None

    cut = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
            "-ss", str(choice["start"]),
            "-i", str(scratch),
            "-t", str(choice["end"] - choice["start"]),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
            "-vf", "scale=-2:720",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(dest),
        ],
        capture_output=True,
        text=True,
    )
    scratch.unlink(missing_ok=True)
    if cut.returncode != 0:
        print(f"  cut failed: {cut.stderr.strip()[:200]}", file=sys.stderr)
        return None
    return clip_id


def fill_gap(concept: str, refresh: bool = False) -> dict:
    """Resolve a gap: a playable clip for a concept, or a pointer for a subject.

    Returns {"concept", "kind", "segment", "resource", "note"}. ``segment`` is
    only ever set for a "concept" gap. A "background" gap deliberately gets no
    clip -- see :func:`classify_gap` for why splicing three minutes of a subject
    into the path is worse than saying the series assumes it.
    """
    cache_file = _cache_path(concept)
    if cache_file.exists() and not refresh:
        cached = json.loads(cache_file.read_text())
        if cached.get("kind"):
            segment = cached.get("segment")
            if not segment or (CLIPS / f"{ClipSegment(**segment).clip_id}.mp4").exists():
                return cached

    kind = classify_gap(concept)
    print(f"  searching for {concept!r} ({kind})...")
    candidates = search_candidates(concept)

    if kind == "background":
        # Point at the whole thing, do not excerpt it. The most-viewed match from
        # the corpus author (or failing that, anyone) is the resource to name.
        # Prefer the corpus author's own treatment of the subject -- for this
        # corpus that is 3Blue1Brown's Essence of Linear Algebra, which is both
        # better and a far more natural continuation than whatever a generic
        # search surfaces. Longest match as a tiebreak: a subject needs a series,
        # not a short.
        def resource_rank(video: dict) -> tuple:
            same_author = CORPUS_AUTHOR.lower() in (video["channel"] or "").lower()
            return (same_author, video["duration"])

        best = max(candidates, key=resource_rank, default=None) if candidates else None
        record = {
            "concept": concept,
            "kind": "background",
            "segment": None,
            "resource": (
                {
                    "title": best["title"],
                    "channel": best["channel"],
                    "url": f"https://www.youtube.com/watch?v={best['id']}",
                }
                if best
                else None
            ),
            "note": (
                f"This series assumes you already know {concept}. It is a subject, "
                f"not a single idea, so Beeline will not pretend a few minutes of "
                f"video covers it."
            ),
        }
        cache_file.write_text(json.dumps(record, indent=2) + "\n")
        return record

    if not candidates:
        return {
            "concept": concept,
            "kind": kind,
            "segment": None,
            "resource": None,
            "note": f"Nothing found online that teaches {concept}.",
        }

    choice = choose_segment(concept, candidates)
    clip_id = fetch_and_cut(concept, choice) if choice else None
    if not choice or not clip_id:
        return {
            "concept": concept,
            "kind": kind,
            "segment": None,
            "resource": None,
            "note": f"No usable section found that teaches {concept}.",
        }

    segment = ClipSegment(
        clip_id=clip_id,
        video_id=choice["video_id"],
        video_title=choice["video_title"],
        youtube_url=f"https://www.youtube.com/watch?v={choice['video_id']}",
        media_url=f"/media/{clip_id}.mp4",
        start_seconds=0.0,  # the cut file starts at zero
        end_seconds=float(choice["end"] - choice["start"]),
        covers=[concept],
        why=(
            choice.get("why")
            or f"This corpus never teaches {concept}, so this fills it in from "
            f"{choice['channel'] or 'another lecture'}."
        ),
        source="external",
    )
    record = {
        "concept": concept,
        "kind": "concept",
        "segment": segment.model_dump(),
        "resource": {
            "title": choice["video_title"],
            "channel": choice["channel"],
            "url": f"https://www.youtube.com/watch?v={choice['video_id']}",
        },
        "note": f"{concept} is never taught in this corpus; this fills it in.",
    }
    cache_file.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("concepts", nargs="*", help="concept names to fill")
    ap.add_argument("--refresh", action="store_true", help="ignore cache")
    args = ap.parse_args()

    if not args.concepts:
        ap.error("name at least one concept")

    failed = 0
    for concept in args.concepts:
        record = fill_gap(concept, args.refresh)
        if record["kind"] == "background":
            resource = record.get("resource") or {}
            print(f"  {concept!r} -> assumed background; see {resource.get('title', 'n/a')[:55]}")
        elif record.get("segment"):
            seg = record["segment"]
            print(
                f"  {concept!r} -> {seg['clip_id']} "
                f"({seg['end_seconds']:.0f}s) {seg['video_title'][:55]}"
            )
        else:
            failed += 1
            print(f"  {concept!r} -> {record['note']}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
