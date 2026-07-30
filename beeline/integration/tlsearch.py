"""Real TwelveLabs moment search, for the Search-only comparison.

The kill-shot toggle is only worth showing if the losing side is genuine. A
simulated "search" that secretly consults the prerequisite graph to find its one
clip would be a strawman, and the honest answer to "is that real search?" has to
be yes.

Two wrinkles make this more than one API call.

Long videos were uploaded to TwelveLabs in ~10 minute chunks, so a hit comes back
with a chunk's id and a timestamp inside that chunk. Both have to be mapped back
to the original lecture's timeline or the player seeks to the wrong place --
invisible in JSON, glaring on stage. The mapping is reconstructed from the upload
cache (chunk -> TwelveLabs id) and the ffmpeg segment lists (chunk -> real
offset).

And a raw hit is a few seconds long, which is not watchable. We widen it to a
minimum window around the moment, which is what a "jump to this moment" player
would do anyway.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests

HERE = Path(__file__).resolve().parent
BEELINE = HERE.parent
MEDIA = BEELINE / "data" / "media"
CLIPS = BEELINE / "data" / "clips"
UPLOAD_CACHE = BEELINE / "data" / "cache" / "tl_upload"

API = "https://api.twelvelabs.io/v1.3"

# A 4-second hit is not something you can watch. Widen it the way any
# jump-to-moment player would.
MIN_WINDOW = 75.0


@lru_cache(maxsize=1)
def video_map() -> Dict[str, Tuple[str, float]]:
    """TwelveLabs video id -> (our video id, offset into the original video).

    Videos short enough to upload whole have a single entry at offset 0.
    """
    offsets: Dict[Tuple[str, int], float] = {}
    for segments in MEDIA.glob("chunks/*/segments.csv"):
        video_id = segments.parent.name
        with segments.open() as fh:
            for index, row in enumerate(csv.reader(fh)):
                if len(row) >= 2:
                    offsets[(video_id, index)] = float(row[1])

    mapping: Dict[str, Tuple[str, float]] = {}
    for entry in UPLOAD_CACHE.glob("*.json"):
        try:
            record = json.loads(entry.read_text())
        except json.JSONDecodeError:
            continue
        key, tl_id = record.get("key") or {}, record.get("value")
        video_id, chunk = key.get("video_id"), key.get("chunk")
        if not (video_id and tl_id):
            continue
        # Both the ingestion cache and segments.csv are 0-based. Getting this
        # wrong shifts every hit in a chunked video by ten minutes, which the
        # JSON cannot show you and the player very much can.
        index = int(chunk) if chunk is not None else 0
        mapping[str(tl_id)] = (video_id, offsets.get((video_id, index), 0.0))
    return mapping


def search(query: str, limit: int = 5) -> Optional[dict]:
    """Top moment for ``query``, already mapped to the original timeline."""
    api_key = os.getenv("TWELVELABS_API_KEY")
    index_id = os.getenv("TWELVELABS_INDEX_ID")
    if not (api_key and index_id):
        return None

    response = requests.post(
        f"{API}/search",
        headers={"x-api-key": api_key},
        files=[
            ("index_id", (None, index_id)),
            ("query_text", (None, query)),
            ("search_options", (None, "visual")),
            ("search_options", (None, "audio")),
            ("page_limit", (None, str(limit))),
        ],
        timeout=45,
    )
    response.raise_for_status()
    hits = response.json().get("data") or []
    if not hits:
        return None

    mapping = video_map()
    for hit in hits:
        known = mapping.get(str(hit.get("video_id")))
        if not known:
            continue  # a chunk we cannot place is worse than no answer
        video_id, offset = known
        start = float(hit["start"]) + offset
        end = float(hit["end"]) + offset
        if end - start < MIN_WINDOW:
            pad = (MIN_WINDOW - (end - start)) / 2
            start, end = max(0.0, start - pad), end + pad
        return {
            "video_id": video_id,
            "start": start,
            "end": end,
            "transcription": hit.get("transcription", ""),
        }
    return None


def cut_moment(video_id: str, start: float, end: float) -> Optional[str]:
    """Cut the searched moment so it plays in the same player as everything else."""
    source = MEDIA / f"{video_id}.mp4"
    if not source.exists():
        return None

    digest = hashlib.sha256(f"{video_id}:{start:.2f}:{end:.2f}".encode()).hexdigest()[:12]
    clip_id = f"search_{digest}"
    dest = CLIPS / f"{clip_id}.mp4"
    if dest.exists() and dest.stat().st_size > 0:
        return clip_id

    CLIPS.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
            "-ss", f"{start}", "-i", str(source), "-t", f"{end - start}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
            "-vf", "scale=-2:720",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
            str(dest),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        dest.unlink(missing_ok=True)
        return None
    return clip_id
