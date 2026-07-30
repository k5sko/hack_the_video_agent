"""Local media handling: probe, split, and caption parsing.

The single most dangerous thing in this slice is timestamp drift. `ffmpeg -c copy`
splits land on keyframes, so a chunk does NOT necessarily start at the requested
offset. We therefore use the segment muxer's `-segment_list` CSV, which reports
each segment's ACTUAL start/end time, and we carry that real offset forward.
Every emitted timestamp is corrected back to the original video's timeline.
"""

from __future__ import annotations

import csv
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import CHUNK_DIR, CHUNK_SECONDS, MEDIA_DIR, SPLIT_THRESHOLD_SECONDS


@dataclass
class Chunk:
    video_id: str
    index: int
    path: Path
    offset: float  # real start time in the ORIGINAL video's timeline
    duration: float

    @property
    def key(self) -> str:
        return f"{self.video_id}_chunk{self.index:02d}"


def video_path(video_id: str) -> Path:
    return MEDIA_DIR / f"{video_id}.mp4"


def caption_path(video_id: str) -> Path:
    return MEDIA_DIR / f"{video_id}.en.vtt"


def have_media(video_id: str) -> bool:
    p = video_path(video_id)
    return p.exists() and p.stat().st_size > 0


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def split_video(video_id: str) -> list[Chunk]:
    """Split into ~10min chunks when over threshold. Returns chunks with REAL offsets."""
    src = video_path(video_id)
    total = probe_duration(src)

    if total <= SPLIT_THRESHOLD_SECONDS:
        return [Chunk(video_id, 0, src, 0.0, total)]

    out_dir = CHUNK_DIR / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    listing = out_dir / "segments.csv"

    if not listing.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-c", "copy",
             "-f", "segment", "-segment_time", str(CHUNK_SECONDS),
             "-reset_timestamps", "1",
             "-segment_list", str(listing), "-segment_list_type", "csv",
             str(out_dir / f"{video_id}_%03d.mp4")],
            check=True, capture_output=True,
        )

    chunks: list[Chunk] = []
    with open(listing) as fh:
        for i, row in enumerate(csv.reader(fh)):
            if len(row) < 3:
                continue
            name, start, end = row[0], float(row[1]), float(row[2])
            chunks.append(Chunk(video_id, i, out_dir / Path(name).name, start, end - start))

    # The segment muxer's own report is the authority on where each chunk starts.
    assert chunks and abs(chunks[0].offset) < 1e-6, f"{video_id}: first chunk must start at 0"
    for a, b in zip(chunks, chunks[1:]):
        assert b.offset >= a.offset, f"{video_id}: chunk offsets not monotonic"
    assert abs((chunks[-1].offset + chunks[-1].duration) - total) < 5.0, (
        f"{video_id}: chunk coverage {chunks[-1].offset + chunks[-1].duration:.1f}s "
        f"!= duration {total:.1f}s"
    )
    return chunks


# --- Captions (fallback chaptering) -----------------------------------------

_TS = re.compile(r"(\d+):(\d{2}):(\d{2})\.(\d{3})\s+-->\s+(\d+):(\d{2}):(\d{2})\.(\d{3})")


def _to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_vtt(video_id: str) -> list[tuple[float, float, str]]:
    """Parse a WebVTT file into (start, end, text) cues, de-duplicated.

    YouTube auto-captions roll words forward across cues, producing heavy
    duplication; we keep only newly-added text per cue.
    """
    p = caption_path(video_id)
    if not p.exists():
        return []

    cues: list[tuple[float, float, str]] = []
    cur: tuple[float, float] | None = None
    buf: list[str] = []

    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = _TS.search(line)
        if m:
            if cur and buf:
                cues.append((cur[0], cur[1], " ".join(buf)))
            cur = (_to_seconds(*m.groups()[:4]), _to_seconds(*m.groups()[4:]))
            buf = []
            continue
        if not cur:
            continue
        txt = re.sub(r"<[^>]+>", "", line).strip()
        if txt and txt.upper() != "WEBVTT" and not txt.startswith("Kind:") and not txt.startswith("Language:"):
            buf.append(txt)
    if cur and buf:
        cues.append((cur[0], cur[1], " ".join(buf)))

    # Collapse rolling duplicates: keep text not already ending the previous cue.
    cleaned: list[tuple[float, float, str]] = []
    seen_tail = ""
    for start, end, text in cues:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        if text == seen_tail:
            continue
        if seen_tail and text.startswith(seen_tail):
            text = text[len(seen_tail):].strip()
        seen_tail = re.sub(r"\s+", " ", " ".join(text.split()[-12:]))
        if text:
            cleaned.append((start, end, text))
    return cleaned


def caption_windows(video_id: str, window: float = 90.0) -> list[dict]:
    """Fixed ~90s windows snapped to caption sentence boundaries.

    This is the fallback chapterer. It is built deliberately so concept
    extraction is never blocked on TwelveLabs indexing.
    """
    cues = parse_vtt(video_id)
    if not cues:
        return []

    windows: list[dict] = []
    start = cues[0][0]
    words: list[str] = []
    last_end = start

    for cue_start, cue_end, text in cues:
        words.append(text)
        last_end = cue_end
        spans = cue_end - start
        ends_sentence = text.rstrip().endswith((".", "?", "!"))
        if spans >= window and (ends_sentence or spans >= window * 1.6):
            body = re.sub(r"\s+", " ", " ".join(words)).strip()
            if body:
                windows.append({"start": round(start, 2), "end": round(cue_end, 2), "text": body})
            start = cue_end
            words = []

    if words:
        body = re.sub(r"\s+", " ", " ".join(words)).strip()
        if body:
            windows.append({"start": round(start, 2), "end": round(last_end, 2), "text": body})
    return windows
