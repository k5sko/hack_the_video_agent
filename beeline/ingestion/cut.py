"""Cut every clip in graph_payload.json into its own MP4.

Two reasons this exists rather than seeking inside an embedded YouTube player.

First, presentation: an IFrame drags YouTube's whole interface along with it --
title bar, channel avatar, related-video grid on pause, occasional ads. Beeline
is supposed to look like a learning tool, not like a browser tab.

Second, and more practically: an embedded player cannot work offline. The demo
is meant to survive venue wifi with the network cable pulled, and every external
API response is already cached for exactly that reason. Streaming the actual
video from YouTube would have been the one remaining live dependency.

Cuts are re-encoded rather than stream-copied. `-c copy` is ~50x faster but snaps
to the nearest keyframe, which drifts a clip's start by up to several seconds --
invisible in the JSON, extremely visible when a clip opens mid-sentence.

    python -m beeline.ingestion.cut                # cut everything, skip existing
    python -m beeline.ingestion.cut --force        # re-cut
    python -m beeline.ingestion.cut --only v6_c2   # one clip
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
BEELINE = HERE.parent
PAYLOAD = HERE / "graph_payload.json"
MEDIA = BEELINE / "data" / "media"
CLIPS = BEELINE / "data" / "clips"

# 720p is legible on a projector without making the files enormous. crf 24 is
# visually clean for line-art lecture animation, which compresses very well.
HEIGHT = 720
CRF = 24
PRESET = "veryfast"


@dataclass
class Cut:
    clip_id: str
    source: Path
    start: float
    duration: float

    @property
    def dest(self) -> Path:
        return CLIPS / f"{self.clip_id}.mp4"


def build_cuts(payload: dict, only: Optional[str]) -> List[Cut]:
    cuts: List[Cut] = []
    missing: set[str] = set()
    for clip in payload["clips"]:
        if only and clip["id"] != only:
            continue
        source = MEDIA / f"{clip['video_id']}.mp4"
        if not source.exists():
            missing.add(clip["video_id"])
            continue
        duration = float(clip["end"]) - float(clip["start"])
        if duration <= 0:
            continue
        cuts.append(Cut(clip["id"], source, float(clip["start"]), duration))
    if missing:
        print(
            f"  ! no source video for {sorted(missing)} "
            f"-- run fetch_media first",
            file=sys.stderr,
        )
    return cuts


def cut_one(cut: Cut, force: bool) -> tuple[str, bool, str]:
    if cut.dest.exists() and not force and cut.dest.stat().st_size > 0:
        return cut.clip_id, True, "exists"

    tmp = cut.dest.with_suffix(".part.mp4")
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        # -ss before -i seeks fast; combined with re-encoding the result is
        # still frame-accurate, unlike -c copy.
        "-ss",
        f"{cut.start}",
        "-i",
        str(cut.source),
        "-t",
        f"{cut.duration}",
        "-c:v",
        "libx264",
        "-preset",
        PRESET,
        "-crf",
        str(CRF),
        "-vf",
        f"scale=-2:{HEIGHT}",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        # put the index at the front so the browser can start playing before the
        # whole file arrives
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        return cut.clip_id, False, (proc.stderr or "ffmpeg failed").strip()[:200]
    tmp.replace(cut.dest)
    return cut.clip_id, True, "cut"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--payload", default=str(PAYLOAD))
    ap.add_argument("--force", action="store_true", help="re-cut clips that exist")
    ap.add_argument("--only", default=None, help="cut a single clip id")
    ap.add_argument("--jobs", type=int, default=6, help="parallel ffmpeg processes")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        print("ffmpeg not found on PATH", file=sys.stderr)
        return 2

    payload = json.loads(Path(args.payload).read_text())
    CLIPS.mkdir(parents=True, exist_ok=True)
    cuts = build_cuts(payload, args.only)
    if not cuts:
        print("nothing to cut")
        return 1

    print(f"cutting {len(cuts)} clips into {CLIPS} with {args.jobs} workers")
    done = failed = skipped = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(cut_one, c, args.force): c for c in cuts}
        for future in as_completed(futures):
            clip_id, ok, note = future.result()
            if not ok:
                failed += 1
                print(f"  FAIL {clip_id}: {note}", file=sys.stderr)
            elif note == "exists":
                skipped += 1
            else:
                done += 1
                if done % 10 == 0:
                    print(f"  {done} cut...")

    total_bytes = sum(f.stat().st_size for f in CLIPS.glob("*.mp4"))
    print(
        f"cut={done} skipped={skipped} failed={failed}  "
        f"total={total_bytes / 1e6:.0f} MB in {CLIPS}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
