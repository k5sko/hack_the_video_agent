"""Download the corpus videos + English captions into data/media/.

TwelveLabs cannot ingest a YouTube page URL, so we need real local files.
Downloads are gitignored (*.mp4/*.webm/*.mkv); only the API cache is committed.

YouTube intermittently answers with "Sign in to confirm you're not a bot" for
the default player client, so we fall through a list of clients per video.
"""

from __future__ import annotations

import subprocess
import sys

from .config import CORPUS, MEDIA_DIR, REPO_ROOT, youtube_url
from .media import have_media, video_path

YTDLP = REPO_ROOT / ".venv" / "bin" / "yt-dlp"

# Tried in order; the default web client is bot-checked most aggressively.
PLAYER_CLIENTS = ["default", "tv", "android_vr", "web_embedded", "ios", "mweb"]


def fetch(video_id: str, youtube_id: str) -> bool:
    if have_media(video_id):
        print(f"[skip] {video_id} already present")
        return True

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    for client in PLAYER_CLIENTS:
        cmd = [
            str(YTDLP),
            "-f", "bv*[height<=480]+ba/b[height<=480]/b",
            "--merge-output-format", "mp4",
            "--write-auto-subs", "--write-subs",
            "--sub-lang", "en", "--sub-format", "vtt", "--convert-subs", "vtt",
            "-o", f"{video_id}.%(ext)s",
            youtube_url(youtube_id),
        ]
        if client != "default":
            cmd[1:1] = ["--extractor-args", f"youtube:player_client={client}"]

        proc = subprocess.run(cmd, cwd=MEDIA_DIR, capture_output=True, text=True)
        if have_media(video_id):
            print(f"[ok  ] {video_id} via client={client} "
                  f"({video_path(video_id).stat().st_size // 1_000_000} MB)")
            return True
        tail = (proc.stderr or "").strip().splitlines()[-1:] or [""]
        print(f"[retry] {video_id} client={client}: {tail[0][:120]}")
    print(f"[FAIL] {video_id}: all player clients rejected")
    return False


def main() -> int:
    ok = 0
    for v in CORPUS:  # demo order: V6/V5 first
        if fetch(v["id"], v["youtube_id"]):
            ok += 1
    print(f"\nfetched {ok}/{len(CORPUS)} videos")
    return 0 if ok >= 5 else 1


if __name__ == "__main__":
    sys.exit(main())
