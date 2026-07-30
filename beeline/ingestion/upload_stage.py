"""Stage 1 driver: split every corpus video and index all chunks CONCURRENTLY.

Run this as early as possible -- indexing is wall-clock bound and everything
downstream (Pegasus chapters) waits on it. Concept extraction does NOT wait on
this stage; it runs off captions in parallel.
"""

from __future__ import annotations

import concurrent.futures as cf
import sys

from . import tl
from .config import CORPUS
from .media import Chunk, have_media, split_video


def all_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for v in CORPUS:
        if not have_media(v["id"]):
            print(f"[skip] no media for {v['id']}", flush=True)
            continue
        chunks.extend(split_video(v["id"]))
    return chunks


def run(chunks: list[Chunk] | None = None) -> dict[str, dict]:
    chunks = chunks if chunks is not None else all_chunks()
    results: dict[str, dict] = {}

    def _one(c: Chunk) -> tuple[str, dict]:
        task_id = tl.upload_chunk(c)
        print(f"[upload] {c.key} task={task_id}", flush=True)
        info = tl.await_ready(c, task_id)
        print(f"[ready ] {c.key} tl_video={info['tl_video_id']}", flush=True)
        return c.key, info

    with cf.ThreadPoolExecutor(max_workers=len(chunks) or 1) as pool:
        futs = {pool.submit(_one, c): c for c in chunks}
        for fut in cf.as_completed(futs):
            c = futs[fut]
            try:
                key, info = fut.result()
                results[key] = info
            except Exception as exc:  # keep going; a partial corpus still demos
                print(f"[FAIL  ] {c.key}: {exc}", flush=True)
                results[c.key] = {"error": str(exc)}
    return results


if __name__ == "__main__":
    res = run()
    ok = sum(1 for v in res.values() if "error" not in v)
    print(f"\nINDEXED {ok}/{len(res)} chunks", flush=True)
    sys.exit(0 if ok else 1)
