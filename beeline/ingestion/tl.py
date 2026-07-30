"""TwelveLabs wrapper: index, upload chunks concurrently, poll, Pegasus chapters.

All network calls route through cache.cached() so a re-run costs zero API calls.
Cache keys are built from stable request descriptors (video id, chunk index,
content hash, prompt version) -- never from task ids or timestamps.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from twelvelabs import TwelveLabs

from . import cache
from .config import MARENGO_MODEL, PEGASUS_MODEL, env
from .media import Chunk

_client: TwelveLabs | None = None

# Bump when the chapter prompt/schema changes so stale chapters are not reused.
CHAPTER_PROMPT_VERSION = "v1"

CHAPTER_PROMPT = (
    "You are segmenting an educational math/ML lecture video into chapters. "
    "Divide this video into 3 to 8 contiguous chapters at genuine topic boundaries. "
    "Chapters must cover the video in order and must not overlap. "
    "For each chapter give: start_time and end_time in SECONDS relative to the start "
    "of THIS video clip, a short title, and a 1-3 sentence summary describing the "
    "specific technical concepts explained in that chapter (name them explicitly, "
    "e.g. 'softmax', 'dot product', 'positional encoding')."
)

CHAPTER_SCHEMA = {
    "type": "object",
    "properties": {
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "number"},
                    "end_time": {"type": "number"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["start_time", "end_time", "title", "summary"],
            },
        }
    },
    "required": ["chapters"],
}


def client() -> TwelveLabs:
    global _client
    if _client is None:
        _client = TwelveLabs(api_key=env("TWELVELABS_API_KEY"))
    return _client


def index_id() -> str:
    return env("TWELVELABS_INDEX_ID")


def file_fingerprint(path: Path) -> str:
    """Cheap stable content id: size + head/tail bytes. Avoids hashing 50MB."""
    st = path.stat()
    h = hashlib.sha256()
    h.update(str(st.st_size).encode())
    with open(path, "rb") as fh:
        h.update(fh.read(65536))
        if st.st_size > 131072:
            fh.seek(-65536, 2)
            h.update(fh.read(65536))
    return h.hexdigest()[:24]


def upload_chunk(chunk: Chunk) -> str:
    """Create an indexing task for one chunk. Returns the TwelveLabs task id."""
    key = {
        "video_id": chunk.video_id,
        "chunk": chunk.index,
        "fingerprint": file_fingerprint(chunk.path),
        "index_id": index_id(),
        "model": MARENGO_MODEL,
    }

    def _do() -> str:
        with open(chunk.path, "rb") as fh:
            task = client().tasks.create(index_id=index_id(), video_file=fh)
        return task.id

    return cache.cached("tl_upload", key, _do)


def await_ready(chunk: Chunk, task_id: str, timeout: float = 3600.0) -> dict:
    """Poll an indexing task until ready. Returns {task_id, tl_video_id, status}."""
    key = {"task_id_of": {"video_id": chunk.video_id, "chunk": chunk.index,
                          "fingerprint": file_fingerprint(chunk.path)}}

    def _do() -> dict:
        deadline = time.time() + timeout
        delay = 5.0
        while time.time() < deadline:
            t = client().tasks.retrieve(task_id=task_id)
            status = (t.status or "").lower()
            if status == "ready":
                return {"task_id": task_id, "tl_video_id": t.video_id, "status": "ready"}
            if status in {"failed", "error"}:
                raise RuntimeError(f"Indexing failed for {chunk.key}: {status}")
            time.sleep(delay)
            delay = min(delay * 1.3, 20.0)
        raise TimeoutError(f"Indexing timed out for {chunk.key}")

    return cache.cached("tl_ready", key, _do)


def get_chapters(chunk: Chunk, tl_video_id: str) -> list[dict]:
    """Pegasus chapters for one chunk, in the CHUNK's local timeline (seconds)."""
    key = {
        "video_id": chunk.video_id,
        "chunk": chunk.index,
        "tl_video_id": tl_video_id,
        "model": PEGASUS_MODEL,
        "prompt_version": CHAPTER_PROMPT_VERSION,
    }

    def _do() -> list[dict]:
        resp = client().analyze(
            model_name=PEGASUS_MODEL,
            video_id=tl_video_id,
            prompt=CHAPTER_PROMPT,
            temperature=0.2,
            max_tokens=2000,
            response_format={"type": "json_schema", "json_schema": CHAPTER_SCHEMA},
        )
        text = getattr(resp, "data", None) or getattr(resp, "text", None) or ""
        if isinstance(text, str):
            parsed = json.loads(text)
        else:
            parsed = text
        return parsed.get("chapters", [])

    return cache.cached("tl_chapters", key, _do)
