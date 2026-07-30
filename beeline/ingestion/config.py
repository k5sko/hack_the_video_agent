"""Static configuration for the Beeline P2 ingestion slice.

Corpus order is DEMO order, not chronological order: V6/V5 (the attention path
the demo hinges on) are processed first so that a partial run still demos.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Credentials live in the REPO ROOT .env (gitignored), not in beeline/.
REPO_ROOT = Path(__file__).resolve().parents[2]
BEELINE = REPO_ROOT / "beeline"
INGESTION = BEELINE / "ingestion"
DATA = BEELINE / "data"
CACHE_DIR = DATA / "cache"
MEDIA_DIR = DATA / "media"
CHUNK_DIR = MEDIA_DIR / "chunks"
PAYLOAD_PATH = INGESTION / "graph_payload.json"
ALIASES_PATH = INGESTION / "aliases.json"

load_dotenv(REPO_ROOT / ".env")


def env(name: str, required: bool = True) -> str:
    val = os.environ.get(name, "").strip()
    if required and not val:
        raise RuntimeError(f"Missing required env var {name} in {REPO_ROOT / '.env'}")
    return val


# --- Models -----------------------------------------------------------------
# Validated against the live TwelveLabs API, which reports the currently valid
# model names as exactly: marengo3.0, pegasus1.2. (The bundled SDK also lists
# pegasus1.5, but this account/API version rejects it -- do not hardcode it.)
MARENGO_MODEL = "marengo3.0"
PEGASUS_MODEL = "pegasus1.2"

OPENAI_EXTRACT_MODEL = "gpt-4.1-mini"
OPENAI_EMBED_MODEL = "text-embedding-3-small"
OPENAI_AGENT_MODEL = "gpt-4.1-mini"

# Videos longer than this get split into CHUNK_SECONDS pieces so that indexing
# happens concurrently -- per-chunk indexing is what collapses wall-clock wait.
SPLIT_THRESHOLD_SECONDS = 20 * 60
CHUNK_SECONDS = 600

MERGE_COSINE_THRESHOLD = 0.85

# --- Corpus (DEMO ORDER) ----------------------------------------------------
CORPUS = [
    {"id": "V6", "youtube_id": "eMlx5fFNoYc", "title": "Attention in transformers, step-by-step"},
    {"id": "V5", "youtube_id": "wjZofJX0v4M", "title": "But what is a GPT? Visual intro to transformers"},
    {"id": "V1", "youtube_id": "aircAruvnKk", "title": "But what is a neural network?"},
    {"id": "V2", "youtube_id": "IHZwWFHWa-w", "title": "Gradient descent, how neural networks learn"},
    {"id": "V3", "youtube_id": "Ilg3gGewQ5U", "title": "What is backpropagation really doing?"},
    {"id": "V4", "youtube_id": "tIeHLnjs5U8", "title": "Backpropagation calculus"},
    {"id": "V7", "youtube_id": "9-Jl0dxWQs8", "title": "How might LLMs store facts"},
]

CORPUS_BY_ID = {v["id"]: v for v in CORPUS}


def youtube_url(youtube_id: str) -> str:
    return f"https://www.youtube.com/watch?v={youtube_id}"
