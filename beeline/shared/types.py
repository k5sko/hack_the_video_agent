"""Python mirror of shared/types.ts.

P3 (path engine) and P4 (API) build PathResult through these models so a contract
drift shows up as a validation error here, not as a blank screen in the demo.

Keep field names, order, and types identical to types.ts.
"""

from typing import List, Literal

from pydantic import BaseModel, Field

NodeState = Literal["on_path", "known", "not_needed", "gap"]


class ClipSegment(BaseModel):
    clip_id: str
    video_id: str
    video_title: str
    youtube_url: str
    start_seconds: float
    end_seconds: float
    covers: List[str]
    why: str


class PathResult(BaseModel):
    query: str
    mode: Literal["graph", "search_only"]
    target_concepts: List[str]
    known: List[str]
    needed_concepts: List[str]
    playlist: List[ClipSegment]
    gaps: List[str]
    total_corpus_seconds: float
    watch_seconds: float
    narration: str


class PathRequest(BaseModel):
    """Body of POST /api/path."""

    query: str
    known: List[str] = Field(default_factory=list)
    mode: Literal["graph", "search_only"] = "graph"
