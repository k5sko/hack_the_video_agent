"""Beeline P2 ingestion entrypoint.

    python -m beeline.ingestion.ingest              # full agent run
    python -m beeline.ingestion.ingest --from-cache # rebuild, ZERO API calls

ONE Strands Agent (OpenAI provider) with six @tool functions drives:
    for each video -> upload -> wait -> chapters -> extract   then canonicalize -> append.

`--from-cache` sets the cache to OFFLINE, so any attempted network call raises
CacheMiss instead of silently costing money. That is how "a re-run costs zero
API calls" is proven rather than asserted.
"""

from __future__ import annotations

import argparse
import json
import sys

from strands import Agent
from strands.models.openai import OpenAIModel

from . import build, cache, tools
from .config import CORPUS, OPENAI_AGENT_MODEL, PAYLOAD_PATH, env
from .media import have_media

SYSTEM_PROMPT = """You are the Beeline ingestion agent. You build a prerequisite
knowledge graph from a corpus of 3Blue1Brown neural-network videos.

Process the videos in EXACTLY this order (demo order -- the first two matter most):
{order}

For EACH video, in order, call these tools one video at a time:
  1. upload_video(video_id)
  2. await_ready(video_id)
  3. get_chapters(video_id)
  4. extract_concepts(video_id)

If a tool returns a string starting with ERROR, do not retry it more than once;
move on to the next video so a partial corpus still produces a payload.

After ALL videos are done, call canonicalize() exactly once, then
append_payload() exactly once. Then reply with a one-paragraph summary of the
counts you saw. Do not call any tool after append_payload().
"""


def run_agent(video_ids: list[str]) -> str:
    model = OpenAIModel(
        client_args={"api_key": env("OPENAI_API_KEY")},
        model_id=OPENAI_AGENT_MODEL,
        params={"temperature": 0},
    )
    agent = Agent(
        model=model,
        tools=tools.ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT.format(order=", ".join(video_ids)),
        callback_handler=None,
    )
    result = agent(
        "Ingest the corpus now, following your instructions exactly. "
        f"The videos to process, in order, are: {', '.join(video_ids)}."
    )
    return str(result)


def rebuild_from_cache() -> dict:
    """Deterministic rebuild with no agent and no network."""
    from . import tl
    from .media import split_video

    ready: dict[str, dict] = {}
    for v in CORPUS:
        if not have_media(v["id"]):
            continue
        for c in split_video(v["id"]):
            hit = cache.peek("tl_ready", {"task_id_of": {
                "video_id": c.video_id, "chunk": c.index,
                "fingerprint": tl.file_fingerprint(c.path)}})
            if hit:
                ready[c.key] = hit
    payload = build.build_payload(ready)
    build.write_payload(payload)
    return payload


def verify(payload: dict) -> list[str]:
    """Report on the acceptance criteria honestly instead of forcing them."""
    problems = []
    if len(payload["videos"]) < 5:
        problems.append(f"only {len(payload['videos'])} videos (need >=5)")
    if len(payload["concepts"]) < 50:
        problems.append(f"only {len(payload['concepts'])} concepts (need >=50)")
    if not payload["explains"]:
        problems.append("no EXPLAINS edges")
    if not payload["requires"]:
        problems.append("no REQUIRES edges")

    names = {c["name"] for c in payload["concepts"]}
    for key in ("attention", "softmax", "embedding", "backpropagation"):
        if key not in names:
            problems.append(f"missing canonical node '{key}'")
    return problems


def near_duplicates(payload: dict) -> dict[str, list[str]]:
    """Informational: other nodes whose name contains a key concept.

    Reported, not treated as failure -- 'cross attention' being its own node is
    defensible; 'attention'/'self-attention' both existing is not.
    """
    names = {c["name"] for c in payload["concepts"]}
    out = {}
    for key in ("attention", "softmax", "embedding", "backpropagation"):
        near = sorted(n for n in names if n != key and key in n)
        if near:
            out[key] = near
    return out


def report(payload: dict) -> None:
    names = {c["name"] for c in payload["concepts"]}
    explained = {e["concept"] for e in payload["explains"]}

    print("\n=== graph_payload.json ===")
    for k in ("videos", "clips", "concepts", "explains", "requires"):
        print(f"  {k:9s} {len(payload[k])}")
    print(f"  chapter source: {json.dumps(build.CHAPTER_SOURCE)}")

    print("\n  canonical key nodes:")
    for key in ("attention", "softmax", "embedding", "backpropagation"):
        variants = sorted(n for n in names if key in n)
        print(f"    {key:18s} present={key in names}  all_matching={variants}")

    pe = "positional encoding"
    required_by = [r["from"] for r in payload["requires"] if r["to"] == pe]
    print(f"\n  '{pe}': in_graph={pe in names} explained_by_clip={pe in explained} "
          f"required_by={required_by[:6]}")
    if pe in names and pe not in explained and required_by:
        print("    -> HONEST GAP present naturally (required, never explained)")
    elif pe in explained:
        print("    -> NOT a gap: some clip explains it")
    elif pe not in names:
        print("    -> NOT extracted at all")

    gaps = sorted({r["to"] for r in payload["requires"]} - explained)
    print(f"\n  total honest gaps (required but never explained): {len(gaps)}")
    print(f"    e.g. {gaps[:10]}")

    problems = verify(payload)
    print("\n  ACCEPTANCE:", "ALL PASS" if not problems else "ISSUES")
    for p in problems:
        print("   -", p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-cache", action="store_true",
                    help="rebuild payload from cache only; any API call is an error")
    ap.add_argument("--videos", default="",
                    help="comma-separated subset, e.g. V6,V5")
    args = ap.parse_args()

    ids = [v.strip().upper() for v in args.videos.split(",") if v.strip()] or \
          [v["id"] for v in CORPUS]
    ids = [i for i in ids if have_media(i)]

    if args.from_cache:
        cache.set_offline(True)
        payload = rebuild_from_cache()
        report(payload)
        calls = cache.api_calls_made()
        print(f"\n  API calls made: {calls}")
        if calls:
            print("  FAIL: --from-cache made network calls")
            return 1
        print("  OK: rebuilt entirely from cache with zero API calls")
        return 0

    tools.reset_state()
    summary = run_agent(ids)
    print("\n--- agent summary ---\n" + summary)

    if not PAYLOAD_PATH.exists():
        print("ERROR: agent did not produce graph_payload.json")
        return 1
    payload = json.loads(PAYLOAD_PATH.read_text())
    report(payload)
    print(f"\n  API calls made this run: {cache.api_calls_made()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
