"""OpenAI concept extraction and canonicalization.

Per chapter: ONE structured-output call returning exactly
    {"teaches": [{"concept": "softmax", "score": 0.88}], "assumes": ["vectors"]}

Canonicalization order matters and is deliberate:
  1. normalize surface form
  2. embed every concept and merge pairs above ~0.85 cosine
  3. THEN apply aliases.json overrides (hand-written, demo-critical)
Aliases run last so a hand-written demo-critical merge always wins over whatever
the embeddings decided.
"""

from __future__ import annotations

import json
import re
from typing import Iterable

from openai import OpenAI

from . import cache
from .config import (ALIASES_PATH, MERGE_COSINE_THRESHOLD, OPENAI_EMBED_MODEL,
                     OPENAI_EXTRACT_MODEL, env)

_client: OpenAI | None = None

EXTRACT_PROMPT_VERSION = "v1"

SYSTEM = (
    "You extract a prerequisite knowledge graph from segments of educational "
    "machine-learning videos (3Blue1Brown neural network series).\n"
    "Given a chapter's title and summary, return:\n"
    "  teaches: concepts this chapter actually EXPLAINS, each with a 0-1 confidence "
    "score for how centrally the chapter teaches it. Max 5.\n"
    "  assumes: concepts a viewer must ALREADY understand to follow this chapter, "
    "but which this chapter does NOT itself explain. Max 5.\n"
    "Rules: concepts are lowercase noun phrases of at most 4 words. Use canonical "
    "technical vocabulary ('softmax', 'dot product', 'positional encoding', "
    "'backpropagation', 'gradient descent'). No verbs, no full sentences, no "
    "markdown. Do not put the same concept in both lists."
)

EXTRACT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "teaches": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "concept": {"type": "string"},
                    "score": {"type": "number"},
                },
                "required": ["concept", "score"],
            },
        },
        "assumes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["teaches", "assumes"],
}


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=env("OPENAI_API_KEY"))
    return _client


# --- surface normalization --------------------------------------------------

_STOP_PREFIX = re.compile(r"^(the|a|an)\s+")

# Embeddings alone do NOT reliably merge "word embeddings" with "word embedding"
# (cosine sits well under 0.85), so plurals are folded by rule. Applied to
# aliases.json too, so both sides normalize identically.
_IRREGULAR = {
    "matrices": "matrix", "indices": "index", "vertices": "vertex",
    "axes": "axis", "biases": "bias", "losses": "loss", "classes": "class",
    "series": "series", "calculus": "calculus", "basis": "basis",
    "bias": "bias", "gradients": "gradient",
}
# Words where a trailing "s" is part of the stem, not a plural marker.
_KEEP_S = re.compile(r"(ss|us|is|as|os|ics)$")


def _singular(word: str) -> str:
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if len(word) <= 3 or _KEEP_S.search(word) or not word.endswith("s"):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and re.search(r"(ch|sh|x|z)es$", word):
        return word[:-2]
    return word[:-1]


def normalize(name: str) -> str:
    n = (name or "").strip().lower()
    n = re.sub(r"[^a-z0-9\s\-]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    n = _STOP_PREFIX.sub("", n)
    n = re.sub(r"\s+", " ", n).strip()
    if not n:
        return ""
    words = n.split()[:4]
    words = [_singular(w) for w in words]
    return " ".join(words).strip()


def extract_chapter(clip_id: str, title: str, summary: str) -> dict:
    """One structured-output call per chapter. Cached by content, not by clip id."""
    text = f"Title: {title}\nSummary: {summary}"
    key = {"text": text, "model": OPENAI_EXTRACT_MODEL, "v": EXTRACT_PROMPT_VERSION}

    def _do() -> dict:
        resp = client().chat.completions.create(
            model=OPENAI_EXTRACT_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": text},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "concept_extraction", "strict": True,
                                "schema": EXTRACT_SCHEMA},
            },
        )
        return json.loads(resp.choices[0].message.content)

    raw = cache.cached("oa_extract", key, _do)

    teaches, seen = [], set()
    for t in raw.get("teaches", []):
        c = normalize(t.get("concept", ""))
        if c and c not in seen:
            seen.add(c)
            score = float(t.get("score", 0.5))
            teaches.append({"concept": c, "score": max(0.0, min(1.0, score))})

    assumes = []
    for a in raw.get("assumes", []):
        c = normalize(a if isinstance(a, str) else a.get("concept", ""))
        if c and c not in seen and c not in assumes:
            assumes.append(c)

    return {"clip_id": clip_id, "teaches": teaches, "assumes": assumes}


# --- canonicalization -------------------------------------------------------

def embed(names: list[str]) -> dict[str, list[float]]:
    """Embed concept names, cached per-name so new concepts cost only themselves."""
    out: dict[str, list[float]] = {}
    missing = []
    for n in names:
        hit = cache.peek("oa_embed", {"text": n, "model": OPENAI_EMBED_MODEL})
        if hit is not None:
            out[n] = hit
            cache.STATS["hits"] += 1
        else:
            missing.append(n)

    if missing:
        if cache.is_offline():
            raise cache.CacheMiss(f"OFFLINE: {len(missing)} concepts never embedded")
        for i in range(0, len(missing), 256):
            batch = missing[i:i + 256]
            cache.STATS["misses"] += 1
            resp = client().embeddings.create(model=OPENAI_EMBED_MODEL, input=batch)
            for name, item in zip(batch, resp.data):
                vec = list(item.embedding)
                out[name] = vec
                cache.cached("oa_embed", {"text": name, "model": OPENAI_EMBED_MODEL},
                             lambda v=vec: v)
    return out


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def load_aliases() -> dict[str, str]:
    """alias surface form -> canonical name."""
    raw = json.loads(ALIASES_PATH.read_text())
    mapping: dict[str, str] = {}
    for canon, variants in raw.items():
        c = normalize(canon)
        mapping[c] = c
        for v in variants:
            mapping[normalize(v)] = c
    return mapping


def canonicalize(concepts: Iterable[str]) -> dict[str, str]:
    """Return {raw_concept -> canonical_concept}.

    Embedding merge first, aliases.json override last (hand-written wins).
    """
    names = sorted({normalize(c) for c in concepts if normalize(c)})
    if not names:
        return {}

    vectors = embed(names)

    aliases = load_aliases()
    canon_targets = set(aliases.values())

    # Frequency-independent but deterministic: prefer the shorter, then
    # alphabetically-first name as the representative of a merged group.
    order = sorted(names, key=lambda n: (len(n.split()), len(n), n))
    parent: dict[str, str] = {n: n for n in order}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def protected(group: str) -> set[str]:
        """Distinct hand-written canonical names currently inside a group."""
        return {n for n in names if find(n) == group and n in canon_targets}

    for i, a in enumerate(order):
        va = vectors.get(a)
        if not va:
            continue
        for b in order[i + 1:]:
            vb = vectors.get(b)
            if not vb:
                continue
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            if _cos(va, vb) < MERGE_COSINE_THRESHOLD:
                continue
            # Never let cosine similarity collapse two DIFFERENT hand-written
            # canonical concepts together. Without this, e.g. "positional
            # encoding" can be absorbed into "embedding" and the node vanishes.
            if len(protected(ra) | protected(rb)) > 1:
                continue
            parent[rb] = ra  # `a` is earlier in order -> representative

    mapping = {n: find(n) for n in names}

    # If any member of an embedding group is a known alias, the whole group
    # collapses onto that hand-written canonical name.
    group_override: dict[str, str] = {}
    for raw, grp in mapping.items():
        if raw in aliases:
            group_override.setdefault(grp, aliases[raw])
    # A group whose representative IS a canonical name keeps it.
    for raw, grp in mapping.items():
        if raw in canon_targets:
            group_override[grp] = raw

    final = {raw: group_override.get(grp, grp) for raw, grp in mapping.items()}
    # Direct alias hits always win over group inference.
    for raw in list(final):
        if raw in aliases:
            final[raw] = aliases[raw]
    return final
