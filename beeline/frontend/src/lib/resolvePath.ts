/**
 * The ONLY module in the frontend that knows where a PathResult comes from.
 *
 * It calls POST /api/path, and falls back to the bundled fixtures if the API is
 * unreachable. That fallback is not just developer convenience: it means the UI
 * still demonstrates the full interaction if the backend dies mid-demo, which is
 * precisely when a blank screen is least affordable.
 */

import type { ClipSegment, PathResult } from "@shared/types";

import coldFixture from "../data/path_attention_cold.json";
import knownFixture from "../data/path_attention_known.json";
import searchOnlyFixture from "../data/path_search_only.json";

export type Mode = PathResult["mode"];

export const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined) ??
  "http://localhost:8000";

/**
 * Concepts offered in the "I already know…" checklist. Every entry is a real
 * node in the concept graph.
 */
export const KNOWABLE_CONCEPTS: string[] = [
  "linear algebra",
  "matrix multiplication",
  "neuron",
  "layer",
  "neural network",
  "dot product",
  "softmax",
  "embedding",
];

/**
 * Ticking any of these prunes the foundational half of the path. Used only by
 * the offline fallback, which cannot actually recompute anything.
 */
const FOUNDATIONAL: ReadonlySet<string> = new Set([
  "linear algebra",
  "matrix multiplication",
  "neuron",
  "layer",
  "neural network",
]);

const COLD = coldFixture as PathResult;
const KNOWN = knownFixture as PathResult;
const SEARCH_ONLY = searchOnlyFixture as PathResult;

/** Absolute URL for a clip's own media file. */
export function mediaSrc(clip: ClipSegment): string {
  const url = clip.media_url || "";
  if (!url) return "";
  return url.startsWith("http") ? url : `${API_BASE}${url}`;
}

function offlinePath(query: string, known: string[], mode: Mode): PathResult {
  if (mode === "search_only") {
    return { ...SEARCH_ONLY, query, known: [...known] };
  }
  const base = known.some((c) => FOUNDATIONAL.has(c)) ? KNOWN : COLD;
  return { ...base, query, known: [...known] };
}

export interface PathOutcome {
  path: PathResult;
  /** True when the API was unreachable and a bundled fixture was served. */
  offline: boolean;
}

/**
 * Resolve a learning path.
 *
 * @param query free-text target concept, e.g. "attention"
 * @param known concepts the learner has ticked as already understood
 * @param mode  "graph" (prerequisite-ordered) or "search_only" (keyword hit)
 */
export async function getPath(
  query: string,
  known: string[],
  mode: Mode,
): Promise<PathOutcome> {
  const target = query.trim() || COLD.query;
  try {
    const response = await fetch(`${API_BASE}/api/path`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: target, known, mode }),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return { path: (await response.json()) as PathResult, offline: false };
  } catch {
    return { path: offlinePath(target, known, mode), offline: true };
  }
}

/** The live concept graph, or null if the API is unreachable (keep the fixture). */
export async function getGraph(): Promise<unknown | null> {
  try {
    const response = await fetch(`${API_BASE}/api/graph`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch {
    return null;
  }
}

/** The full corpus length, used by the counter's left-hand side. */
export const TOTAL_CORPUS_SECONDS = COLD.total_corpus_seconds;
