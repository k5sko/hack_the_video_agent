/**
 * The ONLY module in the frontend that knows where a PathResult comes from.
 *
 * Today it resolves against local JSON fixtures in src/data/ so the whole demo
 * runs with no backend. In P4 the body of `getPath` is replaced with
 * `POST /api/path` and nothing else in the app has to change.
 */

import type { PathResult } from "@shared/types";

import coldFixture from "../data/path_attention_cold.json";
import knownFixture from "../data/path_attention_known.json";
import searchOnlyFixture from "../data/path_search_only.json";

export type Mode = PathResult["mode"];

/**
 * Concepts offered in the "I already know…" checklist. Every entry is a real
 * node in the concept graph fixture.
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
 * Ticking any of these prunes the foundational half of the path, which is the
 * difference between the cold fixture (7 clips) and the pruned one (4 clips).
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

/**
 * Resolve a learning path.
 *
 * @param query   free-text target concept, e.g. "attention"
 * @param known   concepts the learner has ticked as already understood
 * @param mode    "graph" (prerequisite-ordered) or "search_only" (keyword hit)
 */
export function getPath(query: string, known: string[], mode: Mode): PathResult {
  const target = query.trim() || COLD.query;

  if (mode === "search_only") {
    return { ...SEARCH_ONLY, query: target, known: [...known] };
  }

  const pruned = known.some((c) => FOUNDATIONAL.has(c));
  const base = pruned ? KNOWN : COLD;

  return { ...base, query: target, known: [...known] };
}

/** The full corpus length, used by the counter's left-hand side. */
export const TOTAL_CORPUS_SECONDS = COLD.total_corpus_seconds;
