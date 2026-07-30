/**
 * Concept-graph fixture access + derivation of per-node state from the active
 * PathResult. The graph topology is static (it is the corpus); only the node
 * *states* change as the query, known-set and mode change.
 */

import type { NodeState, PathResult } from "@shared/types";
import rawGraph from "../data/concept_graph.json";

export type ConceptEvidence = {
  video_id: string;
  video_title: string;
  youtube_url: string;
  start_seconds: number;
  end_seconds: number;
  /** How strongly this segment is judged to EXPLAIN the concept, 0–1. */
  explains_score: number;
}

export type ConceptNode = {
  id: string;
  evidence: ConceptEvidence | null;
  /** Concepts this one REQUIRES (outgoing edges). */
  assumes: string[];
  /** Concepts that require this one (incoming edges). */
  requiredBy: string[];
  // Mutated in place by the force simulation.
  x?: number;
  y?: number;
}

export type ConceptLink = {
  source: string | ConceptNode;
  target: string | ConceptNode;
}

type RawConcept = {
  id: string;
  evidence: ConceptEvidence | null;
}

export type RawGraph = {
  corpus_name: string;
  concepts: RawConcept[];
  requires: { from: string; to: string }[];
}

/**
 * The corpus topology. Starts as the bundled fixture so the app renders with no
 * backend, and is replaced by the real graph from /api/graph once it arrives —
 * the fixture describes 38 concepts where the live corpus has 100, and this
 * panel is meant to be the evidence for the path, not a decoration.
 */
let RAW = rawGraph as unknown as RawGraph;

export function setGraphSource(graph: RawGraph): void {
  RAW = graph;
}

export function conceptCount(): number {
  return RAW.concepts.length;
}

export function edgeCount(): number {
  return RAW.requires.length;
}

/**
 * Build a fresh, mutable graph-data object. Call ONCE per mount and keep the
 * identity stable — react-force-graph mutates the nodes with x/y and restarts
 * the simulation whenever the object identity changes.
 */
export function buildGraphData(): { nodes: ConceptNode[]; links: ConceptLink[] } {
  const assumes = new Map<string, string[]>();
  const requiredBy = new Map<string, string[]>();

  for (const edge of RAW.requires) {
    if (!assumes.has(edge.from)) assumes.set(edge.from, []);
    assumes.get(edge.from)!.push(edge.to);
    if (!requiredBy.has(edge.to)) requiredBy.set(edge.to, []);
    requiredBy.get(edge.to)!.push(edge.from);
  }

  const nodes: ConceptNode[] = RAW.concepts.map((c) => ({
    id: c.id,
    evidence: c.evidence,
    assumes: assumes.get(c.id) ?? [],
    requiredBy: requiredBy.get(c.id) ?? [],
  }));

  const known = new Set(nodes.map((n) => n.id));
  const links: ConceptLink[] = RAW.requires
    .filter((e) => known.has(e.from) && known.has(e.to))
    .map((e) => ({ source: e.from, target: e.to }));

  return { nodes, links };
}

/** Static lookup of concept metadata by id (never mutated). */
export const CONCEPT_INDEX: ReadonlyMap<string, ConceptNode> = (() => {
  const built = buildGraphData();
  return new Map(built.nodes.map((n) => [n.id, n]));
})();

/**
 * Derive the state of every node from the active path.
 *
 * In search_only mode there is no prerequisite structure to render, so exactly
 * one node — the keyword hit — is highlighted and everything else goes flat.
 */
export function deriveNodeStates(path: PathResult): Map<string, NodeState> {
  const states = new Map<string, NodeState>();
  for (const node of CONCEPT_INDEX.keys()) states.set(node, "not_needed");

  if (path.mode === "search_only") {
    const hit = path.playlist[0]?.covers[0] ?? path.target_concepts[0];
    if (hit && states.has(hit)) states.set(hit, "on_path");
    return states;
  }

  for (const c of path.known) if (states.has(c)) states.set(c, "known");
  for (const c of path.needed_concepts) if (states.has(c)) states.set(c, "on_path");
  for (const clip of path.playlist) {
    for (const c of clip.covers) if (states.has(c)) states.set(c, "on_path");
  }
  for (const c of path.target_concepts) if (states.has(c)) states.set(c, "on_path");
  // Gaps win over everything: they are required but unteachable from this corpus.
  for (const c of path.gaps) if (states.has(c)) states.set(c, "gap");

  return states;
}

/** Index of the playlist clip that covers a concept, or -1. */
export function clipIndexForConcept(path: PathResult, concept: string): number {
  return path.playlist.findIndex((clip) => clip.covers.includes(concept));
}
