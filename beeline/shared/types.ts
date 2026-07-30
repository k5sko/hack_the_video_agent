// Beeline shared response contract.
// Every layer codes against this file. Do not change it without telling everyone.
// Python mirror lives in shared/types.py — keep the two in sync.

export type NodeState = "on_path" | "known" | "not_needed" | "gap";

export interface ClipSegment {
  clip_id: string;
  video_id: string;
  video_title: string;
  /** Source video, kept for attribution and "open the original" links. */
  youtube_url: string;
  /**
   * The clip as its own file, served locally (e.g. /media/v6_c2.mp4). This is
   * what the player actually plays: it keeps YouTube's interface off the screen
   * and lets the whole demo run with no network.
   */
  media_url: string;
  start_seconds: number;
  end_seconds: number;
  covers: string[];
  why: string;
  /**
   * "corpus" — cut from the indexed lecture series.
   * "external" — found elsewhere to fill a gap the corpus never teaches.
   */
  source: ClipSource;
}

export type ClipSource = "corpus" | "external";

export interface PathResult {
  query: string;
  mode: "graph" | "search_only";
  target_concepts: string[];
  known: string[];
  needed_concepts: string[];
  playlist: ClipSegment[];
  gaps: string[];
  total_corpus_seconds: number;
  watch_seconds: number;
  narration: string;
}
