import type { FormEvent } from "react";
import type { PathResult } from "@shared/types";
import { fmtBig, fmtClock } from "../lib/format";
import { KNOWABLE_CONCEPTS, type Mode } from "../lib/resolvePath";
import { CONCEPT_COUNT, EDGE_COUNT } from "../lib/graph";

interface Props {
  queryInput: string;
  onQueryInput: (value: string) => void;
  onSubmit: () => void;
  mode: Mode;
  onMode: (mode: Mode) => void;
  known: string[];
  onToggleKnown: (concept: string) => void;
  path: PathResult;
  currentIndex: number;
  finished: boolean;
  onSelectClip: (index: number) => void;
}

export default function Sidebar({
  queryInput,
  onQueryInput,
  onSubmit,
  mode,
  onMode,
  known,
  onToggleKnown,
  path,
  currentIndex,
  finished,
  onSelectClip,
}: Props) {
  const submit = (e: FormEvent) => {
    e.preventDefault();
    onSubmit();
  };

  const status = (i: number): "done" | "playing" | "upcoming" => {
    if (finished || i < currentIndex) return "done";
    if (i === currentIndex) return "playing";
    return "upcoming";
  };

  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark" aria-hidden />
        <span className="brand-name">Beeline</span>
      </div>
      <p className="brand-tag">The shortest path to understanding.</p>

      <div className="corpus">
        <span className="corpus-title">Neural Networks — 7 lectures · 2h 20m</span>
        <span className="corpus-sub">
          {CONCEPT_COUNT} concepts · {EDGE_COUNT} prerequisite edges
        </span>
      </div>

      <form className="block" onSubmit={submit}>
        <label className="label" htmlFor="query">
          What do you want to understand?
        </label>
        <input
          id="query"
          className="input"
          value={queryInput}
          placeholder="attention"
          onChange={(e) => onQueryInput(e.target.value)}
          autoComplete="off"
        />
        <button type="submit" className="btn-primary">
          Build my path
        </button>
      </form>

      <div className="block">
        <span className="label">Mode</span>
        <div className="toggle" role="group" aria-label="Mode">
          <button
            type="button"
            className="toggle-option"
            data-active={mode === "graph"}
            onClick={() => onMode("graph")}
          >
            Graph path
          </button>
          <button
            type="button"
            className="toggle-option"
            data-active={mode === "search_only"}
            onClick={() => onMode("search_only")}
          >
            Search only
          </button>
        </div>
      </div>

      <div className="block">
        <span className="label">I already know…</span>
        <ul className="checklist">
          {KNOWABLE_CONCEPTS.map((concept) => (
            <li key={concept}>
              <label className="check">
                <input
                  type="checkbox"
                  checked={known.includes(concept)}
                  onChange={() => onToggleKnown(concept)}
                />
                <span>{concept}</span>
              </label>
            </li>
          ))}
        </ul>
      </div>

      <div className="block block-grow">
        <div className="label-row">
          <span className="label">Your playlist</span>
          <span className="label-value">
            {path.playlist.length} clip{path.playlist.length === 1 ? "" : "s"} ·{" "}
            {fmtBig(path.watch_seconds)}
          </span>
        </div>

        <ol className="playlist">
          {path.playlist.map((clip, i) => (
            <li key={clip.clip_id}>
              <button
                type="button"
                className="clip"
                data-status={status(i)}
                onClick={() => onSelectClip(i)}
              >
                <span className="clip-dot" data-status={status(i)} aria-hidden />
                <span className="clip-body">
                  <span className="clip-head">
                    <span className="clip-index">{i + 1}</span>
                    <span className="clip-title">{clip.video_title}</span>
                    <span className="clip-dur">
                      {fmtClock(clip.end_seconds - clip.start_seconds)}
                    </span>
                  </span>
                  <span className="clip-covers">
                    {clip.covers.map((c) => (
                      <span className="chip chip-covers" key={c}>
                        {c}
                      </span>
                    ))}
                  </span>
                </span>
              </button>
            </li>
          ))}
        </ol>

        {path.mode === "search_only" && (
          <p className="playlist-note">
            One keyword hit. Nothing before it, nothing after it.
          </p>
        )}
      </div>
    </aside>
  );
}
