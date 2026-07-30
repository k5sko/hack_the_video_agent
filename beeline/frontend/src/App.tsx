import { useCallback, useEffect, useMemo, useState } from "react";
import type { NodeState } from "@shared/types";
import Sidebar from "./components/Sidebar";
import PlayerPane from "./components/PlayerPane";
import Counter from "./components/Counter";
import ConceptGraph, { NODE_COLORS } from "./components/ConceptGraph";
import EvidenceCard from "./components/EvidenceCard";
import { getPath, type Mode } from "./lib/resolvePath";
import type { PathResult } from "@shared/types";
import { clipIndexForConcept, deriveNodeStates } from "./lib/graph";
import coldFixture from "./data/path_attention_cold.json";

const LEGEND: { state: NodeState; label: string }[] = [
  { state: "on_path", label: "on path" },
  { state: "known", label: "known" },
  { state: "not_needed", label: "not needed" },
  { state: "gap", label: "gap" },
];

export default function App() {
  const [queryInput, setQueryInput] = useState("attention");
  const [query, setQuery] = useState("attention");
  const [known, setKnown] = useState<string[]>([]);
  const [mode, setMode] = useState<Mode>("graph");

  const [path, setPath] = useState<PathResult>(coldFixture as PathResult);
  // What the graph would have produced for the same inputs — the yardstick the
  // search-only coverage number is measured against.
  const [graphPath, setGraphPath] = useState<PathResult>(coldFixture as PathResult);
  const [loading, setLoading] = useState(false);
  const [offline, setOffline] = useState(false);

  const knownKey = known.slice().sort().join(",");

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    // Both modes are fetched together so flipping the kill-shot toggle is
    // instant and the coverage fraction always has a real denominator.
    Promise.all([
      getPath(query, known, mode),
      mode === "graph"
        ? null
        : getPath(query, known, "graph").then((outcome) => outcome.path),
    ])
      .then(([outcome, comparison]) => {
        if (cancelled) return;
        setPath(outcome.path);
        setGraphPath(comparison ?? outcome.path);
        setOffline(outcome.offline);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // knownKey rather than `known`: a new array with the same contents is not a
    // reason to refetch.
  }, [query, knownKey, mode]);

  const pathKey = `${path.mode}::${path.playlist.map((c) => c.clip_id).join(",")}`;

  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [finished, setFinished] = useState(false);
  const [seekNonce, setSeekNonce] = useState(0);
  const [selected, setSelected] = useState<string | null>(null);

  // Any change to the resolved path restarts the run from step one.
  useEffect(() => {
    setIndex(0);
    setFinished(false);
    setSeekNonce((n) => n + 1);
  }, [pathKey]);

  const clip = path.playlist[index] ?? null;
  const states = useMemo(() => deriveNodeStates(path), [path]);

  const currentConcepts = useMemo(
    () => (playing && !finished && clip ? clip.covers : []),
    [playing, finished, clip],
  );

  const completedConcepts = useMemo(() => {
    const upto = finished ? path.playlist.length : index;
    return path.playlist.slice(0, upto).flatMap((c) => c.covers);
  }, [path, index, finished]);

  const advance = useCallback(() => {
    setIndex((i) => {
      if (i + 1 < path.playlist.length) return i + 1;
      setFinished(true);
      setPlaying(false);
      return i;
    });
  }, [path.playlist.length]);

  const togglePlay = useCallback(() => {
    if (finished) {
      setFinished(false);
      setIndex(0);
      setSeekNonce((n) => n + 1);
      setPlaying(true);
      return;
    }
    setPlaying((p) => !p);
  }, [finished]);

  const selectClip = useCallback((i: number) => {
    setFinished(false);
    setIndex(i);
    setSeekNonce((n) => n + 1);
    setPlaying(true);
  }, []);

  const toggleKnown = useCallback((concept: string) => {
    setKnown((prev) =>
      prev.includes(concept) ? prev.filter((c) => c !== concept) : [...prev, concept],
    );
  }, []);

  const onSelectConcept = useCallback(
    (concept: string) => {
      setSelected(concept);
      const target = clipIndexForConcept(path, concept);
      if (target >= 0) selectClip(target);
    },
    [path, selectClip],
  );

  const selectedState: NodeState = selected
    ? (states.get(selected) ?? "not_needed")
    : "not_needed";
  const selectedPosition = selected ? clipIndexForConcept(path, selected) : -1;

  const searchOnly = path.mode === "search_only";

  return (
    <div className="app">
      <Sidebar
        queryInput={queryInput}
        onQueryInput={setQueryInput}
        onSubmit={() => setQuery(queryInput.trim() || "attention")}
        mode={mode}
        onMode={setMode}
        known={known}
        onToggleKnown={toggleKnown}
        path={path}
        currentIndex={index}
        finished={finished}
        onSelectClip={selectClip}
      />

      <main className="main">
        <Counter
          totalCorpusSeconds={path.total_corpus_seconds}
          watchSeconds={path.watch_seconds}
          searchOnly={searchOnly}
          coverage={{ have: path.playlist.length, of: graphPath.playlist.length }}
        />

        <div className="why" data-degraded={searchOnly}>
          <span className="why-now">Now</span>
          {clip ? (
            <p className="why-text">
              <strong>{clip.covers.join(", ")}</strong> — {clip.why}
            </p>
          ) : (
            <p className="why-text">Nothing queued.</p>
          )}
        </div>

        <PlayerPane
          clip={clip}
          clipIndex={index}
          totalClips={path.playlist.length}
          playing={playing}
          seekNonce={seekNonce}
          finished={finished}
          nextClip={path.playlist[index + 1] ?? null}
          onTogglePlay={togglePlay}
          onAdvance={advance}
          onRestartClip={() => setSeekNonce((n) => n + 1)}
          onTime={() => undefined}
        />

        <p className="narration">
          {loading ? "Building your path…" : path.narration}
          {offline && !loading && (
            <span className="narration-flag"> · offline sample data</span>
          )}
        </p>
      </main>

      <section className="panel">
        <header className="panel-head">
          <h2 className="panel-title">Concept graph</h2>
          <span className="panel-sub">
            {searchOnly
              ? "no prerequisite structure available"
              : `target: ${path.target_concepts.join(", ")}`}
          </span>
        </header>

        <div className="legend">
          {LEGEND.map((item) => (
            <span className="legend-item" key={item.state}>
              <span
                className="legend-dot"
                data-state={item.state}
                style={{ background: NODE_COLORS[item.state] }}
              />
              {item.label}
            </span>
          ))}
        </div>

        <div className="panel-body">
          <ConceptGraph
            states={states}
            currentConcepts={currentConcepts}
            completedConcepts={completedConcepts}
            selected={selected}
            onSelect={onSelectConcept}
          />

          {path.gaps.length > 0 && !selected && (
            <div className="gap-banner">
              <strong>{path.gaps.join(", ")}</strong> — required, but never
              explained in this corpus.
            </div>
          )}

          {selected && (
            <EvidenceCard
              concept={selected}
              state={selectedState}
              position={selectedPosition >= 0 ? selectedPosition + 1 : null}
              playlistLength={path.playlist.length}
              onJump={() => selectedPosition >= 0 && selectClip(selectedPosition)}
              onClose={() => setSelected(null)}
            />
          )}
        </div>
      </section>
    </div>
  );
}
