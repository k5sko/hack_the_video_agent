import type { NodeState } from "@shared/types";
import { fmtClock, fmtTimestamp } from "../lib/format";
import { CONCEPT_INDEX } from "../lib/graph";

interface Props {
  concept: string;
  state: NodeState;
  /** 1-based position in the current playlist, or null if it is not on it. */
  position: number | null;
  playlistLength: number;
  onJump: () => void;
  onClose: () => void;
}

const STATE_LABEL: Record<NodeState, string> = {
  on_path: "On your path",
  known: "Marked known — pruned",
  not_needed: "Not needed for this target",
  gap: "Gap",
};

export default function EvidenceCard({
  concept,
  state,
  position,
  playlistLength,
  onJump,
  onClose,
}: Props) {
  const node = CONCEPT_INDEX.get(concept);
  const evidence = node?.evidence ?? null;

  return (
    <aside className="evidence" data-state={state}>
      <header className="evidence-head">
        <div>
          <h3 className="evidence-title">{concept}</h3>
          <span className="evidence-state">{STATE_LABEL[state]}</span>
        </div>
        <button type="button" className="evidence-close" onClick={onClose} aria-label="Close">
          ×
        </button>
      </header>

      {state === "gap" || !evidence ? (
        <p className="evidence-gap">Required, but never explained in this corpus.</p>
      ) : (
        <>
          <dl className="evidence-grid">
            <dt>Video</dt>
            <dd>
              {evidence.video_id} · {evidence.video_title}
            </dd>
            <dt>Segment</dt>
            <dd>
              {fmtTimestamp(evidence.start_seconds)} – {fmtTimestamp(evidence.end_seconds)}{" "}
              <span className="evidence-dim">
                ({fmtClock(evidence.end_seconds - evidence.start_seconds)})
              </span>
            </dd>
            <dt>EXPLAINS</dt>
            <dd>
              <span className="evidence-score">
                <span
                  className="evidence-score-fill"
                  style={{ width: `${Math.round(evidence.explains_score * 100)}%` }}
                />
              </span>
              <span className="evidence-dim">{evidence.explains_score.toFixed(2)}</span>
            </dd>
          </dl>

          <div className="evidence-assumes">
            <span className="evidence-assumes-label">Assumes</span>
            {node && node.assumes.length > 0 ? (
              <span className="evidence-chips">
                {node.assumes.map((a) => (
                  <span className="chip" key={a}>
                    {a}
                  </span>
                ))}
              </span>
            ) : (
              <span className="evidence-dim">nothing — this is a root concept</span>
            )}
          </div>
        </>
      )}

      {node && node.requiredBy.length > 0 && (
        <div className="evidence-assumes">
          <span className="evidence-assumes-label">Required by</span>
          <span className="evidence-chips">
            {node.requiredBy.map((a) => (
              <span className="chip" key={a}>
                {a}
              </span>
            ))}
          </span>
        </div>
      )}

      <footer className="evidence-foot">
        {position !== null ? (
          <>
            <span className="evidence-dim">
              Step {position} of {playlistLength} in your path
            </span>
            <button type="button" className="btn-small" onClick={onJump}>
              Play this clip
            </button>
          </>
        ) : (
          <span className="evidence-dim">Not in your current path.</span>
        )}
      </footer>
    </aside>
  );
}
