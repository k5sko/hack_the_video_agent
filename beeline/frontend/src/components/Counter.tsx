import { useEffect, useRef, useState } from "react";
import { fmtBig } from "../lib/format";

/** Eases a number towards a new target so the counter visibly counts down. */
function useAnimatedNumber(target: number, duration = 850): number {
  const [value, setValue] = useState(target);
  const fromRef = useRef(target);

  useEffect(() => {
    const from = fromRef.current;
    if (from === target) return;
    const started = performance.now();
    let raf = 0;

    const step = (now: number) => {
      const p = Math.min(1, (now - started) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      const next = from + (target - from) * eased;
      fromRef.current = next;
      setValue(next);
      if (p < 1) raf = requestAnimationFrame(step);
      else fromRef.current = target;
    };

    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [target, duration]);

  return value;
}

interface Props {
  totalCorpusSeconds: number;
  watchSeconds: number;
  searchOnly: boolean;
  /** Clips search-only can order, out of the clips the graph path would give. */
  coverage: { have: number; of: number };
}

export default function Counter({
  totalCorpusSeconds,
  watchSeconds,
  searchOnly,
  coverage,
}: Props) {
  const animated = useAnimatedNumber(watchSeconds);
  const saved = Math.max(0, 1 - watchSeconds / totalCorpusSeconds);

  if (searchOnly) {
    return (
      <section className="counter counter-degraded">
        <div className="counter-line">
          <span className="counter-degraded-value">
            Coverage: {coverage.have}/{coverage.of}
          </span>
          <span className="counter-degraded-note">prerequisites unknown</span>
        </div>
        <p className="counter-caption">
          Without a prerequisite graph there is no watch time to compute — only a
          timestamp.
        </p>
      </section>
    );
  }

  return (
    <section className="counter">
      <div className="counter-line">
        <span className="counter-from">{fmtBig(totalCorpusSeconds)}</span>
        <span className="counter-arrow" aria-hidden>
          →
        </span>
        <span className="counter-to">{fmtBig(animated)}</span>
      </div>
      <p className="counter-caption">
        Full corpus vs. your path — {Math.round(saved * 100)}% of the lectures
        skipped, order guaranteed.
      </p>
    </section>
  );
}
