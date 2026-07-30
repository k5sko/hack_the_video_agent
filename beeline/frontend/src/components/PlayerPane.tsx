import { useEffect, useRef, useState } from "react";
import type { ClipSegment } from "@shared/types";
import { fmtClock } from "../lib/format";
import { mediaSrc } from "../lib/resolvePath";

interface Props {
  clip: ClipSegment | null;
  clipIndex: number;
  totalClips: number;
  playing: boolean;
  /** Bumped whenever the player must restart the clip even if the id is unchanged. */
  seekNonce: number;
  finished: boolean;
  nextClip: ClipSegment | null;
  onTogglePlay: () => void;
  onAdvance: () => void;
  onRestartClip: () => void;
  onTime: (seconds: number) => void;
}

/**
 * Plays pre-cut clip files rather than seeking inside an embedded YouTube player.
 *
 * Each clip is its own MP4 served by the API, so the whole surface is ours: no
 * YouTube chrome, no related-video grid on pause, no ads, and — the part that
 * actually matters for a live demo — no network dependency at all.
 *
 * It also makes the boundary logic trivial. The IFrame version had to poll
 * currentTime every 250ms and guess when a seek had landed; here the file simply
 * ends, and `onEnded` fires. The next clip is preloaded in a second, hidden
 * <video> so the cut between them is instant.
 */
export default function PlayerPane({
  clip,
  clipIndex,
  totalClips,
  playing,
  seekNonce,
  finished,
  nextClip,
  onTogglePlay,
  onAdvance,
  onRestartClip,
  onTime,
}: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const preloadRef = useRef<HTMLVideoElement>(null);
  const onAdvanceRef = useRef(onAdvance);
  onAdvanceRef.current = onAdvance;
  const onTimeRef = useRef(onTime);
  onTimeRef.current = onTime;

  const [elapsed, setElapsed] = useState(0);
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);

  const src = clip ? mediaSrc(clip) : null;

  // Load the clip whenever it changes (or a replay is requested).
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !src) return;
    setReady(false);
    setFailed(null);
    setElapsed(0);
    video.src = src;
    video.load();
  }, [src, seekNonce]);

  // Play/pause follows the parent's intent, not the element's own controls.
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !src) return;
    if (playing) {
      video.play().catch(() => {
        /* autoplay refused until a user gesture; the Play button provides it */
      });
    } else {
      video.pause();
    }
  }, [playing, src, ready]);

  // Warm the next file so the transition has nothing to wait for.
  useEffect(() => {
    const pre = preloadRef.current;
    if (!pre || !nextClip) return;
    pre.src = mediaSrc(nextClip);
    pre.load();
  }, [nextClip]);

  const clipLength = clip ? clip.end_seconds - clip.start_seconds : 0;
  const pct = clipLength > 0 ? Math.min(100, (elapsed / clipLength) * 100) : 0;

  return (
    <section className="player">
      <div className="player-stage">
        <video
          ref={videoRef}
          className="player-video"
          playsInline
          preload="auto"
          onCanPlay={() => setReady(true)}
          onTimeUpdate={(e) => {
            const t = e.currentTarget.currentTime;
            setElapsed(t);
            // Report the position on the ORIGINAL video's timeline, since the
            // rest of the app reasons in corpus time, not clip time.
            if (clip) onTimeRef.current(clip.start_seconds + t);
          }}
          onEnded={() => onAdvanceRef.current()}
          onError={() =>
            setFailed(
              clip
                ? `Could not load ${clip.clip_id}.mp4 — has it been cut yet?`
                : "Could not load clip",
            )
          }
        />
        <video ref={preloadRef} className="player-preload" preload="auto" muted />

        {clip && !src && (
          <div className="player-veil player-veil-error">
            <span className="player-veil-label">
              Sample path — the live service is unreachable, so there is no video
              for this clip.
            </span>
          </div>
        )}
        {src && failed && (
          <div className="player-veil player-veil-error">
            <span className="player-veil-label">{failed}</span>
          </div>
        )}
        {src && !failed && !ready && (
          <div className="player-veil">
            <span className="player-veil-label">
              {clip ? `Loading: ${clip.covers.join(", ")}` : "Loading"}
            </span>
          </div>
        )}
        {finished && (
          <div className="player-veil player-veil-done">
            <span className="player-veil-label">Path complete</span>
          </div>
        )}
      </div>

      <div className="transport">
        <button
          type="button"
          className="transport-play"
          onClick={onTogglePlay}
          disabled={!clip}
        >
          {playing ? "Pause" : finished ? "Replay path" : "Play path"}
        </button>

        <div className="transport-progress" aria-hidden>
          <div className="transport-bar">
            <div className="transport-fill" style={{ width: `${pct}%` }} />
          </div>
        </div>

        <span className="transport-time">
          {fmtClock(elapsed)} / {fmtClock(clipLength)}
        </span>

        <span className="transport-count">
          clip {Math.min(clipIndex + 1, totalClips)} of {totalClips}
        </span>

        <button
          type="button"
          className="transport-ghost"
          onClick={onRestartClip}
          disabled={!clip}
        >
          Restart clip
        </button>
        <button
          type="button"
          className="transport-ghost"
          onClick={onAdvance}
          disabled={!clip || clipIndex >= totalClips - 1}
        >
          Next clip
        </button>
      </div>

      <p className="transport-note">
        {clip?.source === "external" ? (
          <>
            Filling a gap this corpus never teaches — sourced from{" "}
            <a href={clip.youtube_url} target="_blank" rel="noreferrer">
              another lecture
            </a>
            .
          </>
        ) : (
          <>
            Each clip is cut to length and ends at its own boundary, then the next
            one starts. No scrubbing, no hunting.
          </>
        )}
      </p>
    </section>
  );
}
