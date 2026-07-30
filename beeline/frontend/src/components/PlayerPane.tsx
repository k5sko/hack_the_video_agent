import { useEffect, useRef, useState } from "react";
import type { ClipSegment } from "@shared/types";
import { fmtClock, youtubeId } from "../lib/format";
import { loadYouTubeApi, type YTPlayer } from "../lib/youtube";

interface Props {
  clip: ClipSegment | null;
  clipIndex: number;
  totalClips: number;
  playing: boolean;
  /** Bumped whenever the player must re-seek even if the clip id is unchanged. */
  seekNonce: number;
  finished: boolean;
  onTogglePlay: () => void;
  onAdvance: () => void;
  onRestartClip: () => void;
  onTime: (seconds: number) => void;
}

export default function PlayerPane({
  clip,
  clipIndex,
  totalClips,
  playing,
  seekNonce,
  finished,
  onTogglePlay,
  onAdvance,
  onRestartClip,
  onTime,
}: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const playerRef = useRef<YTPlayer | null>(null);
  const readyRef = useRef(false);
  const loadedVideoRef = useRef<string | null>(null);
  /** Ignore the boundary check until playback lands inside the clip window. */
  const armedRef = useRef(false);
  const firedRef = useRef(false);

  const clipRef = useRef(clip);
  clipRef.current = clip;
  const playingRef = useRef(playing);
  playingRef.current = playing;
  const onAdvanceRef = useRef(onAdvance);
  onAdvanceRef.current = onAdvance;
  const onTimeRef = useRef(onTime);
  onTimeRef.current = onTime;

  const [ready, setReady] = useState(false);
  const [switching, setSwitching] = useState(false);
  const [elapsed, setElapsed] = useState(0);

  // ---- boot the IFrame player exactly once -------------------------------
  useEffect(() => {
    let cancelled = false;
    const host = hostRef.current;
    if (!host) return;

    // YT replaces the element it is given, so hand it a node React never owns.
    const mount = document.createElement("div");
    host.appendChild(mount);

    loadYouTubeApi().then((YT) => {
      if (cancelled) return;
      const first = clipRef.current;
      playerRef.current = new YT.Player(mount, {
        videoId: first ? youtubeId(first.youtube_url) : undefined,
        playerVars: {
          controls: 0,
          disablekb: 1,
          modestbranding: 1,
          rel: 0,
          fs: 0,
          playsinline: 1,
          iv_load_policy: 3,
          start: first ? Math.floor(first.start_seconds) : 0,
        },
        events: {
          onReady: () => {
            readyRef.current = true;
            loadedVideoRef.current = first?.video_id ?? null;
            setReady(true);
            if (playingRef.current) playerRef.current?.playVideo();
          },
        },
      });
    });

    return () => {
      cancelled = true;
      try {
        playerRef.current?.destroy();
      } catch {
        /* player may already be gone */
      }
      playerRef.current = null;
      readyRef.current = false;
    };
  }, []);

  // ---- move to the active clip ------------------------------------------
  useEffect(() => {
    const player = playerRef.current;
    const active = clip;
    armedRef.current = false;
    firedRef.current = false;
    setElapsed(0);
    if (!player || !readyRef.current || !active) return;

    const videoId = youtubeId(active.youtube_url);

    if (loadedVideoRef.current !== active.video_id) {
      // Different source video: mask the reload so the seam is not visible.
      setSwitching(true);
      loadedVideoRef.current = active.video_id;
      if (playingRef.current) {
        player.loadVideoById({ videoId, startSeconds: active.start_seconds });
      } else {
        player.cueVideoById({ videoId, startSeconds: active.start_seconds });
      }
      const timer = window.setTimeout(() => setSwitching(false), 650);
      return () => window.clearTimeout(timer);
    }

    // Same source video: a plain seek, which is genuinely seamless.
    player.seekTo(active.start_seconds, true);
    if (playingRef.current) player.playVideo();
    return;
  }, [clip, seekNonce, ready]);

  // ---- play / pause ------------------------------------------------------
  useEffect(() => {
    const player = playerRef.current;
    if (!player || !readyRef.current) return;
    if (playing) player.playVideo();
    else player.pauseVideo();
  }, [playing, ready]);

  // ---- hard stop at end_seconds -----------------------------------------
  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      const player = playerRef.current;
      const active = clipRef.current;
      if (!player || !readyRef.current || !active) return;

      let t: number;
      try {
        t = player.getCurrentTime();
      } catch {
        return;
      }

      if (!armedRef.current) {
        // Wait until the head is genuinely inside this clip. Guards against
        // reading the previous video's clock during a loadVideoById.
        if (t >= active.start_seconds - 1.5 && t < active.end_seconds) {
          armedRef.current = true;
        } else {
          return;
        }
      }

      setElapsed(Math.max(0, t - active.start_seconds));
      onTimeRef.current(t);

      if (t >= active.end_seconds - 0.2) {
        if (firedRef.current) return;
        firedRef.current = true;
        onAdvanceRef.current();
      } else if (t < active.start_seconds - 2) {
        player.seekTo(active.start_seconds, true);
      }
    }, 250);
    return () => window.clearInterval(id);
  }, [playing, clip, seekNonce]);

  const clipLength = clip ? clip.end_seconds - clip.start_seconds : 0;
  const pct = clipLength > 0 ? Math.min(100, (elapsed / clipLength) * 100) : 0;

  return (
    <section className="player">
      <div className="player-stage">
        <div className="player-host" ref={hostRef} />
        {switching && (
          <div className="player-veil">
            <span className="player-veil-label">
              {clip ? `Next: ${clip.covers.join(", ")}` : "Loading"}
            </span>
          </div>
        )}
        {finished && (
          <div className="player-veil player-veil-done">
            <span className="player-veil-label">Path complete</span>
          </div>
        )}
        {!ready && <div className="player-veil" />}
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
        Playback is clipped to {clip ? fmtClock(clipLength) : "0:00"} and stops
        hard at the segment boundary. No scrubbing, no hunting.
      </p>
    </section>
  );
}
