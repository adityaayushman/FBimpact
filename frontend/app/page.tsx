"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  API_CONFIGURED,
  API_URL,
  type Analysis,
  type ClipSummary,
  type Faithfulness,
  type Health,
  type Keypoints,
  type SkeletonSpec,
  type Warning,
} from "@/lib/api";
import FaithfulnessChart from "@/components/FaithfulnessChart";
import ScoreChart from "@/components/ScoreChart";
import SkeletonCanvas from "@/components/SkeletonCanvas";
import VideoUpload from "@/components/VideoUpload";

interface Source {
  id: string;
  keypoints: Keypoints;
  fps: number;
  impactFrame: number | null;
  label: string;
}

export default function Page() {
  const [health, setHealth] = useState<Health | null>(null);
  const [skeleton, setSkeleton] = useState<SkeletonSpec | null>(null);
  const [clips, setClips] = useState<ClipSummary[]>([]);
  const [source, setSource] = useState<Source | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [faithfulness, setFaithfulness] = useState<Record<number, Faithfulness>>({});
  const [checking, setChecking] = useState<number | null>(null);

  // -- bootstrap -------------------------------------------------------------

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [h, s, c] = await Promise.all([api.health(), api.skeleton(), api.clips()]);
        if (cancelled) return;
        setHealth(h);
        setSkeleton(s);
        setClips(c);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // -- playback --------------------------------------------------------------

  const rafRef = useRef<number | null>(null);
  const lastTickRef = useRef<number>(0);

  useEffect(() => {
    if (!playing || !source) return;
    const interval = 1000 / source.fps;
    const total = source.keypoints.length;

    const tick = (now: number) => {
      if (now - lastTickRef.current >= interval) {
        lastTickRef.current = now;
        setFrame((f) => {
          if (f + 1 >= total) {
            setPlaying(false);
            return total - 1;
          }
          return f + 1;
        });
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [playing, source]);

  // -- actions ---------------------------------------------------------------

  const run = useCallback(
    async (next: Source) => {
      setSource(next);
      setAnalysis(null);
      setFaithfulness({});
      setFrame(0);
      setPlaying(false);
      setError(null);
      setBusy("Scoring every frame…");
      try {
        const result = await api.analyze({
          keypoints: next.keypoints,
          fps: next.fps,
          impact_frame: next.impactFrame,
          clip_id: next.id,
          explain: true,
        });
        setAnalysis(result);
        // Jump to the first warning: it is the thing worth looking at, and
        // starting at frame 0 makes the user hunt for it.
        if (result.warnings.length) {
          setFrame(result.warnings[0].frame);
        } else {
          setPlaying(true);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(null);
      }
    },
    []
  );

  const loadClip = useCallback(
    async (summary: ClipSummary) => {
      setBusy(`Loading ${summary.clip_id}…`);
      setError(null);
      try {
        const payload = await api.clip(summary.clip_id);
        await run({
          id: payload.clip_id,
          keypoints: payload.keypoints,
          fps: payload.fps,
          impactFrame: payload.impact_frame,
          label: `${payload.activity} · ${payload.label === "fall" ? "fall" : "normal activity"}`,
        });
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setBusy(null);
      }
    },
    [run]
  );

  const verify = useCallback(
    async (warning: Warning) => {
      if (!source) return;
      setChecking(warning.frame);
      setError(null);
      try {
        const result = await api.faithfulness({
          keypoints: source.keypoints,
          frame: warning.frame,
          fps: source.fps,
          impact_frame: source.impactFrame,
          num_random: 3,
          clip_id: source.id,
        });
        setFaithfulness((prev) => ({ ...prev, [warning.frame]: result }));
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setChecking(null);
      }
    },
    [source]
  );

  // -- derived ---------------------------------------------------------------

  const activeWarning = useMemo(() => {
    if (!analysis) return null;
    // Show evidence for the warning whose window the playhead is inside.
    return (
      analysis.warnings.find((w) => frame >= w.frame && frame < w.frame + 12) ?? null
    );
  }, [analysis, frame]);

  const score = analysis?.scores[Math.min(frame, analysis.scores.length - 1)] ?? 0;
  const falls = clips.filter((c) => c.label === "fall");
  const adls = clips.filter((c) => c.label === "adl");

  return (
    <div className="shell">
      <header className="masthead">
        <div className="masthead-row">
          <div>
            <h1>Pre-Impact Fall Anticipation with Grounded Skeletal Evidence</h1>
            <p>
              Predicts a fall <em>before impact</em> from skeleton data alone, and names the joints
              whose instability signalled it — then tests whether that evidence is actually faithful.
            </p>
          </div>
          <div className="status">
            <span className={`status-dot ${health ? "ok" : error ? "error" : ""}`} />
            {health ? (
              <span style={{ fontFamily: "var(--mono)", fontSize: 12 }}>
                {health.model.backbone} · {health.model.parameters.toLocaleString()} params ·{" "}
                {health.model.causal ? "causal" : "NON-CAUSAL"} · {health.model.device}
              </span>
            ) : (
              <span>{error ? "API unreachable" : "connecting…"}</span>
            )}
          </div>
        </div>

        <p className="disclaimer">
          <strong>Research prototype — not a medical device.</strong> The deployed checkpoint is
          trained on a <b>synthetic fixture</b>, not on real fall data, so nothing it outputs carries
          clinical meaning. Public fall datasets use acted falls by young volunteers; this demo
          exists to show the pipeline and the faithfulness test, not to make a safety claim.
        </p>
      </header>

      <div className="grid">
        {/* -- left column ---------------------------------------------------- */}
        <div>
          <section className="card">
            <h2>Demo clips</h2>
            <p className="hint">
              The normal activities are deliberately <em>hard negatives</em> — sitting, bending and
              lying down are all controlled descents. A model that fires on any downward motion
              fails here.
            </p>

            {clips.length === 0 && !error && <div className="empty">Loading…</div>}

            {falls.length > 0 && <div className="group-label">Falls</div>}
            <div className="clip-list">
              {falls.map((c) => (
                <button
                  key={c.clip_id}
                  className="clip"
                  aria-pressed={source?.id === c.clip_id}
                  disabled={!!busy}
                  onClick={() => void loadClip(c)}
                >
                  <span className="clip-dot fall" />
                  <span className="clip-name">{c.activity.replace(/_/g, " ")}</span>
                  <span className="clip-meta">{c.duration_s.toFixed(1)}s</span>
                </button>
              ))}
            </div>

            {adls.length > 0 && <div className="group-label">Normal activity</div>}
            <div className="clip-list">
              {adls.map((c) => (
                <button
                  key={c.clip_id}
                  className="clip"
                  aria-pressed={source?.id === c.clip_id}
                  disabled={!!busy}
                  onClick={() => void loadClip(c)}
                >
                  <span className="clip-dot adl" />
                  <span className="clip-name">{c.activity.replace(/_/g, " ")}</span>
                  <span className="clip-meta">{c.duration_s.toFixed(1)}s</span>
                </button>
              ))}
            </div>
          </section>

          <section className="card">
            <h2>Your own video</h2>
            <VideoUpload
              maxFrames={health?.limits.max_frames ?? 900}
              disabled={!!busy}
              onError={setError}
              onExtracted={({ keypoints, fps, name }) =>
                void run({ id: name, keypoints, fps, impactFrame: null, label: name })
              }
            />
          </section>

          {health && (
            <section className="card">
              <h2>Model</h2>
              <dl className="spec">
                <dt>Window</dt>
                <dd>{health.model.window} frames</dd>
                <dt>Imminent window</dt>
                <dd>{health.model.w_pre_frames} frames</dd>
                <dt>Threshold τ</dt>
                <dd>{health.decision.threshold.toFixed(2)}</dd>
                <dt>Persistence k</dt>
                <dd>{health.decision.persistence}</dd>
                <dt>Features</dt>
                <dd>{health.model.with_velocity ? "x, y, vx, vy" : "x, y"}</dd>
                <dt>Evidence</dt>
                <dd>{health.model.explain_method}</dd>
                <dt>Trained on</dt>
                <dd>{health.model.trained_on}</dd>
              </dl>
              <p className="hint" style={{ marginTop: 10, marginBottom: 0 }}>
                τ and k were chosen on a validation split under a false-alarm budget and frozen into
                the checkpoint — not tuned on what you are looking at.
              </p>
            </section>
          )}
        </div>

        {/* -- right column --------------------------------------------------- */}
        <div>
          {error && (
            <div className="card" style={{ marginBottom: 14 }}>
              <div className="error-box">{error}</div>
              {API_CONFIGURED ? (
                <p className="hint" style={{ marginTop: 10, marginBottom: 0 }}>
                  API: <code style={{ fontFamily: "var(--mono)" }}>{API_URL}</code>
                </p>
              ) : (
                <ol className="hint" style={{ marginTop: 10, marginBottom: 0, paddingLeft: 18 }}>
                  <li>
                    Deploy the API to Render using <code style={{ fontFamily: "var(--mono)" }}>render.yaml</code>.
                  </li>
                  <li>
                    Add <code style={{ fontFamily: "var(--mono)" }}>NEXT_PUBLIC_API_URL</code> to this
                    Vercel project&rsquo;s environment variables.
                  </li>
                  <li>Redeploy — the value is inlined at build time.</li>
                </ol>
              )}
            </div>
          )}

          {!source && !error && (
            <section className="card">
              <div className="empty">Pick a clip on the left, or load a video, to begin.</div>
            </section>
          )}

          {source && skeleton && (
            <section className="card">
              <h2>{source.label}</h2>
              <p className="hint">
                {busy ??
                  (analysis
                    ? `${analysis.frames} frames scored in ${analysis.latency_ms.toFixed(0)} ms. ` +
                      (analysis.warnings.length
                        ? `${analysis.warnings.length} warning${analysis.warnings.length > 1 ? "s" : ""}.`
                        : "No warning fired.")
                    : "")}
              </p>

              <div className="viz-row">
                <div>
                  <SkeletonCanvas
                    keypoints={source.keypoints}
                    skeleton={skeleton}
                    frame={frame}
                    relevance={activeWarning?.evidence?.relevance ?? null}
                    score={score}
                    threshold={analysis?.decision.threshold ?? 0.7}
                  />
                  {activeWarning && (
                    <p style={{ margin: "6px 2px 0", fontSize: 12, color: "var(--text-muted)", textAlign: "center" }}>
                      Highlighted joints are the evidence for this warning.
                    </p>
                  )}
                </div>

                <div>
                  {analysis && <ScoreChart analysis={analysis} frame={frame} onScrub={setFrame} />}
                  <div className="legend">
                    <span>
                      <i className="swatch" style={{ background: "var(--accent)" }} /> p(fall imminent)
                    </span>
                    <span>
                      <i className="swatch" style={{ background: "var(--imminent)" }} /> threshold &amp; triggers
                    </span>
                    {analysis?.imminent_window && (
                      <span>
                        <i className="swatch" style={{ background: "var(--accent)", opacity: 0.35, height: 9 }} />{" "}
                        imminent window
                      </span>
                    )}
                    {analysis?.impact_frame !== null && analysis && (
                      <span>
                        <i className="swatch" style={{ background: "var(--impact)", width: 3, height: 11 }} /> impact
                      </span>
                    )}
                  </div>
                </div>
              </div>

              <div className="transport">
                <button className="btn" onClick={() => setPlaying((p) => !p)} disabled={!analysis}>
                  {playing ? "Pause" : "Play"}
                </button>
                <input
                  type="range"
                  min={0}
                  max={Math.max(source.keypoints.length - 1, 0)}
                  value={frame}
                  onChange={(e) => {
                    setPlaying(false);
                    setFrame(Number(e.target.value));
                  }}
                  aria-label="Scrub through the clip"
                />
                <span className="frame-readout">
                  {frame}/{source.keypoints.length - 1} · {(frame / source.fps).toFixed(2)}s · p={score.toFixed(3)}
                </span>
              </div>
            </section>
          )}

          {analysis && (
            <section className="card">
              <h2>Warnings &amp; evidence</h2>

              {analysis.warnings.length === 0 ? (
                <div className="empty">
                  {analysis.impact_frame !== null
                    ? "No warning fired before impact — this clip is a miss."
                    : "No warning fired. For a normal-activity clip, that is the correct outcome."}
                </div>
              ) : (
                analysis.warnings.map((w) => {
                  const outOfWindow = w.within_imminent_window === false;
                  const check = faithfulness[w.frame];
                  return (
                    <div key={w.frame} className={`warning${outOfWindow ? " out-of-window" : ""}`}>
                      <div className="warning-head">
                        {w.lead_time !== null ? (
                          <span className="warning-lead">{w.lead_time.toFixed(2)}s before impact</span>
                        ) : (
                          <span className="warning-lead">warning</span>
                        )}
                        <span className="warning-sub">
                          frame {w.frame} · {w.time_s.toFixed(2)}s · p={w.score.toFixed(3)}
                        </span>
                        <button
                          className="btn"
                          style={{ marginLeft: "auto" }}
                          onClick={() => setFrame(w.frame)}
                        >
                          Show
                        </button>
                      </div>

                      {outOfWindow && (
                        <p style={{ margin: "0 0 6px", fontSize: 12.5, color: "var(--watch)" }}>
                          Fired before the imminent window — scored as a <b>false alarm</b>, not an
                          anticipation. That frame is labelled normal.
                        </p>
                      )}

                      {w.evidence && (
                        <>
                          <div className="warning-evidence">
                            Evidence: <b>{w.evidence.phrase}</b>
                          </div>
                          <div className="joint-chips">
                            {w.evidence.top_joints.map((j) => (
                              <span key={j.joint} className="chip">
                                <b>{j.joint.replace(/_/g, " ")}</b> {(j.relevance * 100).toFixed(0)}%
                              </span>
                            ))}
                          </div>
                        </>
                      )}

                      <div style={{ marginTop: 10 }}>
                        {check ? (
                          <>
                            <div style={{ marginBottom: 8 }}>
                              <span className={`verdict ${check.faithful ? "pass" : "fail"}`}>
                                {check.faithful ? "Faithful" : "Not demonstrably faithful"}
                              </span>
                            </div>
                            <FaithfulnessChart data={check} />
                          </>
                        ) : (
                          <button
                            className="btn primary"
                            onClick={() => void verify(w)}
                            disabled={checking !== null}
                          >
                            {checking === w.frame ? (
                              <>
                                <span className="spinner" /> Running deletion / insertion…
                              </>
                            ) : (
                              "Verify this explanation"
                            )}
                          </button>
                        )}
                        {checking === w.frame && (
                          <p className="hint" style={{ marginTop: 8, marginBottom: 0 }}>
                            ~150 forward passes on a shared CPU; this takes a few seconds.
                          </p>
                        )}
                      </div>
                    </div>
                  );
                })
              )}
            </section>
          )}
        </div>
      </div>

      <p className="footnote">
        Skeletons only — no pixels reach the server, and there is no endpoint that accepts one.
        Uploaded video is posed in your browser with MoveNet and discarded. Lead time is measured
        from the annotated impact frame <code style={{ fontFamily: "var(--mono)" }}>t*</code>, and a
        warning is only counted as an anticipation if it fires inside the imminent window.{" "}
        <a href="https://github.com/adityaayushman/FBimpact" target="_blank" rel="noreferrer">
          Source and full methodology
        </a>
        .
      </p>
    </div>
  );
}
