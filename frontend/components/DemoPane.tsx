"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api, API_URL,
  type Analysis, type ClipSummary, type Faithfulness, type Health, type Keypoints, type Warning,
} from "@/lib/api";
import FaithfulnessChart from "./FaithfulnessChart";
import Scene3D from "./Scene3D";
import ScoreChart from "./ScoreChart";
import VideoUpload from "./VideoUpload";

interface Source {
  id: string;
  keypoints: Keypoints;
  fps: number;
  impactFrame: number | null;
  title: string;
  isFall: boolean;
}

export default function DemoPane({
  health,
  bootError,
  waking = false,
}: {
  health: Health | null;
  bootError: string | null;
  waking?: boolean;
}) {
  const [clips, setClips] = useState<ClipSummary[]>([]);
  const [source, setSource] = useState<Source | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [checks, setChecks] = useState<Record<number, Faithfulness>>({});
  const [checking, setChecking] = useState<number | null>(null);

  // Deliberately waits for the health handshake rather than firing in parallel.
  // Both requests would otherwise race the same cold start, and the one without
  // retries would fail and report the API down while the other was still
  // patiently waking it.
  useEffect(() => {
    if (!health) return;
    let cancelled = false;
    api.clips()
      .then((c) => { if (!cancelled) setClips(c); })
      .catch((e) => { if (!cancelled) setError(e instanceof Error ? e.message : String(e)); });
    return () => { cancelled = true; };
  }, [health]);

  /* -- playback ---------------------------------------------------------- */
  const rafRef = useRef<number | null>(null);
  const lastRef = useRef(0);

  useEffect(() => {
    if (!playing || !source) return;
    const interval = 1000 / source.fps;
    const total = source.keypoints.length;
    const tick = (now: number) => {
      if (now - lastRef.current >= interval) {
        lastRef.current = now;
        setFrame((f) => {
          if (f + 1 >= total) { setPlaying(false); return total - 1; }
          return f + 1;
        });
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => { if (rafRef.current !== null) cancelAnimationFrame(rafRef.current); };
  }, [playing, source]);

  /* -- actions ----------------------------------------------------------- */
  const run = useCallback(async (next: Source) => {
    setSource(next); setAnalysis(null); setChecks({});
    setFrame(0); setPlaying(false); setError(null);
    setBusy("Scoring every frame…");
    try {
      const result = await api.analyze({
        keypoints: next.keypoints, fps: next.fps,
        impact_frame: next.impactFrame, clip_id: next.id, explain: true,
      });
      setAnalysis(result);
      // Land on the first warning — it is the thing worth looking at, and
      // starting at frame 0 makes the visitor hunt for it.
      if (result.warnings.length) setFrame(result.warnings[0].frame);
      else setPlaying(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally { setBusy(null); }
  }, []);

  const loadClip = useCallback(async (s: ClipSummary) => {
    setBusy(`Loading ${s.activity.replace(/_/g, " ")}…`);
    setError(null);
    try {
      const p = await api.clip(s.clip_id);
      await run({
        id: p.clip_id, keypoints: p.keypoints, fps: p.fps,
        impactFrame: p.impact_frame, isFall: p.label === "fall",
        title: p.activity.replace(/_/g, " "),
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(null);
    }
  }, [run]);

  const verify = useCallback(async (w: Warning) => {
    if (!source) return;
    setChecking(w.frame); setError(null);
    try {
      const r = await api.faithfulness({
        keypoints: source.keypoints, frame: w.frame, fps: source.fps,
        impact_frame: source.impactFrame, num_random: 3, clip_id: source.id,
      });
      setChecks((prev) => ({ ...prev, [w.frame]: r }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally { setChecking(null); }
  }, [source]);

  /* -- derived ----------------------------------------------------------- */
  const activeWarning = useMemo(
    () => analysis?.warnings.find((w) => frame >= w.frame && frame < w.frame + 14) ?? null,
    [analysis, frame]
  );
  const score = analysis?.scores[Math.min(frame, analysis.scores.length - 1)] ?? 0;
  const falls = clips.filter((c) => c.label === "fall");
  const adls = clips.filter((c) => c.label === "adl");

  return (
    <section id="demo" className="pane">
      <span className="eyebrow">Live demo</span>
      <h2 className="pane-title">Run the model on real skeleton data</h2>
      <p className="pane-lede">
        Every result below is computed live by the deployed model, through the same windowing,
        frame alignment and trigger rule as the offline evaluation. The normal activities are
        deliberately <em>hard negatives</em> — sitting, bending and lying down are all controlled
        descents, and a model that fires on any downward motion fails here.
      </p>

      <div className="disclaimer" style={{ marginBottom: 18 }}>
        <svg width="17" height="17" viewBox="0 0 16 16" fill="none" style={{ flexShrink: 0, marginTop: 1 }}>
          <path d="M8 1.8l6.4 11.4H1.6L8 1.8z" stroke="var(--impact)" strokeWidth="1.4" strokeLinejoin="round" />
          <path d="M8 6.2v3.1M8 11.2v.5" stroke="var(--impact)" strokeWidth="1.6" strokeLinecap="round" />
        </svg>
        <div>
          <strong>Research prototype — not a medical device.</strong> The deployed checkpoint is
          trained on a <b>synthetic fixture</b>, not on real fall data, so nothing it outputs
          carries clinical meaning. Public fall datasets use acted falls by young volunteers.
          This demo exists to show the pipeline and the faithfulness test, not to make a safety claim.
        </div>
      </div>

      {waking && (
        <div className="notice" style={{ marginBottom: 16, display: "flex", gap: 11, alignItems: "center" }}>
          <span className="spinner" />
          <span>
            Waking the inference API. It sleeps when idle on the free tier and takes up to a
            minute to start — the demo clips will appear as soon as it answers.
          </span>
        </div>
      )}

      {(error || bootError) && (
        <div className="notice danger" style={{ marginBottom: 16 }}>
          <div>{error ?? bootError}</div>
          <p style={{ margin: "9px 0 0", fontSize: 12.5, color: "var(--text-3)" }}>
            <span className="mono">{API_URL}</span> ·{" "}
            <button className="btn" style={{ padding: "3px 10px", fontSize: 12 }}
                    onClick={() => window.location.reload()}>
              Retry
            </button>
          </p>
        </div>
      )}

      <div className="demo-grid">
        {/* -- source picker ---------------------------------------------- */}
        <div className="stack">
          <div className="glass pad-sm">
            <h3 className="card-title">Demo clips</h3>
            <p className="card-sub">Bundled with the API — no upload needed.</p>

            {clips.length === 0 && !error && (
              <div className="empty">{waking ? "Waiting for the API…" : "Loading…"}</div>
            )}

            {falls.length > 0 && <div className="group-head">Falls</div>}
            {falls.map((c) => (
              <button key={c.clip_id} className="clip-btn" disabled={!!busy}
                      aria-pressed={source?.id === c.clip_id} onClick={() => void loadClip(c)}>
                <span className="dot fall" />
                <span className="clip-name">{c.activity.replace(/_/g, " ")}</span>
                <span className="clip-meta">{c.duration_s.toFixed(1)}s</span>
              </button>
            ))}

            {adls.length > 0 && <div className="group-head">Normal activity</div>}
            {adls.map((c) => (
              <button key={c.clip_id} className="clip-btn" disabled={!!busy}
                      aria-pressed={source?.id === c.clip_id} onClick={() => void loadClip(c)}>
                <span className="dot adl" />
                <span className="clip-name">{c.activity.replace(/_/g, " ")}</span>
                <span className="clip-meta">{c.duration_s.toFixed(1)}s</span>
              </button>
            ))}
          </div>

          <div className="glass pad-sm">
            <h3 className="card-title">Your own video</h3>
            <p className="card-sub">
              Posed in your browser; only joint coordinates are sent.
            </p>
            <VideoUpload
              maxFrames={health?.limits.max_frames ?? 600}
              disabled={!!busy}
              onError={setError}
              onExtracted={({ keypoints, fps, name }) =>
                void run({ id: name, keypoints, fps, impactFrame: null, title: name, isFall: false })
              }
            />
          </div>

          {health && (
            <div className="glass pad-sm">
              <h3 className="card-title">Operating point</h3>
              <p className="card-sub" style={{ marginBottom: 10 }}>
                Chosen on a validation split under a false-alarm budget, then frozen into the
                checkpoint — not tuned on what you are looking at.
              </p>
              <table className="data">
                <tbody>
                  <tr><td>Threshold τ</td><td className="num">{health.decision.threshold.toFixed(2)}</td></tr>
                  <tr><td>Persistence k</td><td className="num">{health.decision.persistence}</td></tr>
                  <tr><td>Window</td><td className="num">{health.model.window}f</td></tr>
                  <tr><td>Imminent window</td><td className="num">{health.model.w_pre_frames}f</td></tr>
                  <tr><td>Evidence</td><td className="num">{health.model.explain_method}</td></tr>
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* -- viewer ------------------------------------------------------ */}
        <div className="stack">
          {!source && !error && (
            <div className="glass"><div className="empty">
              Select a clip to run the model, or load your own video.
            </div></div>
          )}

          {source && (
            <div className="glass pad">
              <div style={{ display: "flex", alignItems: "baseline", gap: 12,
                            flexWrap: "wrap", marginBottom: 14 }}>
                <h3 className="card-title" style={{ margin: 0, textTransform: "capitalize" }}>
                  {source.title}
                </h3>
                <span className="readout">
                  {busy ?? (analysis
                    ? `${analysis.frames} frames scored in ${analysis.latency_ms.toFixed(0)} ms · ` +
                      (analysis.warnings.length
                        ? `${analysis.warnings.length} warning${analysis.warnings.length > 1 ? "s" : ""}`
                        : "no warning")
                    : "")}
                </span>
                {busy && <span className="spinner" />}
              </div>

              <div className="viewer-grid">
                <div>
                  <Scene3D
                    keypoints={source.keypoints}
                    frame={frame}
                    relevance={activeWarning?.evidence?.relevance ?? null}
                    score={score}
                    threshold={analysis?.decision.threshold ?? 0.7}
                    height={330}
                    label={activeWarning ? "evidence at warning frame" : "2D pose · 3D stage"}
                  />
                  {activeWarning && (
                    <p className="card-sub" style={{ margin: "9px 2px 0", textAlign: "center" }}>
                      Glowing joints are the evidence for this warning.
                    </p>
                  )}
                </div>

                <div>
                  {analysis && <ScoreChart analysis={analysis} frame={frame} onScrub={setFrame} />}
                  <div className="legend">
                    <span><i className="swatch" style={{ background: "var(--accent)" }} /> p(fall imminent)</span>
                    <span><i className="swatch" style={{ background: "var(--imminent)" }} /> threshold &amp; triggers</span>
                    {analysis?.imminent_window && (
                      <span><i className="swatch" style={{ background: "var(--accent)", opacity: 0.4, height: 9 }} /> imminent window</span>
                    )}
                    {analysis?.impact_frame !== null && analysis && (
                      <span><i className="swatch" style={{ background: "var(--impact)", width: 3, height: 11 }} /> impact</span>
                    )}
                  </div>
                </div>
              </div>

              <div className="transport">
                <button className="btn" onClick={() => setPlaying((p) => !p)} disabled={!analysis}>
                  {playing ? "Pause" : "Play"}
                </button>
                <input type="range" min={0} max={Math.max(source.keypoints.length - 1, 0)}
                       value={frame} aria-label="Scrub through the clip"
                       onChange={(e) => { setPlaying(false); setFrame(Number(e.target.value)); }} />
                <span className="readout">
                  {frame}/{source.keypoints.length - 1} · {(frame / source.fps).toFixed(2)}s · p={score.toFixed(3)}
                </span>
              </div>
            </div>
          )}

          {/* -- warnings ------------------------------------------------- */}
          {analysis && (
            <div className="glass pad">
              <h3 className="card-title">Warnings &amp; grounded evidence</h3>
              <p className="card-sub">
                Each warning names the joints that drove it. Whether that evidence is{" "}
                <em>faithful</em> is a separate question — press verify to run the
                deletion/insertion test against a random joint ordering.
              </p>

              {analysis.warnings.length === 0 ? (
                <div className="empty">
                  {analysis.impact_frame !== null
                    ? "No warning fired before impact — this clip is a miss."
                    : "No warning fired. For a normal-activity clip, that is the correct outcome."}
                </div>
              ) : analysis.warnings.map((w) => {
                const early = w.within_imminent_window === false;
                const check = checks[w.frame];
                return (
                  <div key={w.frame} className={`warn-card${early ? " early" : ""}`}>
                    <div className="warn-head">
                      <span className="warn-lead">
                        {w.lead_time !== null ? `${w.lead_time.toFixed(2)}s before impact` : "warning"}
                      </span>
                      <span className="warn-meta">
                        frame {w.frame} · {w.time_s.toFixed(2)}s · p={w.score.toFixed(3)}
                      </span>
                      <button className="btn" style={{ marginLeft: "auto" }}
                              onClick={() => { setPlaying(false); setFrame(w.frame); }}>
                        Show
                      </button>
                    </div>

                    {early && (
                      <p className="notice warn" style={{ margin: "0 0 9px", fontSize: 12.5 }}>
                        Fired <b>before</b> the imminent window, so it is scored as a false alarm
                        rather than an anticipation — that frame is labelled normal.
                      </p>
                    )}

                    {w.evidence && (
                      <>
                        <div style={{ fontSize: 13.5 }}>
                          Evidence: <b>{w.evidence.phrase}</b>
                        </div>
                        <div className="chips">
                          {w.evidence.top_joints.map((j) => (
                            <span key={j.joint} className="chip">
                              <b>{j.joint.replace(/_/g, " ")}</b> {(j.relevance * 100).toFixed(0)}%
                            </span>
                          ))}
                        </div>
                      </>
                    )}

                    <div style={{ marginTop: 12 }}>
                      {check ? (
                        <>
                          <div style={{ marginBottom: 10 }}>
                            <span className={`verdict ${check.faithful ? "pass" : "fail"}`}>
                              {check.faithful ? "Faithful" : "Not demonstrably faithful"}
                            </span>
                          </div>
                          <FaithfulnessChart data={check} />
                        </>
                      ) : (
                        <>
                          <button className="btn primary" disabled={checking !== null}
                                  onClick={() => void verify(w)}>
                            {checking === w.frame
                              ? <><span className="spinner" /> Running deletion / insertion…</>
                              : "Verify this explanation"}
                          </button>
                          {checking === w.frame && (
                            <p className="card-sub" style={{ margin: "9px 0 0" }}>
                              ~150 forward passes on a shared CPU — around ten seconds.
                            </p>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
