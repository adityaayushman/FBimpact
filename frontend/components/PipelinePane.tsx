"use client";

import { PIPELINE } from "@/lib/content";

/**
 * The A–H pipeline, with the privacy boundary drawn where it actually falls.
 *
 * The boundary between stage A and stage B is the whole privacy argument: after
 * it, no component in the system can obtain a pixel, because none is retained
 * or passed on. Marking it in the diagram is the point of the diagram.
 */
export default function PipelinePane() {
  return (
    <section id="pipeline" className="pane">
      <span className="eyebrow">Architecture</span>
      <h2 className="pane-title">Eight stages, three of them novel</h2>
      <p className="pane-lede">
        Perception is frozen and reused. The research effort concentrates on a small temporal
        model, a joint-level explanation, and a test that proves the explanation is faithful —
        so the whole thing is completable by one person and runs in real time on modest hardware.
      </p>

      {/* privacy boundary marker */}
      <div className="glass pad-sm" style={{
        display: "flex", alignItems: "center", gap: 12, marginBottom: 16,
        borderColor: "var(--accent-line)", background: "var(--accent-dim)",
      }}>
        <svg width="17" height="17" viewBox="0 0 16 16" fill="none" style={{ flexShrink: 0 }}>
          <path d="M8 1.4l5.2 2v4c0 3.3-2.2 6-5.2 7.2C5 13.4 2.8 10.7 2.8 7.4v-4L8 1.4z"
                stroke="var(--accent)" strokeWidth="1.4" strokeLinejoin="round" />
          <path d="M5.9 7.9l1.5 1.5 2.8-2.9" stroke="var(--accent)" strokeWidth="1.5"
                strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <p style={{ margin: 0, fontSize: 13, color: "var(--text-2)", lineHeight: 1.55 }}>
          <b style={{ color: "var(--text)" }}>Privacy boundary sits between A and B.</b>{" "}
          Frames are decoded, posed and discarded. Everything downstream — training, evaluation,
          this website — sees joint coordinates and confidences, never an image. In this
          deployment pose estimation runs in your browser, so video never leaves your device and
          the API has no endpoint that accepts one.
        </p>
      </div>

      <div className="cols-4">
        {PIPELINE.map((s) => (
          <div key={s.id} className={`stage${s.novel ? " novel" : ""}`}>
            {s.novel && <span className="badge-novel">NOVEL</span>}
            <div className="stage-id">STAGE {s.id}</div>
            <div className="stage-name">{s.name}</div>
            <div className="stage-body">{s.body}</div>
            <div className="stage-io">{s.io}</div>
          </div>
        ))}
      </div>

      <div className="legend">
        <span><i className="swatch" style={{ background: "var(--accent)" }} /> research contribution</span>
        <span><i className="swatch" style={{ background: "var(--text-4)" }} /> reusable plumbing</span>
      </div>
    </section>
  );
}
