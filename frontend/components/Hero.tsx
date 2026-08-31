"use client";

import { useEffect, useMemo, useState } from "react";
import Scene3D from "./Scene3D";
import { ABSTRACT, HERO_STATS, PROJECT } from "@/lib/content";
import { SCHEMATIC_IMPACT, schematicFallLoop } from "@/lib/skeleton3d";

const FRAMES = 150;
const IMPACT = SCHEMATIC_IMPACT(FRAMES);

/**
 * Landing view.
 *
 * The 3D figure runs a *schematic* fall built from joint angles, not model
 * output and not data — it is labelled that way in the viewer. It exists so the
 * page opens on the 17-joint graph the model actually operates on, moving,
 * rather than on a still image standing in for a moving system. Real inference
 * lives one section down, against the live API.
 */
export default function Hero() {
  const sequence = useMemo(() => schematicFallLoop(FRAMES), []);
  const [frame, setFrame] = useState(0);

  useEffect(() => {
    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (reduced) {
      // Hold on the moment of maximum instability rather than animating.
      setFrame(IMPACT - 8);
      return;
    }
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      raf = requestAnimationFrame(tick);
      if (now - last >= 1000 / 30) {
        last = now;
        setFrame((f) => (f + 1) % FRAMES);
      }
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  // A stand-in imminence curve for the schematic: rises through the ballistic
  // phase so the figure's colour shifts as it loses balance. Clearly tied to
  // the schematic, never presented as a model score.
  const onset = Math.round(FRAMES * 0.36);
  const schematicScore =
    frame < onset ? 0.05
      : frame <= IMPACT ? Math.min(((frame - onset) / (IMPACT - onset)) ** 1.6, 1) * 0.99
      : 0.2;

  return (
    <section id="overview" className="pane hero">
      <div className="hero-grid">
        <div>
          <span className="eyebrow">Vision · privacy-preserving · explanation-verified</span>

          <h1 className="hero-title">
            Predict the fall <span className="grad">before impact</span> — and prove which joints said so.
          </h1>

          <p className="hero-lede">{ABSTRACT}</p>

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 26 }}>
            <a className="btn primary" href="#demo">Run the live demo</a>
            <a className="btn" href="#research">Read the research</a>
            <a className="btn ghost" href={PROJECT.repo} target="_blank" rel="noreferrer">Source ↗</a>
          </div>

          <div className="stat-row">
            {HERO_STATS.map((s) => (
              <div key={s.label} className="glass stat">
                <div className="stat-value">{s.value}</div>
                <div className="stat-label">{s.label}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="glass" style={{ padding: 12 }}>
          <Scene3D
            keypoints={sequence}
            frame={frame}
            score={schematicScore}
            threshold={0.7}
            height={432}
            label="schematic · 17-joint graph"
          />
          <p className="card-sub" style={{ margin: "11px 4px 3px" }}>
            A schematic of the COCO-17 graph the model operates on, animated through a
            ballistic collapse. Not model output — the live analyser is below.
          </p>
        </div>
      </div>
    </section>
  );
}
