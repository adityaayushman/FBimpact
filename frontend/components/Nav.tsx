"use client";

import { useEffect, useState } from "react";

const SECTIONS = [
  { id: "overview", label: "Overview" },
  { id: "demo", label: "Live demo" },
  { id: "research", label: "Research" },
  { id: "pipeline", label: "Pipeline" },
  { id: "method", label: "Method" },
  { id: "api", label: "API" },
];

/**
 * Sticky glass navigation with scroll-spy.
 *
 * Uses IntersectionObserver rather than a scroll handler so highlighting costs
 * nothing on the main thread while the 3D scene is running.
 */
export default function Nav({ apiOk }: { apiOk: boolean | null }) {
  const [active, setActive] = useState("overview");

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (visible) setActive(visible.target.id);
      },
      // A band across the upper-middle of the viewport: a section counts as
      // "current" once its top third is in view, which matches what a reader
      // perceives better than an element-centre test does.
      { rootMargin: "-18% 0px -55% 0px", threshold: [0.05, 0.3, 0.6] }
    );

    SECTIONS.forEach(({ id }) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });
    return () => observer.disconnect();
  }, []);

  return (
    <nav className="nav">
      <div className="nav-inner">
        <a href="#overview" className="brand" style={{ color: "inherit", textDecoration: "none" }}>
          <span className="brand-mark" aria-hidden>
            <svg width="15" height="15" viewBox="0 0 16 16" fill="none">
              <circle cx="8" cy="3" r="1.9" fill="#04231f" />
              <path d="M8 5.2v4.4M8 9.6l-2.6 3.6M8 9.6l2.6 3.6M4.4 6.7L11.6 6.7"
                    stroke="#04231f" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </span>
          <span>
            FBimpact
            <span className="dim" style={{ fontWeight: 400 }}> · pre-impact anticipation</span>
          </span>
        </a>

        <div className="nav-links">
          {SECTIONS.map((s) => (
            <a key={s.id} href={`#${s.id}`} className={`nav-link${active === s.id ? " active" : ""}`}>
              {s.label}
            </a>
          ))}
        </div>

        <span className="status" style={{ marginLeft: "auto", flexShrink: 0 }} title={
          apiOk === null ? "Connecting to the inference API" :
          apiOk ? "Inference API reachable" : "Inference API unreachable"
        }>
          <span className={`status-dot ${apiOk === null ? "wait" : apiOk ? "ok" : "bad"}`} />
          <span className="mono" style={{ fontSize: 11 }}>
            {apiOk === null ? "connecting" : apiOk ? "api live" : "api down"}
          </span>
        </span>
      </div>
    </nav>
  );
}
