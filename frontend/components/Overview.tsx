"use client";

import { useMemo } from "react";
import Hero from "./Hero";
import runs from "@/lib/runs.json";
import datasets from "@/lib/datasets.json";

const EXPECTED_CONFIRM = 85; // 4 arms x 4 seeds x 5 folds, plus arm D seed 0 x 5 folds

const LADDER = [
  { level: "Replicated", tone: "calm", text: "Pre-impact time-weighted loss did not help on UR Fall (3 of 3 seeds)." },
  { level: "Single seed", tone: "watch", text: "Richer features and four-phase supervision cut false alarms on the combined benchmark." },
  { level: "Pre-registered", tone: "imminent", text: "Seeds 1-4 test that hypothesis against rules fixed before they ran." },
];

/** Landing dashboard: headline state of the evidence, not a sales page. */
export default function Overview() {
  const confirm = useMemo(() => {
    const done = (runs.runs as { role: string }[]).filter((r) => r.role === "confirmatory").length;
    return Math.min(done, EXPECTED_CONFIRM);
  }, []);
  const d = (datasets as { datasets?: unknown }).datasets;
  const pct = Math.round((confirm / EXPECTED_CONFIRM) * 100);

  return (
    <>
      <Hero />

      <section className="pane">
        <div className="cols-3">
          <div className="glass card">
            <div className="card-title">Confirmation test</div>
            <div className="big">{confirm}<span className="dim"> / {EXPECTED_CONFIRM}</span></div>
            <div className="bar" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}
                 aria-label="Confirmation runs completed">
              <span style={{ width: `${pct}%` }} />
            </div>
            <p className="card-sub">Pre-registered runs finished (seeds 1-4). The verdict stays pending until all are in.</p>
          </div>
          <div className="glass card">
            <div className="card-title">Benchmark</div>
            <div className="big">200<span className="dim"> clips</span></div>
            <p className="card-sub">129 falls from UR Fall and Le2i, about 26 minutes of normal activity. {d ? "" : ""}Clip-independent splits.</p>
          </div>
          <div className="glass card">
            <div className="card-title">Caveat that matters</div>
            <div className="big" style={{ color: "var(--watch)" }}>Acted falls</div>
            <p className="card-sub">Young volunteers, controlled settings. No actor IDs, so splits are weaker than subject-independent.</p>
          </div>
        </div>
      </section>

      <section className="pane">
        <h2 className="pane-title">Evidence ladder</h2>
        <p className="pane-sub">Each claim is labelled at the strength it was measured.</p>
        <div className="cols-3">
          {LADDER.map((l) => (
            <div key={l.level} className="glass card">
              <span className="chip" style={{ color: `var(--${l.tone})` }}>{l.level}</span>
              <p className="card-sub" style={{ marginTop: 10 }}>{l.text}</p>
            </div>
          ))}
        </div>
        <p style={{ marginTop: 16 }}>
          <a className="btn" href="#/results">See the measured tables →</a>{" "}
          <a className="btn" href="#/analyzer">Run the analyzer</a>
        </p>
      </section>
    </>
  );
}
