"use client";

import { useMemo } from "react";
import Hero from "./Hero";
import { Icon, Section } from "./ui";
import { hrefFor, type ViewId } from "@/lib/route";
import runs from "@/lib/runs.json";

const EXPECTED_CONFIRM = 85; // 4 arms x seeds 1-4 x 5 folds, plus arm D seed 0 x 5 folds

const LADDER = [
  { level: "Replicated", tone: "var(--calm)", text: "The pre-impact time-weighted loss did not help on UR Fall: recall 0.633 against 0.767 with it switched off, in 3 of 3 seeds." },
  { level: "Single seed", tone: "var(--watch)", text: "Richer features and four-phase supervision cut false alarms on the combined benchmark from 135 to 100 to 67. One seed: a hypothesis." },
  { level: "Pre-registered", tone: "var(--imminent)", text: "Seeds 1 to 4 test that hypothesis against decision rules that were committed before any of those runs existed." },
];

const EXPLORE: { view: ViewId; sub?: string; icon: string; title: string; body: string }[] = [
  { view: "analyzer", icon: "pulse", title: "Run the live analyzer", body: "Score a skeleton sequence, or your own video, and see which joints drove each warning." },
  { view: "results", icon: "chart", title: "Read the results", body: "Benchmark tables, operating curves and the pre-registered confirmation test." },
  { view: "generalisation", icon: "globe", title: "Check generalisation", body: "Does the result hold in a room and on a camera the model never saw?" },
  { view: "method", sub: "lead-time", icon: "layers", title: "See how it is measured", body: "Lead time defined precisely, and every judgement call behind the headline number." },
];

/** Landing dashboard: the headline state of the evidence, not a sales page. */
export default function Overview() {
  const confirm = useMemo(() => {
    const done = (runs.runs as { role: string }[]).filter((r) => r.role === "confirmatory").length;
    return Math.min(done, EXPECTED_CONFIRM);
  }, []);
  const pct = Math.round((confirm / EXPECTED_CONFIRM) * 100);

  return (
    <>
      <Hero />

      <Section
        title="Where the evidence stands"
        sub="Three numbers to carry away, each stated at the strength it was measured."
      >
        <div className="cols-3">
          <div className="card">
            <div className="card-title">Confirmation test</div>
            <div className="big">{confirm}<span className="dim" style={{ fontSize: 20 }}> / {EXPECTED_CONFIRM} runs</span></div>
            <div className="bar" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}
                 aria-label="Confirmation runs completed">
              <span style={{ width: `${pct}%` }} />
            </div>
            <p className="card-sub">Pre-registered runs finished on seeds 1 to 4. The verdict stays pending until all are in.</p>
          </div>
          <div className="card">
            <div className="card-title">Benchmark size</div>
            <div className="big">200<span className="dim" style={{ fontSize: 20 }}> clips</span></div>
            <p className="card-sub">129 falls from UR Fall and Le2i and about 26 minutes of normal activity. Clip-independent splits.</p>
          </div>
          <div className="card">
            <div className="card-title">Caveat that matters most</div>
            <div className="big" style={{ color: "var(--watch)" }}>Acted falls</div>
            <p className="card-sub">Young volunteers in controlled settings. No actor IDs, so splits are weaker than subject-independent.</p>
          </div>
        </div>
      </Section>

      <Section title="Evidence ladder" sub="Each claim is labelled at the strength it was measured.">
        <div className="cols-3">
          {LADDER.map((l) => (
            <div key={l.level} className="card" style={{ borderTop: `3px solid ${l.tone}` }}>
              <span className="chip" style={{ color: l.tone, borderColor: l.tone }}>{l.level}</span>
              <p className="card-sub" style={{ marginTop: 12 }}>{l.text}</p>
            </div>
          ))}
        </div>
      </Section>

      <Section title="Explore the project">
        <div className="cols-4">
          {EXPLORE.map((e) => (
            <a key={e.title} href={hrefFor(e.view, e.sub)} className="card link-card">
              <span className="link-icon"><Icon name={e.icon} size={20} /></span>
              <span className="card-title">{e.title}</span>
              <span className="card-sub">{e.body}</span>
              <span className="link-more">Open <Icon name="arrow" size={15} /></span>
            </a>
          ))}
        </div>
      </Section>
    </>
  );
}
