"use client";

import data from "@/lib/faithfulness.json";
import { Section } from "./ui";

interface Pooled { mean: number; lo: number | null; hi: number | null; runs: number; share_positive: number }
interface Row {
  benchmark: string; variant: string; method: string;
  runs: number; explanations: number; deletion: Pooled; insertion: Pooled;
}

const ROWS = data.rows as unknown as Row[];
const OVERALL = data.overall as unknown as { runs: number; explanations: number; deletion: Pooled; insertion: Pooled };

const VARIANT: Record<string, string> = {
  ours_preimpact: "Full model", baseline_stgcn: "Baseline (plain loss, no attention)",
  ablation_no_preimpact_loss: "No pre-impact loss", ablation_no_temporal: "No temporal modelling",
  ablation_no_velocity: "No velocity features", ablation_no_grounding: "No grounding head",
  combined_xyv: "A · control features", combined: "B · 14-channel features", combined_phases: "C · + four-phase head",
};

const AXIS: [number, number] = [-0.35, 0.4];
const x = (v: number) => ((Math.min(Math.max(v, AXIS[0]), AXIS[1]) - AXIS[0]) / (AXIS[1] - AXIS[0])) * 100;

/** One interval on a shared axis: line = 95% interval over runs, dot = pooled estimate, tick = zero. */
function Interval({ p }: { p: Pooled }) {
  const clear = p.lo !== null && p.lo > 0;          // interval entirely above zero
  const below = p.hi !== null && p.hi < 0;
  const colour = clear ? "var(--burgundy)" : below ? "var(--impact)" : "var(--mocha-dark)";
  return (
    <span className="interval" aria-hidden>
      <i className="zero" style={{ left: `${x(0)}%` }} />
      {p.lo !== null && p.hi !== null && (
        <i className="span" style={{ left: `${x(p.lo)}%`, width: `${x(p.hi) - x(p.lo)}%`, background: colour }} />
      )}
      <i className="dotpt" style={{ left: `${x(p.mean)}%`, background: colour }} />
    </span>
  );
}

const sgn = (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(3)}`;
const range = (p: Pooled) => (p.lo === null || p.hi === null ? "single run" : `${sgn(p.lo)} to ${sgn(p.hi)}`);

/** RQ2: are the joint explanations faithful? Pooled from every run that scored explanations. */
export default function Explanations() {
  const benchmarks = Array.from(new Set(ROWS.map((r) => r.benchmark)));
  return (
    <>
      <Section
        title="Are the joint explanations faithful?"
        sub="A faithful explanation names joints the model really relies on: deleting them should hurt the score more than deleting random joints, and adding them back should restore it faster."
      >
        <div className="notice warn" style={{ marginBottom: 18 }}>
          <b>Exploratory pooling, not a pre-registered test.</b> This combines every run that scored
          explanations, using only the &ldquo;attention&rdquo; method (and &ldquo;gradient × input&rdquo;
          for the no-grounding ablation). The pre-registered comparison of all four methods on the same
          models is still to run (see below).
        </div>

        <div className="cols-3">
          <div className="card">
            <div className="card-title">Deletion gap</div>
            <div className="big">{sgn(OVERALL.deletion.mean)}</div>
            <p className="card-sub">95% interval {range(OVERALL.deletion)} over {OVERALL.runs} runs.</p>
          </div>
          <div className="card">
            <div className="card-title">Insertion gap</div>
            <div className="big">{sgn(OVERALL.insertion.mean)}</div>
            <p className="card-sub">95% interval {range(OVERALL.insertion)}. Positive means better than a random joint order.</p>
          </div>
          <div className="card">
            <div className="card-title">Explanations scored</div>
            <div className="big">{OVERALL.explanations.toLocaleString()}</div>
            <p className="card-sub">Only a few per run, because a warning must fire to be explained. The run is the unit of analysis.</p>
          </div>
        </div>

        <div className="card" style={{ marginTop: 18 }}>
          <div className="card-title">How to read the headline</div>
          <p className="card-sub">
            Overall, the named joints beat a random ordering by a small but reliable margin: both
            intervals sit above zero. The effect is modest, and it is not uniform. It is strongest for
            the combined-benchmark models and weak on UR Fall alone. For the Le2i baseline the point
            estimates are below zero, though their intervals include it, and the no-temporal ablation
            on Le2i is reliably below zero (4 runs, 24 explanations). The explanations are better than
            chance on average, not a guarantee for any single model.
          </p>
        </div>
      </Section>

      {benchmarks.map((b) => (
        <Section key={b} title={b}>
          <div className="card">
            <div className="table-wrap">
              <table className="data">
                <thead>
                  <tr>
                    <th>Variant</th><th className="num">Runs</th><th className="num">Expl.</th>
                    <th>Deletion gap (95% interval)</th><th>Insertion gap (95% interval)</th>
                  </tr>
                </thead>
                <tbody>
                  {ROWS.filter((r) => r.benchmark === b).map((r) => (
                    <tr key={`${r.variant}-${r.method}`}>
                      <td>{VARIANT[r.variant] ?? r.variant}
                        {r.method !== "attention" && <span className="dim"> · {r.method.replace("_", " × ")}</span>}</td>
                      <td className="num mono">{r.runs}</td>
                      <td className="num mono">{r.explanations}</td>
                      <td><Interval p={r.deletion} /> <span className="mono readout">{sgn(r.deletion.mean)}</span></td>
                      <td><Interval p={r.insertion} /> <span className="mono readout">{sgn(r.insertion.mean)}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </Section>
      ))}
      <p className="card-sub">
        Dark intervals lie entirely above zero (reliably better than random), grey ones include zero,
        and red ones lie below it. The scale runs from {AXIS[0]} to {AXIS[1]}. Intervals resample runs,
        not explanations, because explanations from one model are not independent. Generated by{" "}
        <code>scripts/analyze_faithfulness.py</code> · <span className="mono">{data.generated_utc}</span>
      </p>

      <Section
        title="Pre-registered work still to run"
        sub="Fixed in results/analysis/PLAN_COMPLETION.txt before any of it was run."
      >
        <div className="stack">
          {[
            ["Explanation methods compared", "Attention, gradient, gradient × input and occlusion, re-scored on the same 40 saved models. A method is called more faithful only if both paired gaps clear zero."],
            ["The pre-impact loss on the combined benchmark", "Arm B with the loss switched on, 20 runs. Until now this was tested only on UR Fall."],
            ["The ablation grids", "Every ablation to seeds 0–2 × 5 folds on UR Fall and Le2i, as the Method page promises."],
          ].map(([t, b]) => (
            <div key={t} className="card">
              <span className="chip" style={{ color: "var(--watch)", borderColor: "var(--watch)" }}>pending</span>
              <div className="card-title" style={{ marginTop: 10 }}>{t}</div>
              <p className="card-sub">{b}</p>
            </div>
          ))}
        </div>
      </Section>
    </>
  );
}
