"use client";

import completion from "@/lib/completion.json";
import { Section } from "./ui";

/**
 * Results of the three experiments in results/analysis/PLAN_COMPLETION.txt that follow the main
 * confirmation: the pre-impact loss on the combined benchmark (RQ3) and the ablation grids.
 * Every verdict is passed through from scripts/analyze_arms.py; nothing is re-worded here, so a
 * "not shown" cannot be promoted by the page.
 */

interface Arm { arm: string; n_seeds: number; recall_mean: number; recall_sd: number | null; fa_mean: number; fa_sd: number | null; lead_mean: number }
interface Contrast {
  contrast: string; seeds: number; d_fa: number; d_fa_lo: number; d_fa_hi: number;
  d_recall: number; d_recall_lo: number; d_recall_hi: number;
  fa_verdict: string; recall_verdict?: string; fa_two_sided?: string;
  seeds_fewer_fa: string; seeds_lower_recall?: string; seeds_longer_lead?: string;
  lead_mean_treat?: number; lead_mean_ref?: number;
}
interface Exp { status: string; runs: number; expected: number; arms: Arm[]; contrasts: Contrast[] }

const E = completion as unknown as { rq3: Exp; ablation_urfd: Exp; ablation_le2i: Exp };

const NAME: Record<string, string> = {
  ours_preimpact: "Full model", baseline_stgcn: "Baseline (plain loss, no attention)", no_preimpact_loss: "No pre-impact loss",
  no_temporal: "No temporal modelling", no_velocity: "No velocity features", no_grounding: "No grounding head",
  combined: "B · plain loss (λ = 0)", combined_preimpact: "G · pre-impact loss (λ = 1.5)",
};

const sg = (v: number, d = 3) => `${v >= 0 ? "+" : ""}${v.toFixed(d)}`;
const pm = (m: number, s: number | null, d = 3) => (s == null ? m.toFixed(d) : `${m.toFixed(d)} ± ${s.toFixed(d)}`);

function Status({ e }: { e: Exp }) {
  const colour = e.status === "complete" ? "var(--calm)" : e.status === "partial" ? "var(--watch)" : "var(--text-3)";
  return <span className="chip" style={{ color: colour, borderColor: colour }}>{e.status} · {e.runs} of {e.expected} runs</span>;
}

function Verdict({ v }: { v?: string }) {
  const shown = v && v !== "not shown";
  return <span className="chip" style={shown ? { color: "var(--burgundy)", borderColor: "var(--burgundy)", fontWeight: 600 } : undefined}>{v ?? "—"}</span>;
}

function Pending({ e, what }: { e: Exp; what: string }) {
  return (
    <div className="card">
      <div className="empty">
        {what} has {e.runs === 0 ? "not been run yet" : `only ${e.runs} of ${e.expected} runs`}. It is pre-registered
        (PLAN_COMPLETION.txt) and runs on one Google Colab GPU, so every comparison in it is same-hardware. The table appears once
        at least one full seed (all five folds of every arm) is finished; verdicts appear only when the seeds the rules need are in.
      </div>
    </div>
  );
}

export default function FollowUps() {
  const rq3 = E.rq3;
  const g = rq3.contrasts[0];
  return (
    <>
      <Section
        title="Does the pre-impact loss help on the larger benchmark?"
        sub="On UR Fall alone (30 falls, 3 seeds) the time-weighted pre-impact loss cost recall and bought no lead time. This repeats the test on the combined benchmark (129 falls), with four seeds and the same rules as the main confirmation."
        actions={<Status e={rq3} />}
      >
        {rq3.arms.length === 0 ? <Pending e={rq3} what="The combined-benchmark test" /> : (
          <>
            <div className="card">
              <div className="table-wrap">
                <table className="data">
                  <thead><tr><th>Arm</th><th className="num">Recall</th><th className="num">False alarms</th><th className="num">Lead time (s)</th></tr></thead>
                  <tbody>
                    {rq3.arms.map((a) => (
                      <tr key={a.arm}>
                        <td>{NAME[a.arm] ?? a.arm}</td>
                        <td className="num">{pm(a.recall_mean, a.recall_sd)}</td>
                        <td className="num">{pm(a.fa_mean, a.fa_sd, 1)}</td>
                        <td className="num">{a.lead_mean.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            {g && (
              <div className="card" style={{ marginTop: 16 }}>
                <div className="card-title">G minus B, seeds 1–{g.seeds}, paired on identical clips</div>
                <ul className="kv">
                  <li><span>Δ false alarms (95% interval)</span><b>{sg(g.d_fa, 1)} [{sg(g.d_fa_lo, 1)}, {sg(g.d_fa_hi, 1)}]</b></li>
                  <li><span>Δ recall (95% interval)</span><b>{sg(g.d_recall)} [{sg(g.d_recall_lo)}, {sg(g.d_recall_hi)}]</b></li>
                  <li><span>Seeds with fewer false alarms</span><b>{g.seeds_fewer_fa}</b></li>
                  {g.seeds_lower_recall && <li><span>Seeds with lower recall</span><b>{g.seeds_lower_recall}</b></li>}
                  {g.seeds_longer_lead && <li><span>Seeds with longer lead time</span><b>{g.seeds_longer_lead}{g.lead_mean_treat != null && ` (${g.lead_mean_treat.toFixed(3)} s against ${g.lead_mean_ref?.toFixed(3)} s)`}</b></li>}
                  <li><span>False alarms</span><b><Verdict v={g.fa_verdict} /></b></li>
                  {g.recall_verdict && <li><span>Recall</span><b><Verdict v={g.recall_verdict} /></b></li>}
                </ul>
                <p className="card-sub" style={{ marginTop: 10 }}>
                  Wording comes from the pre-registered rules in PLAN_COMPLETION.txt. &ldquo;Not shown&rdquo; means the
                  experiment could not separate an effect from noise, not that there is no effect.
                </p>
              </div>
            )}
          </>
        )}
      </Section>

      {([["ablation_urfd", "UR Fall"], ["ablation_le2i", "Le2i"]] as const).map(([key, label]) => {
        const e = E[key];
        return (
          <Section
            key={key}
            title={`Ablations on ${label}`}
            sub="Each variant against the full model, seeds 0–2 × 5 folds, clips resampled once and shared across seeds. A change needs an interval that excludes zero and the same sign in at least 2 of 3 seeds."
            actions={<Status e={e} />}
          >
            {e.arms.length === 0 ? <Pending e={e} what={`The ${label} ablation grid`} /> : (
              <div className="card">
                <div className="table-wrap">
                  <table className="data">
                    <thead>
                      <tr>
                        <th>Variant</th><th className="num">Recall</th><th className="num">False alarms</th><th className="num">Lead (s)</th>
                        <th className="num">Δ recall vs full</th><th>Recall</th><th>False alarms</th>
                      </tr>
                    </thead>
                    <tbody>
                      {e.arms.map((a) => {
                        const c = e.contrasts.find((x) => x.contrast === `${a.arm}-full`);
                        return (
                          <tr key={a.arm} className={a.arm === "ours_preimpact" ? "ours" : undefined}>
                            <td>{NAME[a.arm] ?? a.arm}</td>
                            <td className="num">{pm(a.recall_mean, a.recall_sd)}</td>
                            <td className="num">{pm(a.fa_mean, a.fa_sd, 1)}</td>
                            <td className="num">{a.lead_mean.toFixed(3)}</td>
                            <td className="num">{c ? `${sg(c.d_recall)} [${sg(c.d_recall_lo)}, ${sg(c.d_recall_hi)}]` : "reference"}</td>
                            <td>{c ? <Verdict v={c.recall_verdict} /> : "—"}</td>
                            <td>{c ? <Verdict v={c.fa_two_sided} /> : "—"}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <p className="card-sub" style={{ marginTop: 12 }}>
                  Twenty comparisons are made across the two benchmarks and no multiplicity correction is applied,
                  so an isolated finding should be read cautiously.
                </p>
              </div>
            )}
          </Section>
        );
      })}
      <p className="card-sub">Generated by <code>scripts/analyze_arms.py</code> and <code>scripts/export_completion.py</code> · <span className="mono">{completion.generated_utc}</span></p>
    </>
  );
}
