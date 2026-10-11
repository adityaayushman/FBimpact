"use client";

import data from "@/lib/generalisation.json";

type Protocol = (typeof data.protocols)[number];
type Run = { seed: number; recall: number | null; false_alarms: number | null; normal_hours: number | null };

const pct = (v: number | null | undefined) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);
const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);

function RecallStrip({ runs, reference }: { runs: Run[]; reference: number | null }) {
  return (
    <span className="strip-track" aria-hidden>
      {reference != null && (
        <i title={`in-domain ${pct(reference)}`}
           style={{ left: `${reference * 100}%`, background: "var(--text-3)", borderRadius: 2, width: 3, height: 13, top: 1 }} />
      )}
      {runs.map((r) => r.recall != null && (
        <i key={r.seed} title={`seed ${r.seed}: ${pct(r.recall)}`} style={{ left: `${r.recall * 100}%` }} />
      ))}
    </span>
  );
}

function ProtocolCard({ p }: { p: Protocol }) {
  return (
    <div className="glass card">
      <div className="card-title">{p.title}</div>
      <p className="card-sub">Train: {p.train}. Test: {p.test}.</p>

      <table className="data" style={{ marginBottom: 12 }}>
        <thead><tr><th>Split</th><th className="num">Clips</th><th className="num">Falls</th><th className="num">Normal min</th></tr></thead>
        <tbody>
          {(["train", "val", "test"] as const).map((s) => (
            <tr key={s}>
              <td>{s}</td>
              <td className="num mono">{p.sizes[s].clips}</td>
              <td className="num mono">{p.sizes[s].falls}</td>
              <td className="num mono">{p.sizes[s].normal_minutes}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {(Object.entries(p.arms) as [string, Protocol["arms"]["ctrl"]][]).map(([id, arm]) => {
        const runs = arm.runs as Run[];
        const recalls = runs.map((r) => r.recall).filter((v): v is number => v != null);
        const alarms = runs.map((r) => r.false_alarms).filter((v): v is number => v != null);
        const ref = arm.in_domain?.recall ?? null;
        return (
          <div key={id} style={{ marginTop: 10 }}>
            <div className="strip">
              <span className="strip-label">{arm.label}</span>
              {runs.length ? <RecallStrip runs={runs} reference={ref} />
                           : <span className="chip">pending · 0 of 3 seeds</span>}
            </div>
            {runs.length > 0 && (
              <p className="card-sub mono" style={{ margin: "2px 0 0 0" }}>
                recall {pct(mean(recalls))} over {runs.length} seed{runs.length > 1 ? "s" : ""}
                {" · "}false alarms {alarms.join(", ")}
                {ref != null && <> {" · "}in-domain {pct(ref)} (gap {pct((mean(recalls) ?? 0) - ref)})</>}
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}

/** Held-out-domain results: does the in-domain behaviour carry over to an unseen place? */
export default function Generalisation() {
  const label = data.status === "complete" ? "complete"
    : data.status === "partial" ? "partial" : "not yet run";
  return (
    <section className="pane">
      <h2 className="pane-title">Generalisation</h2>
      <p className="pane-sub">
        Random splits inside one dataset let a model see the same rooms and cameras on both sides.
        These protocols hold a whole dataset or scene out, which is what a deployment faces: a place
        the model has never seen. Protocol and the claims it can support were fixed before any run
        (<span className="mono">{data.plan}</span>).
      </p>

      <p className={`notice ${data.status === "complete" ? "" : "warn"}`}>
        <b>Status: {label}.</b> {data.runs_done} of {data.runs_expected} runs finished (4 protocols
        × 2 arms × 3 seeds). Dots are individual seeds; the grey tick is the in-domain pooled recall
        for the same arm. Nothing here is estimated for cells that have not run.
      </p>

      <div className="cols-2">
        {data.protocols.map((p) => <ProtocolCard key={p.id} p={p} />)}
      </div>

      <div className="glass card" style={{ marginTop: 16 }}>
        <div className="card-title">What these numbers can and cannot say</div>
        <ul className="card-sub" style={{ paddingLeft: 18, margin: "8px 0 0" }}>
          {data.limits.map((l) => <li key={l}>{l}</li>)}
        </ul>
      </div>
    </section>
  );
}
