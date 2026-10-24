"use client";

import data from "@/lib/generalisation.json";
import { PageHeader, Section } from "./ui";

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
    <div className="card">
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
              <td className="num mono">{p.sizes[s].normal_minutes.toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {(Object.entries(p.arms) as [string, Protocol["arms"]["ctrl"]][]).map(([id, arm]) => {
        const runs = arm.runs as Run[];
        const recalls = runs.map((r) => r.recall).filter((v): v is number => v != null);
        const alarms = runs.map((r) => r.false_alarms).filter((v): v is number => v != null);
        const perHour = runs
          .map((r) => (r.false_alarms != null && r.normal_hours ? r.false_alarms / r.normal_hours : null))
          .filter((v): v is number => v != null);
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
                {perHour.length > 0 && <> (≈ {Math.round(mean(perHour) ?? 0)} per hour)</>}
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
    <>
      <PageHeader eyebrow="Evidence" title="Generalisation" lede={<>Random splits inside one dataset let a model see the same rooms and cameras on both sides. These protocols hold a whole dataset or scene out, which is what a deployment faces: a place the model has never seen. Protocol and the claims it can support were fixed before any run (<span className="mono">{data.plan}</span>).</>} />


      <p className={`notice ${data.status === "complete" ? "" : "warn"}`}>
        <b>Status: {label}.</b> {data.runs_done} of {data.runs_expected} runs finished (4 protocols
        × 2 arms × 3 seeds). Dots are individual seeds; the grey tick is the in-domain pooled recall
        for the same arm. Nothing here is estimated for cells that have not run.
      </p>

      {data.status === "complete" && (
        <Section title="What the held-out runs show">
          <div className="stack">
            <div className="card">
              <div className="card-title">Recall depends heavily on the domain, in both directions</div>
              <p className="card-sub">
                Between a quarter and 0.4 when training on UR Fall alone or leaving out the Coffee
                room, but about 0.9 for Le2i to UR Fall and when leaving out the Home scene. It did
                not fall uniformly below the in-domain figure.
              </p>
            </div>
            <div className="card">
              <div className="card-title">High recall did not mean a good detector</div>
              <p className="card-sub">
                False alarms stay in the hundreds per hour in every protocol (shown on each card).
                The high recalls were bought with a permissive threshold, chosen on the training
                domain&apos;s validation clips, that does not transfer to a new place.
              </p>
            </div>
            <div className="card">
              <div className="card-title">The four-phase arm shows no consistent benefit</div>
              <p className="card-sub">
                Slightly higher recall in two protocols and lower in two. Trained on UR Fall alone it
                collapsed in one seed (recall 0.03). With three seeds and test sets this small none of
                the differences is distinguishable from noise, which matches the confirmation result.
              </p>
            </div>
            <div className="card">
              <div className="card-title">Why the domains differ so much is not established</div>
              <p className="card-sub">
                The training sets differ in size (UR Fall alone has only 22 training falls) and the test
                sets differ in difficulty (every UR Fall fall has the same fixed 30-frame onset). These
                runs measure the size of the shift, not its cause.
              </p>
            </div>
          </div>
        </Section>
      )}

      <div className="cols-2" style={{ marginTop: 28 }}>
        {data.protocols.map((p) => <ProtocolCard key={p.id} p={p} />)}
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <div className="card-title">What these numbers can and cannot say</div>
        <ul className="card-sub" style={{ paddingLeft: 18, margin: "8px 0 0" }}>
          {data.limits.map((l) => <li key={l}>{l}</li>)}
        </ul>
      </div>
    </>
  );
}
