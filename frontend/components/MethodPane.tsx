"use client";

import { ABLATIONS, DECISIONS, METRICS } from "@/lib/content";
import { SUBPAGES } from "@/lib/route";
import { PageHeader, Section, TabPanel, Tabs } from "./ui";
import PipelinePane from "./PipelinePane";

/** A drawn definition of lead time, the project's headline quantity. */
function LeadTimeDiagram() {
  const W = 760, H = 150;
  const pad = { l: 16, r: 16, t: 34 };
  const x = (f: number) => pad.l + (f / 100) * (W - pad.l - pad.r);
  const impact = 78;
  const wPre = 20;
  const warn = impact - 14;
  const y = pad.t + 36;            // the timeline
  const boxH = 72;                 // height of the shaded windows

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", display: "block" }}
         role="img" aria-label="Lead time is the interval between the warning frame and the impact frame">
      {/* imminent window, labelled above its box */}
      <rect x={x(impact - wPre)} y={pad.t} width={x(impact) - x(impact - wPre)} height={boxH}
            fill="var(--mocha)" opacity={0.28} rx={4} />
      <text x={(x(impact - wPre) + x(impact)) / 2} y={pad.t - 10} textAnchor="middle"
            fontSize={11} fill="var(--text-2)" fontFamily="var(--mono)">
        imminent window [t*−W, t*]
      </text>

      {/* after impact: masked out of scoring */}
      <rect x={x(impact)} y={pad.t} width={(W - pad.r) - x(impact)} height={boxH}
            fill="var(--text-3)" opacity={0.1} rx={4} />
      <text x={(x(impact) + W - pad.r) / 2} y={pad.t - 10} textAnchor="middle"
            fontSize={11} fill="var(--text-3)" fontFamily="var(--mono)">masked</text>

      {/* timeline */}
      <line x1={pad.l} y1={y} x2={W - pad.r} y2={y} stroke="var(--text-3)" strokeWidth={1.4} />
      <text x={pad.l} y={y + 24} fontSize={11} fill="var(--text-3)" fontFamily="var(--mono)">
        normal activity
      </text>

      {/* warning and impact markers, labelled above the line */}
      <line x1={x(warn)} y1={pad.t + 4} x2={x(warn)} y2={y + 10} stroke="var(--imminent)" strokeWidth={1.8} />
      <circle cx={x(warn)} cy={y} r={5} fill="var(--imminent)" />
      <text x={x(warn) - 8} y={y - 12} textAnchor="end" fontSize={11.5} fontWeight={600}
            fill="var(--imminent)" fontFamily="var(--mono)">t_warn</text>

      <line x1={x(impact)} y1={pad.t + 4} x2={x(impact)} y2={y + 10} stroke="var(--impact)" strokeWidth={2.2} />
      <circle cx={x(impact)} cy={y} r={5} fill="var(--impact)" />
      <text x={x(impact) + 8} y={y - 12} textAnchor="start" fontSize={11.5} fontWeight={600}
            fill="var(--impact)" fontFamily="var(--mono)">t* impact</text>

      {/* the measured span, drawn beneath the timeline with its own label */}
      <line x1={x(warn)} y1={y + 34} x2={x(impact)} y2={y + 34} stroke="var(--imminent)" strokeWidth={2.5} />
      <line x1={x(warn)} y1={y + 28} x2={x(warn)} y2={y + 40} stroke="var(--imminent)" strokeWidth={2} />
      <line x1={x(impact)} y1={y + 28} x2={x(impact)} y2={y + 40} stroke="var(--imminent)" strokeWidth={2} />
      <text x={(x(warn) + x(impact)) / 2} y={y + 60} textAnchor="middle"
            fontSize={12.5} fill="var(--imminent)" fontWeight={650} fontFamily="var(--mono)">
        lead time = (t* − t_warn) / fps
      </text>
    </svg>
  );
}

/**
 * Methodology: how the headline number is defined, and every judgement call
 * that the definition leaves open.
 *
 * The decisions table is the most important thing on this page. Each of those
 * choices moves the reported lead time, and each is the kind of detail a paper
 * usually leaves implicit — which is precisely why two systems reporting
 * "0.5 s lead time" can be measuring different quantities.
 */
export default function MethodPane({ sub }: { sub: string }) {
  return (
    <>
      <PageHeader
        eyebrow="Reference"
        title="How it works, and how it is measured"
        lede="The pipeline from video to a verified warning, the precise definition of the headline number, and every judgement call that definition leaves open."
      />
      <Tabs view="method" tabs={SUBPAGES.method} value={sub} label="Method sections" />
      <TabPanel view="method" id="pipeline" value={sub}>
        <PipelinePane />
      </TabPanel>
      <TabPanel view="method" id="lead-time" value={sub}>
        <Section title="Lead time, defined precisely" sub={<>For each fall clip, the impact frame <code>t*</code> — first ground contact — is annotated by hand. Frames in <code>[t* − W, t*]</code> are positive, everything before is negative, and every frame of a normal-activity clip is negative. A warning fires when the score holds at or above τ for k consecutive frames. Lead time is reported only over falls that were correctly warned; a fall with no warning before <code>t*</code> is a miss.</>}>
      <div className="card">
        <LeadTimeDiagram />
      </div>

      
      
        </Section>
      </TabPanel>
      <TabPanel view="method" id="decisions" value={sub}>
        <Section title="Judgement calls, made explicit" sub={<>The definition above still leaves five things open, and each one changes the reported number. They are named options in the code with documented defaults, not silent conventions — which is why two systems can both report &ldquo;0.5&nbsp;s lead time&rdquo; and not be measuring the same quantity.</>}>
      <div className="stack">
        {DECISIONS.map((d) => (
          <div key={d.question} className="card">
            <div style={{ display: "flex", gap: 14, flexWrap: "wrap", alignItems: "baseline", marginBottom: 8 }}>
              <h3 className="card-title" style={{ margin: 0, flex: "1 1 300px" }}>{d.question}</h3>
              <span className="chip" style={{
                borderColor: "var(--accent-line)", background: "var(--accent-dim)",
                color: "var(--accent)", fontWeight: 600,
              }}>
                {d.answer}
              </span>
            </div>
            <p className="card-sub" style={{ margin: 0 }}>{d.why}</p>
          </div>
        ))}
      </div>

      
      
        </Section>
      </TabPanel>
      <TabPanel view="method" id="metrics" value={sub}>
        <Section title="What is measured" sub={<>Recall leads, because a missed fall is the costly error. A single balanced-accuracy figure would flatter the method and tell a care setting nothing it needs.</>}>
      <div className="cols-3">
        {METRICS.map((m) => (
          <div key={m.name} className="card" style={
            m.primary ? { borderColor: "var(--accent-line)" } : undefined
          }>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
              <h3 className="card-title" style={{ margin: 0, fontSize: 14.5 }}>{m.name}</h3>
              {m.primary && (
                <span className="mono" style={{ fontSize: 9.5, letterSpacing: "0.1em",
                  color: "var(--accent)", padding: "2px 6px", borderRadius: 100,
                  background: "var(--accent-dim)", border: "1px solid var(--accent-line)" }}>
                  PRIMARY
                </span>
              )}
            </div>
            <p className="card-sub" style={{ margin: 0, fontSize: 12.5 }}>{m.body}</p>
          </div>
        ))}
      </div>

      
      
        </Section>
      </TabPanel>
      <TabPanel view="method" id="ablations" value={sub}>
        <Section title="Ablation plan" sub={<>Each variant removes exactly one component and re-measures both anticipation and faithfulness. Every cell of the results table is a mean ± standard deviation over seeds &#123;0,&nbsp;1,&nbsp;2&#125; — a single run cannot distinguish a real effect from seed noise.</>}>
      <div className="card">
        <div className="table-scroll">
          <table className="data">
            <thead>
              <tr>
                <th>Variant</th>
                <th>Change from the full model</th>
                <th>Isolates</th>
              </tr>
            </thead>
            <tbody>
              {ABLATIONS.map((a, i) => (
                <tr key={a.variant} className={i === 0 ? "ours" : undefined}>
                  <td className="mono" style={{ whiteSpace: "nowrap" }}>{a.variant}</td>
                  <td className="dim">{a.change}</td>
                  <td>{a.isolates}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="card-sub" style={{ marginTop: 12, marginBottom: 0 }}>
          The <code>− pre-impact loss</code> row is the direct test of RQ3: setting λ to zero
          recovers plain class-weighted cross-entropy <em>exactly</em>, verified by a unit test,
          so architecture, data, schedule and seeds are untouched and the difference in lead time
          is attributable to the time weighting alone.
        </p>
      </div>
    
        </Section>
      </TabPanel>
    </>
  );
}
