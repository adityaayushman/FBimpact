"use client";

import { ABLATIONS, DECISIONS, METRICS } from "@/lib/content";

/** A drawn definition of lead time — the project's headline quantity. */
function LeadTimeDiagram() {
  const W = 760, H = 132;
  const pad = { l: 16, r: 16, t: 26, b: 30 };
  const x = (f: number) => pad.l + (f / 100) * (W - pad.l - pad.r);
  const impact = 78;
  const wPre = 20;
  const warn = impact - 14;
  const y = pad.t + 30;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", display: "block" }}
         role="img" aria-label="Lead time is the interval between the warning frame and the impact frame">
      {/* imminent window */}
      <rect x={x(impact - wPre)} y={pad.t} width={x(impact) - x(impact - wPre)} height={46}
            fill="var(--accent)" opacity={0.16} rx={3} />
      <text x={(x(impact - wPre) + x(impact)) / 2} y={pad.t - 8} textAnchor="middle"
            fontSize={10.5} fill="var(--accent)" fontFamily="var(--mono)">
        imminent window [t*−W, t*]
      </text>

      {/* timeline */}
      <line x1={pad.l} y1={y} x2={W - pad.r} y2={y} stroke="var(--text-4)" strokeWidth={1.2} />
      <text x={pad.l} y={y + 22} fontSize={10.5} fill="var(--text-4)" fontFamily="var(--mono)">
        normal activity
      </text>

      {/* warning */}
      <line x1={x(warn)} y1={pad.t - 2} x2={x(warn)} y2={y + 12} stroke="var(--imminent)" strokeWidth={1.6} />
      <circle cx={x(warn)} cy={y} r={4.5} fill="var(--imminent)" />
      <text x={x(warn)} y={y + 26} textAnchor="middle" fontSize={10.5}
            fill="var(--imminent)" fontFamily="var(--mono)">t_warn</text>

      {/* impact */}
      <line x1={x(impact)} y1={pad.t - 2} x2={x(impact)} y2={y + 12} stroke="var(--impact)" strokeWidth={2} />
      <circle cx={x(impact)} cy={y} r={4.5} fill="var(--impact)" />
      <text x={x(impact)} y={y + 26} textAnchor="middle" fontSize={10.5}
            fill="var(--impact)" fontFamily="var(--mono)">t* impact</text>

      {/* the measured span */}
      <line x1={x(warn)} y1={y - 17} x2={x(impact)} y2={y - 17}
            stroke="var(--imminent)" strokeWidth={2.5} opacity={0.75} />
      <text x={(x(warn) + x(impact)) / 2} y={y - 23} textAnchor="middle"
            fontSize={11.5} fill="var(--imminent)" fontWeight={640} fontFamily="var(--mono)">
        lead time = (t* − t_warn) / fps
      </text>

      {/* post impact */}
      <rect x={x(impact)} y={pad.t} width={(W - pad.r) - x(impact)} height={46}
            fill="var(--text-4)" opacity={0.1} rx={3} />
      <text x={(x(impact) + W - pad.r) / 2} y={pad.t + 28} textAnchor="middle"
            fontSize={10.5} fill="var(--text-4)" fontFamily="var(--mono)">masked</text>
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
export default function MethodPane() {
  return (
    <section id="method" className="pane">
      <span className="eyebrow">Method</span>
      <h2 className="pane-title">Lead time, defined precisely</h2>
      <p className="pane-lede">
        For each fall clip, the impact frame <code>t*</code> — first ground contact — is
        annotated by hand. Frames in <code>[t* − W, t*]</code> are positive, everything before is
        negative, and every frame of a normal-activity clip is negative. A warning fires when the
        score holds at or above τ for k consecutive frames. Lead time is reported only over falls
        that were correctly warned; a fall with no warning before <code>t*</code> is a miss.
      </p>

      <div className="glass pad">
        <LeadTimeDiagram />
      </div>

      {/* -- decisions ----------------------------------------------------- */}
      <h2 className="pane-title" style={{ marginTop: 52 }}>Judgement calls, made explicit</h2>
      <p className="pane-lede">
        The definition above still leaves five things open, and each one changes the reported
        number. They are named options in the code with documented defaults, not silent
        conventions — which is why two systems can both report &ldquo;0.5&nbsp;s lead time&rdquo;
        and not be measuring the same quantity.
      </p>

      <div className="stack">
        {DECISIONS.map((d) => (
          <div key={d.question} className="glass pad">
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

      {/* -- metrics ------------------------------------------------------- */}
      <h2 className="pane-title" style={{ marginTop: 52 }}>What is measured</h2>
      <p className="pane-lede">
        Recall leads, because a missed fall is the costly error. A single balanced-accuracy figure
        would flatter the method and tell a care setting nothing it needs.
      </p>

      <div className="cols-3">
        {METRICS.map((m) => (
          <div key={m.name} className="glass pad-sm" style={
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

      {/* -- ablations ----------------------------------------------------- */}
      <h2 className="pane-title" style={{ marginTop: 52 }}>Ablation plan</h2>
      <p className="pane-lede">
        Each variant removes exactly one component and re-measures both anticipation and
        faithfulness. Every cell of the results table is a mean ± standard deviation over seeds
        &#123;0,&nbsp;1,&nbsp;2&#125; — a single run cannot distinguish a real effect from seed noise.
      </p>

      <div className="glass pad">
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
    </section>
  );
}
