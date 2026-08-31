"use client";

import {
  CONTRIBUTIONS,
  GUARDS,
  LIMITATIONS,
  POSITIONING,
  RESEARCH_QUESTIONS,
} from "@/lib/content";

function Cell({ value }: { value: boolean | string }) {
  if (value === true) return <span className="yes">Yes</span>;
  if (value === false) return <span className="no">No</span>;
  return <span className="no">{value}</span>;
}

/**
 * The research showcase: what the project claims, why the claim is unusual,
 * and what would falsify it.
 *
 * Ordered as a reviewer reads rather than as the work was built — the gap
 * first, because "pre-impact + privacy + verified explanation" is the only
 * reason the rest is interesting.
 */
export default function ResearchPane() {
  return (
    <section id="research" className="pane">
      <span className="eyebrow">Research</span>
      <h2 className="pane-title">The gap this fills</h2>
      <p className="pane-lede">
        The fall-monitoring field is crowded, but concentrated in three places this project
        deliberately avoids. Most published systems recognise a fall <em>after</em> it happens.
        A second cluster achieves genuine pre-impact warning but requires a wearable inertial
        sensor, which older adults frequently forget, refuse or remove. A third uses raw RGB
        video, which is unacceptable in the bedrooms and bathrooms where most serious falls
        occur. The intersection — vision-based, pre-impact, privacy-preserving, and with an
        explanation that is <em>scored</em> for faithfulness — is empty.
      </p>

      <div className="glass pad">
        <div className="table-scroll">
          <table className="data">
            <thead>
              <tr>
                <th>Dimension</th>
                {POSITIONING.columns.map((c) => (
                  <th key={c} className={c === "This project" ? "" : ""}>{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {POSITIONING.rows.map((row) => (
                <tr key={row.dimension}>
                  <td>{row.dimension}</td>
                  {row.values.map((v, i) => (
                    <td key={i} style={i === 3 ? { background: "var(--accent-dim)" } : undefined}>
                      <Cell value={v} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="card-sub" style={{ marginTop: 12, marginBottom: 0 }}>
          To be explicit about what is <em>not</em> claimed: this work performs no causal
          inference. Free-text causal narratives and counterfactual percentages cannot be
          validated from passive video, which contains no interventions and no counterfactual
          ground truth. Only two things are kept, because only two are identifiable from the
          data — <b>temporal precedence</b> (which cues precede the fall) and{" "}
          <b>evidence grounding</b> (which joints the model relied on).
        </p>
      </div>

      {/* -- research questions ------------------------------------------- */}
      <h2 className="pane-title" style={{ marginTop: 52 }}>Research questions</h2>
      <p className="pane-lede">
        Three questions, each answerable against ground truth. Anything that could not be
        checked was left out of scope on purpose.
      </p>

      <div className="stack">
        {RESEARCH_QUESTIONS.map((rq) => (
          <div key={rq.id} className="glass pad">
            <div className="rq">
              <span className="rq-tag">{rq.id}</span>
              <div>
                <h3 className="card-title" style={{ marginBottom: 8 }}>{rq.question}</h3>
                <p className="card-sub" style={{ margin: 0 }}>{rq.approach}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* -- contributions ------------------------------------------------- */}
      <h2 className="pane-title" style={{ marginTop: 52 }}>Contribution</h2>
      <p className="pane-lede">
        Deliberately narrow, so each part can be executed and defended. The novelty budget is
        spent entirely on the anticipation objective and the verified explanation; perception
        is treated as solved plumbing and reused frozen.
      </p>

      <div className="cols-2">
        {CONTRIBUTIONS.map((c) => (
          <div key={c.kind} className="glass pad">
            <span className="eyebrow" style={{ marginBottom: 12 }}>{c.kind}</span>
            <h3 className="card-title" style={{ marginTop: 12 }}>{c.title}</h3>
            <p className="card-sub" style={{ marginBottom: 0 }}>{c.body}</p>
          </div>
        ))}
      </div>

      {/* -- validity guards ----------------------------------------------- */}
      <h2 className="pane-title" style={{ marginTop: 52 }}>What protects the result</h2>
      <p className="pane-lede">
        Three properties that, if silently broken, would make every reported number meaningless
        while producing no error at all. Each is enforced in code and asserted in the test suite,
        because each fails silently — and two of them fail in the flattering direction.
      </p>

      <div className="cols-3">
        {GUARDS.map((g) => (
          <div key={g.title} className="glass pad">
            <h3 className="card-title">{g.title}</h3>
            <p className="card-sub">{g.body}</p>
            <p style={{
              margin: 0, paddingTop: 11, borderTop: "1px solid var(--glass-edge)",
              fontSize: 12.5, color: "var(--imminent)", lineHeight: 1.5,
            }}>
              {g.consequence}
            </p>
          </div>
        ))}
      </div>

      {/* -- limitations --------------------------------------------------- */}
      <h2 className="pane-title" style={{ marginTop: 52 }}>Honest boundary of the claim</h2>
      <p className="pane-lede">
        The edge here is pre-impact timing, privacy, efficiency and verifiable explanation — not
        raw benchmark accuracy. These limits are stated in the paper rather than in a footnote.
      </p>

      <div className="glass pad">
        <ul style={{ margin: 0, paddingLeft: 20, display: "grid", gap: 11 }}>
          {LIMITATIONS.map((l) => (
            <li key={l} style={{ color: "var(--text-2)", fontSize: 14, lineHeight: 1.6 }}>{l}</li>
          ))}
        </ul>
      </div>
    </section>
  );
}
