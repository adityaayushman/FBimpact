"use client";

/**
 * Section 10's operating-point curve.
 *
 * A single threshold is always open to the charge of having been chosen after
 * seeing the test set, so the whole sweep is published instead. Each fold's
 * checkpoint scored its own held-out clips and the streams were pooled, so
 * every clip appears exactly once and none is scored by a model that trained on
 * it.
 *
 * Plotted against the **absolute** false-alarm count rather than a per-hour
 * rate: the pooled negative time is under six minutes, so one trigger moves the
 * rate by ~52/hour and a rate axis would imply a precision the data cannot
 * support.
 *
 * Only the Pareto frontier is drawn. The raw sweep contains many dominated
 * points - a threshold that costs more false alarms *and* catches fewer falls -
 * and drawing them turns the comparison into a scatter of noise.
 */

import curvesData from "@/lib/curves.json";

interface Point {
  threshold: number; persistence: number; recall: number;
  mean_lead_time: number | null; false_alarms: number;
  false_alarms_per_hour: number; specificity: number; num_falls: number;
}
interface Curve {
  variant: string; folds: number; falls: number;
  negative_minutes: number; points: Point[];
}

const CURVES = (curvesData.curves ?? {}) as unknown as Record<string, Curve>;

const STYLE: Record<string, { label: string; colour: string }> = {
  baseline_stgcn: { label: "Baseline (plain loss)", colour: "var(--calm)" },
  no_preimpact_loss: { label: "− pre-impact loss (deployed)", colour: "var(--accent)" },
  ours_preimpact: { label: "Ours (pre-impact loss)", colour: "var(--imminent)" },
};

const W = 720, H = 300;
const PAD = { top: 16, right: 18, bottom: 44, left: 52 };

/** Points not beaten on both axes: fewer false alarms and higher recall. */
function frontier(points: Point[]): Point[] {
  const sorted = [...points].sort(
    (a, b) => a.false_alarms - b.false_alarms || b.recall - a.recall
  );
  const out: Point[] = [];
  let best = -1;
  for (const p of sorted) {
    if (p.recall > best) {
      out.push(p);
      best = p.recall;
    }
  }
  return out;
}

export default function OperatingCurve() {
  const names = Object.keys(CURVES).filter((n) => CURVES[n]?.points?.length);
  if (!names.length) return null;

  const any = CURVES[names[0]];
  const maxFA = Math.max(
    12,
    ...names.flatMap((n) => frontier(CURVES[n].points).map((p) => p.false_alarms))
  );
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const x = (fa: number) => PAD.left + Math.min(fa / maxFA, 1) * plotW;
  const y = (r: number) => PAD.top + (1 - r) * plotH;

  return (
    <div className="glass pad">
      <h3 className="card-title">Operating-point curve — every threshold, not one</h3>
      <p className="card-sub">
        Recall against the number of false alarms, over {any.falls} falls and{" "}
        {any.negative_minutes} minutes of normal activity pooled from all {any.folds} folds.
        Each fold&rsquo;s model scored only its own held-out clips.
      </p>

      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", display: "block" }}
           role="img" aria-label="Recall against false-alarm count for each variant">
        {[0, 0.25, 0.5, 0.75, 1].map((r) => (
          <g key={r}>
            <line x1={PAD.left} y1={y(r)} x2={W - PAD.right} y2={y(r)} stroke="var(--grid-line)" />
            <text x={PAD.left - 8} y={y(r) + 3.5} textAnchor="end" fontSize={10}
                  fill="var(--text-4)" fontFamily="var(--mono)">{r.toFixed(2)}</text>
          </g>
        ))}
        {Array.from({ length: 7 }, (_, i) => Math.round((i / 6) * maxFA)).map((fa) => (
          <text key={fa} x={x(fa)} y={H - 24} textAnchor="middle" fontSize={10}
                fill="var(--text-4)" fontFamily="var(--mono)">{fa}</text>
        ))}
        <text x={PAD.left + plotW / 2} y={H - 8} textAnchor="middle" fontSize={11}
              fill="var(--text-3)">false alarms in {any.negative_minutes} min of normal activity</text>
        <text x={14} y={PAD.top + plotH / 2} textAnchor="middle" fontSize={11}
              fill="var(--text-3)" transform={`rotate(-90 14 ${PAD.top + plotH / 2})`}>
          recall
        </text>

        {names.map((name) => {
          const style = STYLE[name] ?? { label: name, colour: "var(--text-3)" };
          const pts = frontier(CURVES[name].points);
          // Step path: between two frontier points, recall only improves once
          // the extra false alarms have been paid for.
          const d = pts.map((p, i) =>
            i === 0
              ? `M${x(p.false_alarms)},${y(p.recall)}`
              : `L${x(p.false_alarms)},${y(pts[i - 1].recall)} L${x(p.false_alarms)},${y(p.recall)}`
          ).join(" ");
          return (
            <g key={name}>
              <path d={d} fill="none" stroke={style.colour} strokeWidth={2}
                    strokeLinejoin="round" />
              {pts.map((p) => (
                <circle key={`${p.threshold}-${p.persistence}`} cx={x(p.false_alarms)}
                        cy={y(p.recall)} r={3} fill={style.colour}
                        stroke="var(--bg-0)" strokeWidth={1}>
                  <title>
                    {style.label}: recall {p.recall.toFixed(3)},{" "}
                    {p.false_alarms} false alarm{p.false_alarms === 1 ? "" : "s"},{" "}
                    lead {p.mean_lead_time?.toFixed(3) ?? "—"} s (τ={p.threshold.toFixed(2)}, k={p.persistence})
                  </title>
                </circle>
              ))}
            </g>
          );
        })}
      </svg>

      <div className="legend">
        {names.map((n) => (
          <span key={n}>
            <i className="swatch" style={{ background: (STYLE[n] ?? {}).colour ?? "var(--text-3)" }} />
            {(STYLE[n] ?? {}).label ?? n}
          </span>
        ))}
      </div>

      <p className="card-sub" style={{ marginTop: 12, marginBottom: 0 }}>
        The baseline curve sits above both others across the whole sweep, so the negative
        result is not an artefact of the frozen operating point — it holds at every threshold.
        The baseline also reaches <b>zero false alarms in these 5.8 minutes at 0.60 recall and
        0.44 s of lead</b>. Whether that survives hours of real activity is not something six
        minutes of negative data can establish.
      </p>
    </div>
  );
}
