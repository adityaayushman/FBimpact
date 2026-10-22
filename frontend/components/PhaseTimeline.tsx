"use client";

/**
 * The model's four-phase output over time: normal, imminent, falling, grounded.
 *
 * What this is and is not:
 *  - It is the phase head's own per-frame probabilities, stacked so each column
 *    sums to one. It is never derived from the imminence score; a checkpoint
 *    without a phase head shows an explanation instead of a picture.
 *  - It is model output, not ground truth. The label says so.
 *  - Phases do not trigger alarms. The warning still comes from the binary
 *    imminence score, so lead time stays comparable with every reported number.
 *
 * Each phase carries a text label as well as a colour (Okabe-Ito, which stays
 * distinguishable under the common forms of colour-blindness), and the same
 * information is available as a sentence summary and a table.
 */

import { useMemo } from "react";
import type { Analysis } from "@/lib/api";
import { PHASES } from "@/lib/phases";

const W = 900;
const H = 150;
const PAD = { top: 10, right: 14, bottom: 26, left: 34 };
const MAX_COLUMNS = 300; // stack is drawn per column; longer clips are strided

interface Props {
  analysis: Analysis;
  frame: number;
  onScrub?: (frame: number) => void;
  /** Shown beside the title: the phase head's benefit was not confirmed on fresh seeds. */
  preliminary?: boolean;
}

interface Segment {
  phase: string;
  from: number;
  to: number;
}

/** Contiguous runs of the most probable phase, for the text summary. */
function segments(probs: number[][], names: string[]): Segment[] {
  const out: Segment[] = [];
  probs.forEach((row, t) => {
    const phase = names[row.indexOf(Math.max(...row))];
    const last = out[out.length - 1];
    if (last && last.phase === phase) last.to = t;
    else out.push({ phase, from: t, to: t });
  });
  return out;
}

export default function PhaseTimeline({ analysis, frame, onScrub, preliminary }: Props) {
  const { phases, fps, frames, impact_frame, warnings, interpolated_frames } = analysis;

  const model = useMemo(() => {
    if (!phases || !phases.probs.length) return null;
    const n = phases.probs.length;
    const stride = Math.max(1, Math.ceil(n / MAX_COLUMNS));
    const cols = phases.probs.filter((_, i) => i % stride === 0 || i === n - 1);
    return { n, stride, cols, segs: segments(phases.probs, phases.names) };
  }, [phases]);

  if (!phases || !model) {
    return (
      <div className="notice" role="note">
        <b>No phase output for this checkpoint.</b> The deployed model was trained without a
        four-phase head, so there is nothing to draw — the phase view is never reconstructed from
        the imminence score. The score chart above is the model&rsquo;s whole output.
      </div>
    );
  }

  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const x = (f: number) => PAD.left + (frames <= 1 ? 0 : (f / (frames - 1)) * plotW);
  const y = (p: number) => PAD.top + (1 - p) * plotH;

  // Cumulative bottom-up stack: band k spans [cum_{k-1}, cum_k].
  const bands = PHASES.map((phase, k) => {
    const idx = phases.names.indexOf(phase.name);
    const top: string[] = [];
    const bottom: string[] = [];
    model.cols.forEach((row, c) => {
      const t = Math.min(c * model.stride, frames - 1);
      const lo = phases.names.slice(0, idx).reduce((s, _, j) => s + row[j], 0);
      const hi = lo + (idx >= 0 ? row[idx] : 0);
      top.push(`${x(t).toFixed(1)},${y(hi).toFixed(1)}`);
      bottom.unshift(`${x(t).toFixed(1)},${y(lo).toFixed(1)}`);
    });
    return { ...phase, k, d: `M${top.join(" L")} L${bottom.join(" L")} Z` };
  });

  const handleScrub = (event: React.MouseEvent<SVGSVGElement>) => {
    if (!onScrub) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const px = ((event.clientX - rect.left) / rect.width) * W;
    onScrub(Math.max(0, Math.min(frames - 1, Math.round(((px - PAD.left) / plotW) * (frames - 1)))));
  };

  const current = phases.probs[Math.min(frame, phases.probs.length - 1)] ?? [];
  const seconds = (f: number) => (f / fps).toFixed(1);
  const summary = model.segs
    .map((s) => `${s.phase} ${seconds(s.from)}–${seconds(s.to + 1)} s`)
    .join(", then ");

  return (
    <div>
      <div style={{ display: "flex", gap: 10, alignItems: "baseline", flexWrap: "wrap", marginBottom: 6 }}>
        <h3 className="card-title" style={{ margin: 0 }}>Phase timeline</h3>
        <span className="readout">model output, not ground truth</span>
        {preliminary && (
          <span className="chip" style={{ borderColor: "var(--watch)", color: "var(--watch)" }}>
            preliminary · gain not confirmed
          </span>
        )}
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: "100%", height: "auto", display: "block", cursor: onScrub ? "crosshair" : "default" }}
        onClick={handleScrub}
        role="img"
        aria-label={`Phase probabilities over ${(frames / fps).toFixed(1)} seconds. ${summary}.`}
      >
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line x1={PAD.left} y1={y(t)} x2={W - PAD.right} y2={y(t)} stroke="var(--grid-line)" />
            <text x={PAD.left - 6} y={y(t) + 3.5} textAnchor="end" fontSize={10}
                  fill="var(--text-4)" fontFamily="var(--mono)">{t.toFixed(1)}</text>
          </g>
        ))}

        {bands.map((b) => (
          <path key={b.name} d={b.d} fill={b.colour} opacity={0.88} stroke="var(--bg-0)" strokeWidth={0.4} />
        ))}

        {/* Frames that were missing and bridged: hatched, so a smooth-looking
            stretch is never mistaken for observed data. */}
        {(interpolated_frames ?? []).map((f) => (
          <rect key={f} x={x(f)} y={PAD.top} width={Math.max(plotW / frames, 1.5)} height={plotH}
                fill="var(--bg-0)" opacity={0.55} />
        ))}

        {impact_frame !== null && (
          <g>
            <line x1={x(impact_frame)} y1={PAD.top} x2={x(impact_frame)} y2={PAD.top + plotH}
                  stroke="var(--impact)" strokeWidth={1.8} />
            <text x={x(impact_frame) - 4} y={PAD.top + 10} textAnchor="end" fontSize={10}
                  fill="var(--impact)" fontWeight={600} fontFamily="var(--mono)">impact</text>
          </g>
        )}
        {warnings.map((w) => (
          <line key={w.frame} x1={x(w.frame)} y1={PAD.top} x2={x(w.frame)} y2={PAD.top + plotH}
                stroke="var(--text)" strokeWidth={1.4} strokeDasharray="3 3" />
        ))}

        <line x1={x(frame)} y1={PAD.top} x2={x(frame)} y2={PAD.top + plotH}
              stroke="var(--text)" strokeWidth={1.2} opacity={0.8} />

        {[0, 0.25, 0.5, 0.75, 1].map((r) => (
          <text key={r} x={PAD.left + r * plotW} y={H - 8} textAnchor="middle" fontSize={10}
                fill="var(--text-4)" fontFamily="var(--mono)">{seconds(r * (frames - 1))}s</text>
        ))}
      </svg>

      <div className="legend" aria-label="Phase colours">
        {PHASES.map((p) => (
          <span key={p.name} title={p.blurb}>
            <i className="swatch" style={{ background: p.colour, height: 9 }} />
            {p.name}
          </span>
        ))}
        <span><i className="swatch" style={{ background: "var(--text)", width: 2, height: 11 }} /> warning</span>
        {interpolated_frames && interpolated_frames.length > 0 && (
          <span>shaded = frames bridged over missing detections</span>
        )}
      </div>

      <p className="card-sub" style={{ margin: "8px 0 0" }}>
        At the cursor:{" "}
        {PHASES.map((p) => {
          const k = phases.names.indexOf(p.name);
          return `${p.name} ${k >= 0 && current[k] !== undefined ? Math.round(current[k] * 100) : 0}%`;
        }).join(" · ")}
        . Phases are displayed only; the warning above still comes from the imminence score.
      </p>

      <details style={{ marginTop: 8 }}>
        <summary className="card-sub" style={{ cursor: "pointer" }}>Text and table version</summary>
        <p className="card-sub">Most probable phase over time: {summary}.</p>
        <div className="table-scroll">
          <table className="data">
            <thead>
              <tr><th>From</th><th>To</th><th>Most probable phase</th></tr>
            </thead>
            <tbody>
              {model.segs.map((s, i) => (
                <tr key={i}>
                  <td className="num">{seconds(s.from)} s</td>
                  <td className="num">{seconds(s.to + 1)} s</td>
                  <td>{s.phase}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  );
}
