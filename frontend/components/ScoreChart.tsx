"use client";

/**
 * The per-frame imminence trace - the project's headline quantity, drawn.
 *
 * Shows p_t against time with the decision threshold, the imminent window
 * [t*-W_pre, t*], the impact frame, and every trigger. The gap between a
 * trigger marker and the impact line *is* the lead time, so the number in the
 * warning card and the picture cannot disagree.
 *
 * Plain SVG: the whole chart is four paths and a handful of lines, and a
 * charting library would cost more bundle than it saves in code.
 */

import { useMemo } from "react";
import type { Analysis } from "@/lib/api";

interface Props {
  analysis: Analysis;
  frame: number;
  onScrub?: (frame: number) => void;
  height?: number;
}

const PAD = { top: 12, right: 14, bottom: 26, left: 34 };

export default function ScoreChart({ analysis, frame, onScrub, height = 190 }: Props) {
  const { scores, fps, impact_frame, imminent_window, decision, warnings } = analysis;
  const width = 900; // viewBox units; the SVG scales to its container
  const plotW = width - PAD.left - PAD.right;
  const plotH = height - PAD.top - PAD.bottom;
  const n = scores.length;

  const x = (f: number) => PAD.left + (n <= 1 ? 0 : (f / (n - 1)) * plotW);
  const y = (p: number) => PAD.top + (1 - p) * plotH;

  const path = useMemo(
    () => scores.map((p, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(2)},${y(p).toFixed(2)}`).join(" "),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [scores, n, plotW, plotH]
  );

  const area = useMemo(() => `${path} L${x(n - 1)},${y(0)} L${x(0)},${y(0)} Z`, [path, n]);

  const handleScrub = (event: React.MouseEvent<SVGSVGElement>) => {
    if (!onScrub) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const ratio = ((event.clientX - rect.left) / rect.width) * width;
    const f = Math.round(((ratio - PAD.left) / plotW) * (n - 1));
    onScrub(Math.max(0, Math.min(n - 1, f)));
  };

  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const seconds = Math.max(1, Math.round(n / fps));
  const timeTicks = Array.from({ length: seconds + 1 }, (_, s) => s).filter(
    (s) => seconds <= 8 || s % Math.ceil(seconds / 8) === 0
  );

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      style={{ width: "100%", height: "auto", display: "block", cursor: onScrub ? "crosshair" : "default" }}
      onClick={handleScrub}
      role="img"
      aria-label={`Fall-imminence score over ${(n / fps).toFixed(1)} seconds`}
    >
      {/* Imminent window: the only interval in which a warning counts as an
          anticipation rather than a false alarm. */}
      {imminent_window && (
        <rect
          x={x(imminent_window[0])}
          y={PAD.top}
          width={Math.max(x(imminent_window[1]) - x(imminent_window[0]), 1)}
          height={plotH}
          fill="var(--accent)"
          opacity={0.09}
        />
      )}

      {ticks.map((t) => (
        <g key={t}>
          <line x1={PAD.left} y1={y(t)} x2={width - PAD.right} y2={y(t)} stroke="var(--grid)" strokeWidth={1} />
          <text x={PAD.left - 6} y={y(t) + 3.5} textAnchor="end" fontSize={10} fill="var(--text-faint)" fontFamily="var(--mono)">
            {t.toFixed(2)}
          </text>
        </g>
      ))}

      {timeTicks.map((s) => (
        <text
          key={s}
          x={x(s * fps)}
          y={height - 8}
          textAnchor="middle"
          fontSize={10}
          fill="var(--text-faint)"
          fontFamily="var(--mono)"
        >
          {s}s
        </text>
      ))}

      <path d={area} fill="var(--accent)" opacity={0.1} />
      <path d={path} fill="none" stroke="var(--accent)" strokeWidth={1.6} strokeLinejoin="round" />

      {/* Decision threshold. */}
      <line
        x1={PAD.left}
        y1={y(decision.threshold)}
        x2={width - PAD.right}
        y2={y(decision.threshold)}
        stroke="var(--imminent)"
        strokeWidth={1.2}
        strokeDasharray="5 4"
      />
      <text x={width - PAD.right} y={y(decision.threshold) - 4} textAnchor="end" fontSize={10} fill="var(--imminent)" fontFamily="var(--mono)">
        τ={decision.threshold.toFixed(2)}
      </text>

      {/* Impact. */}
      {impact_frame !== null && (
        <g>
          <line x1={x(impact_frame)} y1={PAD.top} x2={x(impact_frame)} y2={PAD.top + plotH} stroke="var(--impact)" strokeWidth={1.6} />
          <text x={x(impact_frame) - 5} y={PAD.top + 10} textAnchor="end" fontSize={10} fill="var(--impact)" fontWeight={600} fontFamily="var(--mono)">
            impact
          </text>
        </g>
      )}

      {/* Triggers, and the lead-time span each one bought. */}
      {warnings.map((w) => (
        <g key={w.frame}>
          {impact_frame !== null && w.lead_time !== null && (
            <line
              x1={x(w.frame)}
              y1={y(decision.threshold)}
              x2={x(impact_frame)}
              y2={y(decision.threshold)}
              stroke="var(--imminent)"
              strokeWidth={3}
              opacity={0.35}
            />
          )}
          <line
            x1={x(w.frame)}
            y1={PAD.top}
            x2={x(w.frame)}
            y2={PAD.top + plotH}
            stroke="var(--imminent)"
            strokeWidth={1.2}
            strokeDasharray="2 3"
          />
          <circle cx={x(w.frame)} cy={y(w.score)} r={4} fill="var(--imminent)" stroke="var(--surface)" strokeWidth={1.5} />
          {w.lead_time !== null && (
            <text
              x={x(w.frame) + 5}
              y={PAD.top + 10}
              fontSize={10.5}
              fill="var(--imminent)"
              fontWeight={640}
              fontFamily="var(--mono)"
            >
              {w.lead_time.toFixed(2)}s lead
            </text>
          )}
        </g>
      ))}

      {/* Playhead. */}
      <line x1={x(frame)} y1={PAD.top} x2={x(frame)} y2={PAD.top + plotH} stroke="var(--text)" strokeWidth={1} opacity={0.5} />
      <circle cx={x(frame)} cy={y(scores[Math.min(frame, n - 1)] ?? 0)} r={3} fill="var(--text)" />
    </svg>
  );
}
