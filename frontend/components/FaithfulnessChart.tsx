"use client";

/**
 * The RQ2 answer, drawn: deletion and insertion curves against a random control.
 *
 * Neither curve means anything alone. A model whose score barely depends on any
 * one joint gives flat curves and a middling area under both, which is why the
 * dashed random-ordering curve is always drawn alongside and why the reported
 * number is the *gap* between them. A gap at or below zero is a real result -
 * the explanation looked plausible and explained nothing - not a failed run.
 */

import type { Faithfulness } from "@/lib/api";

interface Props {
  data: Faithfulness;
}

const PAD = { top: 12, right: 12, bottom: 26, left: 32 };
const W = 420;
const H = 170;

function curve(values: number[]): string {
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const n = values.length;
  return values
    .map((v, i) => {
      const x = PAD.left + (i / (n - 1)) * plotW;
      const y = PAD.top + (1 - Math.max(0, Math.min(1, v))) * plotH;
      return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

function Panel({
  title,
  subtitle,
  ours,
  random,
  gap,
  better,
}: {
  title: string;
  subtitle: string;
  ours: number[];
  random: number[];
  gap: number;
  better: string;
}) {
  const plotH = H - PAD.top - PAD.bottom;
  const good = gap > 0;

  return (
    <div style={{ flex: "1 1 300px", minWidth: 0 }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 2 }}>
        <strong style={{ fontSize: 13 }}>{title}</strong>
        <span
          style={{
            fontFamily: "var(--mono)",
            fontSize: 12,
            fontWeight: 640,
            color: good ? "var(--calm)" : "var(--watch)",
          }}
        >
          gap {gap >= 0 ? "+" : ""}
          {gap.toFixed(3)}
        </span>
      </div>
      <p style={{ margin: "0 0 4px", fontSize: 11.5, color: "var(--text-3)", lineHeight: 1.45 }}>
        {subtitle} <span style={{ color: "var(--text-4)" }}>({better})</span>
      </p>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: "100%", height: "auto", display: "block" }}
        role="img"
        aria-label={`${title} curve against a random joint ordering`}
      >
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line
              x1={PAD.left}
              y1={PAD.top + (1 - t) * plotH}
              x2={W - PAD.right}
              y2={PAD.top + (1 - t) * plotH}
              stroke="var(--grid-line)"
            />
            <text
              x={PAD.left - 5}
              y={PAD.top + (1 - t) * plotH + 3.5}
              textAnchor="end"
              fontSize={9.5}
              fill="var(--text-4)"
              fontFamily="var(--mono)"
            >
              {t.toFixed(1)}
            </text>
          </g>
        ))}

        <path d={curve(random)} fill="none" stroke="var(--text-4)" strokeWidth={1.4} strokeDasharray="4 3" />
        <path d={curve(ours)} fill="none" stroke={good ? "var(--calm)" : "var(--watch)"} strokeWidth={2} />

        <text x={PAD.left} y={H - 8} fontSize={9.5} fill="var(--text-4)" fontFamily="var(--mono)">
          0 joints
        </text>
        <text x={W - PAD.right} y={H - 8} textAnchor="end" fontSize={9.5} fill="var(--text-4)" fontFamily="var(--mono)">
          17
        </text>
      </svg>
    </div>
  );
}

export default function FaithfulnessChart({ data }: Props) {
  return (
    <div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 20 }}>
        <Panel
          title="Deletion"
          subtitle="Score as the highest-relevance joints are removed first."
          ours={data.deletion}
          random={data.deletion_random}
          gap={data.deletion_gap}
          better="steeper drop is better"
        />
        <Panel
          title="Insertion"
          subtitle="Score as joints are added back in relevance order."
          ours={data.insertion}
          random={data.insertion_random}
          gap={data.insertion_gap}
          better="faster rise is better"
        />
      </div>

      <div className="legend">
        <span>
          <i className="swatch" style={{ background: data.deletion_gap > 0 ? "var(--calm)" : "var(--watch)" }} />
          ranked by relevance
        </span>
        <span>
          <i className="swatch" style={{ background: "var(--text-4)" }} />
          random joint order (control)
        </span>
        <span className="mono">
          {data.num_random} random orderings · baseline &ldquo;{data.baseline}&rdquo; ·{" "}
          {(data.latency_ms / 1000).toFixed(1)}s
        </span>
      </div>

      <p style={{ marginTop: 10, marginBottom: 0, fontSize: 12.5, color: "var(--text-3)", lineHeight: 1.55 }}>
        {data.faithful ? (
          <>
            Both gaps are positive: removing the named joints collapses the score faster than
            removing random ones, and restoring them recovers it faster. The explanation is doing
            work, not decorating.
          </>
        ) : (
          <>
            At least one gap is not positive, so this ranking does not beat a random joint ordering
            on this warning &mdash; the explanation is{" "}
            <b>plausible but not demonstrably faithful</b>. That is a reportable result, and the
            reason the test exists rather than the attention weights being shown on their own.
          </>
        )}
      </p>
    </div>
  );
}
