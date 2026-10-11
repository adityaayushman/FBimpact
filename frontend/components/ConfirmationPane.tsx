"use client";

/**
 * The confirmatory test of the combined-benchmark result.
 *
 * The single-seed result above generated a hypothesis: richer features, then
 * phase supervision, each cut false alarms. A hypothesis cannot confirm itself,
 * so a grid of fresh seeds is run against decision rules that were committed
 * before any of it was read (results/analysis/PLAN.txt).
 *
 * Until that grid finishes this component says so in plain words. When it has,
 * the wording shown for each contrast comes from `analysis.json`, where it was
 * chosen mechanically from the verdict - not written here - so no component can
 * promote "not shown" into "improves".
 */

import analysis from "@/lib/analysis.json";

interface Contrast {
  contrast: string; means: string; seeds: number; clips: number;
  d_fa: number | null; d_fa_lo: number | null; d_fa_hi: number | null;
  d_recall: number | null; d_recall_lo: number | null; d_recall_hi: number | null;
  seeds_fewer_fa: string; matched_recall: string;
  verdict: string; level: string; wording: string;
}

const CONTRASTS = (analysis.contrasts ?? []) as unknown as Contrast[];
const PENDING = analysis.status !== "complete" || CONTRASTS.length === 0;

const LEVEL_COLOUR: Record<string, string> = {
  supported: "var(--calm)",
  partial: "var(--watch)",
  pending: "var(--watch)",
  not_shown: "var(--text-3)",
};

const sign = (v: number | null, d = 1) =>
  v === null ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(d)}`;

export default function ConfirmationPane() {
  return (
    <div className="glass pad">
      <h3 className="card-title">Does it replicate? A pre-registered confirmation</h3>
      <p className="card-sub">
        The single-seed combined result is a hypothesis. It is being tested on fresh seeds (1–4),
        with the question, the contrasts and the decision rules committed to the repository
        before any of those runs existed. Seed 0, which produced the hypothesis, is shown above
        and never counted as evidence for it.
      </p>

      {PENDING ? (
        <div className="notice warn">
          <b>Confirmation is not complete.</b> Until it is, the combined-benchmark numbers above
          are one seed each, the recall differences are 1–4 falls of 129, and nothing here should
          be read as &ldquo;improves&rdquo; or &ldquo;more accurate&rdquo;. The grid is four arms
          × five folds × four seeds; this panel fills in as it finishes.
        </div>
      ) : (
        <>
          <div className="table-scroll">
            <table className="data">
              <thead>
                <tr>
                  <th>Contrast</th>
                  <th className="num">Δ false alarms (95% CI)</th>
                  <th className="num">Δ recall (95% CI)</th>
                  <th className="num">Seeds with fewer FA</th>
                  <th>Matched recall</th>
                </tr>
              </thead>
              <tbody>
                {CONTRASTS.map((c) => (
                  <tr key={c.contrast}>
                    <td title={c.means}>
                      <b>{c.contrast}</b>
                      <div className="dim" style={{ fontSize: 11.5 }}>{c.means}</div>
                    </td>
                    <td className="num">
                      {sign(c.d_fa)} [{sign(c.d_fa_lo)}, {sign(c.d_fa_hi)}]
                    </td>
                    <td className="num">
                      {sign(c.d_recall, 3)} [{sign(c.d_recall_lo, 3)}, {sign(c.d_recall_hi, 3)}]
                    </td>
                    <td className="num">{c.seeds_fewer_fa}</td>
                    <td>{c.matched_recall}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="stack" style={{ marginTop: 14 }}>
            {CONTRASTS.map((c) => (
              <div key={c.contrast} className="notice"
                   style={{ borderLeft: `3px solid ${LEVEL_COLOUR[c.level] ?? "var(--text-3)"}` }}>
                <b>{c.contrast}.</b> {c.wording}
              </div>
            ))}
          </div>

          <p className="card-sub" style={{ marginTop: 12, marginBottom: 0 }}>
            Intervals resample clips once and reuse the same resample for every seed, because the seeds
            are repeated measurements of the same {CONTRASTS[0]?.clips} clips rather than more clips.
            Splits are clip-independent, not subject-independent, so none of this speaks to people the
            model has never seen.
          </p>
        </>
      )}
    </div>
  );
}
