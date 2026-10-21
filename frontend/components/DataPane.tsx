"use client";

/**
 * The data behind every number on this site.
 *
 * Generated from the skeleton caches by scripts/export_datasets.py, so these
 * counts are what the models were trained and tested on rather than a
 * description of them. Shown on purpose: results without the data's size,
 * provenance and flaws are results nobody can weigh.
 */

import datasets from "@/lib/datasets.json";
import { PHASES } from "@/lib/phases";
import { SUBPAGES } from "@/lib/route";
import { PageHeader, Section, TabPanel, Tabs } from "./ui";

interface FrameRange { min: number; median: number; max: number }
interface Dataset {
  id: string; name: string; citation: string;
  clips: number; falls: number; adls: number; minutes: number; negative_minutes: number;
  fps: number[]; pose_coverage: number; annotation: string; onset_note: string;
  pre_impact_frames: FrameRange | null; onset_to_impact_frames: FrameRange | null;
  phase_frames: Record<string, number>; w_pre_frames: number;
}
interface Combined {
  clips: number; falls: number; adls: number; minutes: number; negative_minutes: number;
  phase_frames: Record<string, number>;
}

const SETS = datasets.datasets as unknown as Dataset[];
const COMBINED = datasets.combined as unknown as Combined | null;
const LIMITS = datasets.limits as string[];

function PhaseBar({ frames, label }: { frames: Record<string, number>; label: string }) {
  const total = PHASES.reduce((s, p) => s + (frames[p.name] ?? 0), 0) || 1;
  return (
    <div>
      <div className="readout" style={{ marginBottom: 5 }}>{label}</div>
      <div role="img" style={{ display: "flex", height: 16, borderRadius: 5, overflow: "hidden" }}
           aria-label={PHASES.map((p) => `${p.name} ${((frames[p.name] ?? 0) / total * 100).toFixed(1)}%`).join(", ")}>
        {PHASES.map((p) => (
          <span key={p.name} title={`${p.name}: ${frames[p.name]?.toLocaleString()} frames`}
                style={{ width: `${((frames[p.name] ?? 0) / total) * 100}%`, background: p.colour }} />
        ))}
      </div>
      <div className="legend" style={{ marginTop: 6 }}>
        {PHASES.map((p) => (
          <span key={p.name}>
            <i className="swatch" style={{ background: p.colour, height: 9 }} />
            {p.name} {((frames[p.name] ?? 0) / total * 100).toFixed(1)}%
          </span>
        ))}
      </div>
    </div>
  );
}

const range = (r: FrameRange | null) => (r ? `${r.min} / ${r.median} / ${r.max}` : "—");

export default function DataPane({ sub }: { sub: string }) {
  if (!SETS.length) return null;
  return (
    <>
      <PageHeader
        eyebrow="Evidence"
        title="What the models were trained and tested on"
        lede={<>
          Two public datasets of real video, pooled. Every figure is computed from the cached
          skeleton files that training read. Both contain falls <b>acted</b> by volunteers; none
          are recorded elderly falls, and that limit applies to everything here.
        </>}
      />

      {COMBINED && (
        <div className="stat-row" style={{ marginTop: 6 }}>
          {[
            [COMBINED.clips, "clips pooled"],
            [COMBINED.falls, "falls, each tested exactly once"],
            [COMBINED.adls, "normal-activity clips"],
            [`${COMBINED.negative_minutes} min`, "of normal footage that false alarms are counted over"],
          ].map(([v, l]) => (
            <div key={String(l)} className="card stat">
              <div className="stat-value">{v}</div>
              <div className="stat-label">{l}</div>
            </div>
          ))}
        </div>
      )}

      <div style={{ height: 28 }} />
      <Tabs view="data" tabs={SUBPAGES.data} value={sub} label="Data sections" />

      <TabPanel view="data" id="datasets" value={sub}>
        <div className="cols-2">
          {SETS.map((d) => (
            <div key={d.id} className="card">
              <h3 className="card-title">{d.name}</h3>
              <p className="card-sub">{d.citation}</p>
              <div className="table-wrap">
                <table className="data">
                  <tbody>
                    <tr><td>Clips</td><td className="num">{d.clips} ({d.falls} falls · {d.adls} normal)</td></tr>
                    <tr><td>Footage</td><td className="num">{d.minutes} min at {d.fps.join(" / ")} fps</td></tr>
                    <tr><td>Normal time</td><td className="num">{d.negative_minutes} min</td></tr>
                    <tr><td>Pose detected</td>
                        <td className="num">{(d.pose_coverage * 100).toFixed(1)}% of joints</td></tr>
                    <tr><td title="min / median / max frames before ground contact">Frames before impact</td>
                        <td className="num">{range(d.pre_impact_frames)}</td></tr>
                    <tr><td title="min / median / max frames from fall onset to impact">Onset → impact (frames)</td>
                        <td className="num">{range(d.onset_to_impact_frames)}</td></tr>
                  </tbody>
                </table>
              </div>
              <p className="card-sub" style={{ margin: "14px 0 6px" }}>
                <b style={{ color: "var(--text-2)" }}>Impact frame:</b> {d.annotation}.
              </p>
              <p className="card-sub">{d.onset_note}</p>
              <PhaseBar frames={d.phase_frames} label={`phase frames · imminent window ${d.w_pre_frames} frames`} />
            </div>
          ))}
        </div>
      </TabPanel>

      <TabPanel view="data" id="balance" value={sub}>
        {COMBINED && (
          <Section
            title="Class balance when pooled"
            sub="Falls are rare in any recording, so the phases that matter are a small share of frames."
          >
            <div className="card">
              <p className="card-sub">
                That imbalance is why the loss is class-weighted, why recall and false-alarm counts
                are reported instead of accuracy, and why a model that always says
                &ldquo;normal&rdquo; would look about 70% accurate while being useless.
              </p>
              <PhaseBar frames={COMBINED.phase_frames} label="all frames, both datasets" />
            </div>
          </Section>
        )}
      </TabPanel>

      <TabPanel view="data" id="limits" value={sub}>
        <Section title="Limits of this data" sub="What these datasets cannot tell you, however the numbers come out.">
          <div className="card">
            <ul style={{ margin: 0, paddingLeft: 20, display: "grid", gap: 10 }}>
              {LIMITS.map((l) => (
                <li key={l} style={{ color: "var(--text-2)", fontSize: 14.5, lineHeight: 1.6 }}>{l}</li>
              ))}
            </ul>
          </div>
        </Section>
      </TabPanel>
    </>
  );
}
