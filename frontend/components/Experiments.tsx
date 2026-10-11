"use client";

import { useMemo, useState } from "react";
import data from "@/lib/runs.json";

type Run = (typeof data.runs)[number];
type Key = "arm_label" | "seed" | "fold" | "recall" | "lead" | "false_alarms" | "auc";

const COLS: { key: Key; label: string; num?: boolean }[] = [
  { key: "arm_label", label: "Arm" },
  { key: "seed", label: "Seed", num: true },
  { key: "fold", label: "Fold", num: true },
  { key: "recall", label: "Recall", num: true },
  { key: "lead", label: "Lead (s)", num: true },
  { key: "false_alarms", label: "False alarms", num: true },
  { key: "auc", label: "AUC", num: true },
];

const fmt = (v: unknown) => (typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(3)) : "—");

/** Explorer over every individual run, so seed-to-seed spread is visible and not averaged away. */
export default function Experiments() {
  const benches = useMemo(() => Array.from(new Map(data.runs.map((r) => [r.benchmark, r.benchmark_name])).entries()), []);
  const [bench, setBench] = useState(benches[0]?.[0] ?? "");
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<{ key: Key; dir: 1 | -1 }>({ key: "arm_label", dir: 1 });

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    return data.runs
      .filter((r) => r.benchmark === bench && (!q || r.arm_label.toLowerCase().includes(q)))
      .sort((a, b) => {
        const x = a[sort.key] as number | string | null, y = b[sort.key] as number | string | null;
        if (x == null) return 1;
        if (y == null) return -1;
        return (x < y ? -1 : x > y ? 1 : 0) * sort.dir;
      });
  }, [bench, query, sort]);

  const spread = useMemo(() => {
    const by = new Map<string, number[]>();
    rows.forEach((r) => { if (r.false_alarms != null) by.set(r.arm_label, [...(by.get(r.arm_label) ?? []), r.false_alarms]); });
    const max = Math.max(1, ...rows.map((r) => r.false_alarms ?? 0));
    return { by: Array.from(by.entries()), max };
  }, [rows]);

  const exportCsv = () => {
    const head = ["benchmark", "arm", "seed", "fold", "role", "recall", "lead", "false_alarms", "auc"];
    const body = rows.map((r: Run) => [r.benchmark, r.arm, r.seed, r.fold, r.role, r.recall, r.lead, r.false_alarms, r.auc].join(","));
    const url = URL.createObjectURL(new Blob([[head.join(","), ...body].join("\n")], { type: "text/csv" }));
    const a = document.createElement("a");
    a.href = url; a.download = `runs_${bench}.csv`; a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section className="pane">
      <h2 className="pane-title">Experiments</h2>
      <p className="pane-sub">
        {data.runs.length} individual runs, one row per arm, seed and fold. Aggregates hide spread;
        this view does not. Only seeds 1-4 of the confirmation grid count as confirmatory evidence.
      </p>

      <div className="toolbar">
        <label>Benchmark{" "}
          <select value={bench} onChange={(e) => setBench(e.target.value)}>
            {benches.map(([id, name]) => <option key={id} value={id}>{name}</option>)}
          </select>
        </label>
        <label>Filter{" "}
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="arm name" />
        </label>
        <button className="btn" onClick={exportCsv}>Export CSV</button>
        <span className="dim mono" style={{ marginLeft: "auto" }}>{rows.length} rows</span>
      </div>

      <div className="glass card" style={{ marginBottom: 16 }}>
        <div className="card-title">False alarms per fold, by arm</div>
        {spread.by.length === 0 && <p className="card-sub">No runs for this selection.</p>}
        {spread.by.map(([arm, vals]) => (
          <div key={arm} className="strip">
            <span className="strip-label">{arm}</span>
            <span className="strip-track">
              {vals.map((v, i) => (
                <i key={i} title={`${v} false alarms`} style={{ left: `${(v / spread.max) * 100}%` }} />
              ))}
            </span>
          </div>
        ))}
      </div>

      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              {COLS.map((c) => (
                <th key={c.key} className={c.num ? "num" : ""}
                    aria-sort={sort.key === c.key ? (sort.dir === 1 ? "ascending" : "descending") : "none"}>
                  <button className="th-btn"
                          onClick={() => setSort((s) => ({ key: c.key, dir: s.key === c.key ? (s.dir === 1 ? -1 : 1) : 1 }))}>
                    {c.label}{sort.key === c.key ? (sort.dir === 1 ? " ▲" : " ▼") : ""}
                  </button>
                </th>
              ))}
              <th>Role</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={`${r.arm}-${r.seed}-${r.fold}-${i}`}>
                {COLS.map((c) => (
                  <td key={c.key} className={c.num ? "num mono" : ""}>{c.num ? fmt(r[c.key]) : String(r[c.key])}</td>
                ))}
                <td><span className="chip">{r.role}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
