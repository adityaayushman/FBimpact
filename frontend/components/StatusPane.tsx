"use client";

import { useState } from "react";
import { API_URL, type Health } from "@/lib/api";
import ApiPane from "./ApiPane";

interface Props { health: Health | null; bootError: string | null }

/** Deployment status: reachability and round-trip latency of the inference API. */
export default function StatusPane({ health, bootError }: Props) {
  const [pings, setPings] = useState<number[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const ping = async () => {
    setBusy(true); setErr(null);
    const t0 = performance.now();
    try {
      const res = await fetch(`${API_URL}/health`, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setPings((p) => [...p.slice(-9), Math.round(performance.now() - t0)]);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const m = health?.model;
  const avg = pings.length ? Math.round(pings.reduce((a, b) => a + b, 0) / pings.length) : null;

  return (
    <>
      <section className="pane">
        <h2 className="pane-title">System status</h2>
        <div className="cols-3">
          <div className="glass card">
            <div className="card-title">Inference API</div>
            <div className="big" style={{ color: health ? "var(--calm)" : bootError ? "var(--impact)" : "var(--watch)" }}>
              {health ? "Reachable" : bootError ? "Unreachable" : "Waking…"}
            </div>
            <p className="card-sub mono">{API_URL}</p>
            {bootError && <p className="card-sub">{bootError}</p>}
          </div>
          <div className="glass card">
            <div className="card-title">Latency</div>
            <div className="big">{pings.length ? `${pings[pings.length - 1]} ms` : "—"}</div>
            <p className="card-sub">{avg != null ? `mean of last ${pings.length}: ${avg} ms` : "Not measured yet."}</p>
            <button className="btn" onClick={ping} disabled={busy}>{busy ? "Pinging…" : "Ping /health"}</button>
            {err && <p className="card-sub" style={{ color: "var(--impact)" }}>{err}</p>}
          </div>
          <div className="glass card">
            <div className="card-title">Deployed checkpoint</div>
            {m ? (
              <ul className="kv">
                <li><span>Backbone</span><b>{m.backbone}</b></li>
                <li><span>Parameters</span><b>{m.parameters.toLocaleString()}</b></li>
                <li><span>Input channels</span><b>{m.in_channels}</b></li>
                <li><span>Window</span><b>{m.window} frames</b></li>
                <li><span>Trained on</span><b>{m.trained_on}</b></li>
                <li><span>Causal</span><b>{m.causal ? "yes" : "no"}</b></li>
              </ul>
            ) : <p className="card-sub">Available once the API responds.</p>}
          </div>
        </div>
      </section>
      <ApiPane health={health} />
    </>
  );
}
