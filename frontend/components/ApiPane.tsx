"use client";

import { API_URL, type Health } from "@/lib/api";
import { DEPLOYMENT_FACTS, PROJECT } from "@/lib/content";

const ENDPOINTS = [
  { method: "GET", path: "/health", body: "Liveness, the loaded model's identity, and the frozen operating point." },
  { method: "GET", path: "/skeleton", body: "Joint names and bones, so a client draws the same graph the model convolves over." },
  { method: "GET", path: "/clips", body: "Bundled demo clips — three falls and four hard-negative daily activities." },
  { method: "POST", path: "/analyze", body: "Score a [T,17,3] skeleton sequence. Returns per-frame p(t), warnings, and joint evidence." },
  { method: "POST", path: "/faithfulness", body: "Deletion/insertion test for one warning. Separate because it costs ~150 forward passes." },
];

/**
 * Deployment and API surface.
 *
 * Included because "runs in real time on modest hardware" is a claim in the
 * proposal, and a claim about deployment should be checkable against a running
 * deployment rather than taken on trust.
 */
export default function ApiPane({ health }: { health: Health | null }) {
  return (
    <section id="api" className="pane">
      <span className="eyebrow">Deployment</span>
      <h2 className="pane-title">A public inference API</h2>
      <p className="pane-lede">
        The service accepts <b>skeletons and nothing else</b>. There is no endpoint that takes an
        image, so the privacy property is structural rather than a promise about how uploads are
        handled. It routes through the same code as the offline evaluation — if the demo and the
        reported metrics disagreed, one would be wrong with no way to tell which.
      </p>

      <div className="cols-2">
        <div className="glass pad">
          <h3 className="card-title">Endpoints</h3>
          <p className="card-sub">
            Base <code>{API_URL}</code> · OpenAPI at{" "}
            <a href={`${API_URL}/docs`} target="_blank" rel="noreferrer"
               style={{ color: "var(--accent)" }}>/docs</a>
          </p>
          <table className="data">
            <tbody>
              {ENDPOINTS.map((e) => (
                <tr key={e.path}>
                  <td className="mono" style={{ whiteSpace: "nowrap", verticalAlign: "top" }}>
                    <span style={{
                      color: e.method === "GET" ? "var(--calm)" : "var(--accent-2)",
                      fontWeight: 600, fontSize: 11,
                    }}>{e.method}</span>{" "}
                    {e.path}
                  </td>
                  <td className="dim" style={{ fontSize: 12.5 }}>{e.body}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="stack">
          <div className="glass pad">
            <h3 className="card-title">Live status</h3>
            {health ? (
              <>
                <p className="card-sub" style={{ marginBottom: 12 }}>
                  Reported by the running service.
                </p>
                <table className="data">
                  <tbody>
                    <tr><td>Backbone</td><td className="num">{health.model.backbone}</td></tr>
                    <tr><td>Parameters</td><td className="num">{health.model.parameters.toLocaleString()}</td></tr>
                    <tr><td>Causal</td><td className="num">
                      <span className={health.model.causal ? "yes" : "no"}>
                        {health.model.causal ? "yes" : "NO — lead times invalid"}
                      </span>
                    </td></tr>
                    <tr><td>Device</td><td className="num">{health.model.device}</td></tr>
                    <tr><td>Trained on</td><td className="num">{health.model.trained_on}</td></tr>
                    <tr><td>Max frames</td><td className="num">{health.limits.max_frames}</td></tr>
                  </tbody>
                </table>
              </>
            ) : (
              <div className="empty">Contacting the service…</div>
            )}
          </div>

          <div className="glass pad">
            <h3 className="card-title">Measured, not claimed</h3>
            <p className="card-sub">
              On the reference machine (RTX 4050 laptop) and the deployed free-tier container.
            </p>
            <table className="data">
              <tbody>
                {DEPLOYMENT_FACTS.map((f) => (
                  <tr key={f.label}>
                    <td>{f.label}</td>
                    <td className="num">{f.value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="glass pad" style={{ marginTop: 16 }}>
        <h3 className="card-title">Two things worth knowing before you use it</h3>
        <div className="cols-2" style={{ marginTop: 12 }}>
          <div>
            <p className="card-sub" style={{ marginBottom: 0 }}>
              <b style={{ color: "var(--text)" }}>Cold starts.</b> The container sleeps after
              about fifteen minutes idle; the next request takes 30–60&nbsp;seconds while it wakes.
              The client says so rather than reporting a network failure.
            </p>
          </div>
          <div>
            <p className="card-sub" style={{ marginBottom: 0 }}>
              <b style={{ color: "var(--text)" }}>Memory is the binding constraint.</b> Importing
              torch alone costs 403&nbsp;MB of the 512&nbsp;MB tier, so windows are scored in
              chunks of eight. Scoring a clip in one batch peaked at 681&nbsp;MB and the container
              was killed.
            </p>
          </div>
        </div>
        <p className="card-sub" style={{ marginTop: 14, marginBottom: 0 }}>
          Full deployment notes, including the measured memory and latency tables, are in{" "}
          <a href={`${PROJECT.repo}/blob/main/DEPLOY.md`} target="_blank" rel="noreferrer"
             style={{ color: "var(--accent)" }}>DEPLOY.md</a>.
        </p>
      </div>
    </section>
  );
}
