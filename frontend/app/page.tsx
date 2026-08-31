"use client";

import { useEffect, useState } from "react";
import { api, type Health } from "@/lib/api";
import ApiPane from "@/components/ApiPane";
import DemoPane from "@/components/DemoPane";
import Hero from "@/components/Hero";
import MethodPane from "@/components/MethodPane";
import Nav from "@/components/Nav";
import PipelinePane from "@/components/PipelinePane";
import ResearchPane from "@/components/ResearchPane";
import { PROJECT } from "@/lib/content";

export default function Page() {
  const [health, setHealth] = useState<Health | null>(null);
  const [bootError, setBootError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api.health()
      .then((h) => { if (!cancelled) { setHealth(h); setBootError(null); } })
      .catch((e) => { if (!cancelled) setBootError(e instanceof Error ? e.message : String(e)); });
    return () => { cancelled = true; };
  }, []);

  return (
    <>
      <div className="backdrop" aria-hidden />
      <Nav apiOk={health ? true : bootError ? false : null} />

      <main className="shell">
        <Hero />
        <DemoPane health={health} bootError={bootError} />
        <ResearchPane />
        <PipelinePane />
        <MethodPane />
        <ApiPane health={health} />

        <footer className="footer">
          <p>
            <b style={{ color: "var(--text-2)" }}>{PROJECT.title} {PROJECT.subtitle}</b> ·{" "}
            {PROJECT.author} · {PROJECT.year}
          </p>
          <p>
            Skeletons only — no pixels reach the server, and no endpoint accepts one. Video you
            load is posed in your browser and discarded. Lead time is measured from an annotated
            impact frame, and a warning counts as an anticipation only if it fires inside the
            imminent window.
          </p>
          <p>
            <b style={{ color: "var(--imminent)" }}>Not a medical device.</b> A research
            prototype. The deployed checkpoint is trained on a synthetic fixture, not on real
            fall data. Public fall datasets use acted falls performed by young volunteers, so
            results are a controlled study and not a clinical guarantee.
          </p>
          <p style={{ marginTop: 14 }}>
            <a href={PROJECT.repo} target="_blank" rel="noreferrer">Source &amp; methodology ↗</a>
            {"  ·  "}
            <a href={`${PROJECT.api}/docs`} target="_blank" rel="noreferrer">API reference ↗</a>
          </p>
        </footer>
      </main>
    </>
  );
}
