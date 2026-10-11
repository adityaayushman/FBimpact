"use client";

import { useEffect, useState } from "react";
import { api, type Health } from "@/lib/api";
import { useRoute } from "@/lib/route";
import DataPane from "@/components/DataPane";
import DemoPane from "@/components/DemoPane";
import Experiments from "@/components/Experiments";
import MethodPane from "@/components/MethodPane";
import Overview from "@/components/Overview";
import PipelinePane from "@/components/PipelinePane";
import ResearchPane from "@/components/ResearchPane";
import ResultsPane from "@/components/ResultsPane";
import Shell from "@/components/Shell";
import StatusPane from "@/components/StatusPane";

export default function Page() {
  const [view] = useRoute();
  const [health, setHealth] = useState<Health | null>(null);
  const [bootError, setBootError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    api.health(3, (n) => { if (!cancelled) setAttempt(n); })
      .then((h) => { if (!cancelled) { setHealth(h); setBootError(null); } })
      .catch((e) => { if (!cancelled) setBootError(e instanceof Error ? e.message : String(e)); });
    return () => { cancelled = true; };
  }, []);

  return (
    <>
      <div className="backdrop" aria-hidden />
      <Shell view={view} apiOk={health ? true : bootError ? false : null} attempt={attempt}>
        {view === "overview" && <Overview />}
        {view === "analyzer" && <DemoPane health={health} bootError={bootError} waking={!health && !bootError} />}
        {view === "results" && <ResultsPane />}
        {view === "experiments" && <Experiments />}
        {view === "data" && <DataPane />}
        {view === "method" && <><PipelinePane /><MethodPane /></>}
        {view === "research" && <ResearchPane />}
        {view === "status" && <StatusPane health={health} bootError={bootError} />}
      </Shell>
    </>
  );
}
