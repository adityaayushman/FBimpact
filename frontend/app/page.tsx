"use client";

import { useEffect, useState } from "react";
import { api, type Health } from "@/lib/api";
import { resolveSub, useRoute } from "@/lib/route";
import DataPane from "@/components/DataPane";
import DemoPane from "@/components/DemoPane";
import Experiments from "@/components/Experiments";
import Generalisation from "@/components/Generalisation";
import MethodPane from "@/components/MethodPane";
import Overview from "@/components/Overview";
import ResearchPane from "@/components/ResearchPane";
import ResultsPane from "@/components/ResultsPane";
import Shell from "@/components/Shell";
import StatusPane from "@/components/StatusPane";

export default function Page() {
  const { view, sub: rawSub, ready } = useRoute();
  const sub = resolveSub(view, rawSub) ?? "";
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
      <Shell view={view} sub={rawSub} apiOk={health ? true : bootError ? false : null} attempt={attempt}>
        {ready && view === "overview" && <Overview />}
        {ready && view === "analyzer" && <DemoPane health={health} bootError={bootError} waking={!health && !bootError} />}
        {ready && view === "results" && <ResultsPane sub={sub} />}
        {ready && view === "experiments" && <Experiments />}
        {ready && view === "generalisation" && <Generalisation />}
        {ready && view === "data" && <DataPane sub={sub} />}
        {ready && view === "method" && <MethodPane sub={sub} />}
        {ready && view === "research" && <ResearchPane sub={sub} />}
        {ready && view === "status" && <StatusPane sub={sub} health={health} bootError={bootError} />}
      </Shell>
    </>
  );
}
