"use client";

import { useCallback, useEffect, useState } from "react";

export const VIEWS = [
  { id: "overview", label: "Overview", key: "1", group: "Project", hint: "Headline state of the evidence" },
  { id: "analyzer", label: "Live analyzer", key: "2", group: "Project", hint: "Run the model on a skeleton sequence or your own video" },
  { id: "results", label: "Results", key: "3", group: "Evidence", hint: "Aggregated benchmarks, operating curves, confirmation test" },
  { id: "experiments", label: "Experiments", key: "4", group: "Evidence", hint: "Every individual run, with seed-to-seed spread" },
  { id: "generalisation", label: "Generalisation", key: "5", group: "Evidence", hint: "Does the result hold in a place the model has never seen?" },
  { id: "data", label: "Data", key: "6", group: "Evidence", hint: "The datasets, their labelling and limits" },
  { id: "method", label: "Method", key: "7", group: "Reference", hint: "Pipeline, model and evaluation protocol" },
  { id: "research", label: "Research", key: "8", group: "Reference", hint: "Question, claims and what was not shown" },
  { id: "status", label: "System status", key: "9", group: "Reference", hint: "API health, latency and deployed checkpoint" },
] as const;

export type ViewId = (typeof VIEWS)[number]["id"];

const IDS = new Set<string>(VIEWS.map((v) => v.id));

/** `#/results` -> "results". Anything unknown falls back to the overview. */
export function parseHash(hash: string): ViewId {
  const id = hash.replace(/^#\/?/, "").split(/[/?]/)[0];
  return (IDS.has(id) ? id : "overview") as ViewId;
}

/** Legacy section anchors (`#demo`, `#pipeline`, ...) from earlier links keep working. */
const LEGACY: Record<string, ViewId> = {
  demo: "analyzer", pipeline: "method", api: "status", overview: "overview",
};

export function useRoute(): [ViewId, (id: ViewId) => void] {
  const [view, setView] = useState<ViewId>("overview");

  useEffect(() => {
    const read = () => {
      const raw = window.location.hash.replace(/^#\/?/, "");
      setView(LEGACY[raw] ?? parseHash(window.location.hash));
    };
    read();
    window.addEventListener("hashchange", read);
    return () => window.removeEventListener("hashchange", read);
  }, []);

  const go = useCallback((id: ViewId) => {
    window.location.hash = `/${id}`;
  }, []);

  // Keyboard: 1-9 jump between views, unless the user is typing somewhere.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable)) return;
      const hit = VIEWS.find((v) => v.key === e.key);
      if (hit) go(hit.id);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go]);

  // Moving between views should land at the top, not mid-scroll of the last one.
  useEffect(() => { window.scrollTo({ top: 0 }); }, [view]);

  return [view, go];
}
