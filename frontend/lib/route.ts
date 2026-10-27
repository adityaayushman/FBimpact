"use client";

import { useCallback, useEffect, useState } from "react";

/** Sub-pages of a view, shown as tabs. The first entry is the default. */
export const SUBPAGES = {
  results: [
    { id: "findings", label: "Findings" },
    { id: "tables", label: "Benchmark tables" },
    { id: "confirmation", label: "Confirmation test" },
    { id: "explanations", label: "Explanations" },
  ],
  data: [
    { id: "datasets", label: "Datasets" },
    { id: "balance", label: "Class balance" },
    { id: "limits", label: "Limits" },
  ],
  method: [
    { id: "pipeline", label: "Pipeline" },
    { id: "lead-time", label: "Lead time" },
    { id: "decisions", label: "Design decisions" },
    { id: "metrics", label: "Metrics" },
    { id: "ablations", label: "Ablation plan" },
  ],
  research: [
    { id: "gap", label: "The gap" },
    { id: "questions", label: "Questions" },
    { id: "contribution", label: "Contribution" },
    { id: "safeguards", label: "Safeguards" },
  ],
  status: [
    { id: "health", label: "Health" },
    { id: "model-card", label: "Model card" },
    { id: "api", label: "API" },
  ],
} as const;

export type SubpageView = keyof typeof SUBPAGES;

export const VIEWS = [
  { id: "overview", label: "Overview", key: "1", group: "Project", icon: "home", hint: "Headline state of the evidence" },
  { id: "analyzer", label: "Live analyzer", key: "2", group: "Project", icon: "pulse", hint: "Run the model on a skeleton sequence or your own video" },
  { id: "results", label: "Results", key: "3", group: "Evidence", icon: "chart", hint: "Aggregated benchmarks, operating curves, confirmation test" },
  { id: "experiments", label: "Experiments", key: "4", group: "Evidence", icon: "flask", hint: "Every individual run, with seed-to-seed spread" },
  { id: "generalisation", label: "Generalisation", key: "5", group: "Evidence", icon: "globe", hint: "Does the result hold in a place the model has never seen?" },
  { id: "data", label: "Data", key: "6", group: "Evidence", icon: "database", hint: "The datasets, their labelling and limits" },
  { id: "method", label: "Method", key: "7", group: "Reference", icon: "layers", hint: "Pipeline, model and evaluation protocol" },
  { id: "research", label: "Research", key: "8", group: "Reference", icon: "book", hint: "Question, claims and what was not shown" },
  { id: "status", label: "System status", key: "9", group: "Reference", icon: "server", hint: "API health, latency and deployed checkpoint" },
] as const;

export type ViewId = (typeof VIEWS)[number]["id"];

const IDS = new Set<string>(VIEWS.map((v) => v.id));

/** Anchors from earlier versions of the site keep working. */
const LEGACY: Record<string, string> = {
  demo: "analyzer", pipeline: "method/pipeline", api: "status/api", overview: "overview",
};

export interface Route { view: ViewId; sub: string | null }

/** `#/results/tables` -> { view: "results", sub: "tables" }. Unknown input falls back to the overview. */
export function parseHash(hash: string): Route {
  let raw = hash.replace(/^#\/?/, "");
  if (LEGACY[raw]) raw = LEGACY[raw];
  const [view, sub] = raw.split(/[?]/)[0].split("/");
  if (!IDS.has(view)) return { view: "overview", sub: null };
  return { view: view as ViewId, sub: sub || null };
}

/** The sub-page to show: the one in the URL if it exists for this view, else the first. */
export function resolveSub(view: ViewId, sub: string | null): string | null {
  const pages = (SUBPAGES as Record<string, readonly { id: string }[]>)[view];
  if (!pages) return null;
  return pages.some((p) => p.id === sub) ? sub : pages[0].id;
}

export function hrefFor(view: ViewId, sub?: string | null): string {
  return `#/${view}${sub ? `/${sub}` : ""}`;
}

export function useRoute(): { view: ViewId; sub: string | null; ready: boolean; go: (view: ViewId, sub?: string | null) => void } {
  const [route, setRoute] = useState<Route>({ view: "overview", sub: null });
  // The hash is only readable in the browser, so the first render cannot know the page.
  // `ready` lets the caller hold back content until it does, instead of flashing the overview.
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const read = () => setRoute(parseHash(window.location.hash));
    read();
    setReady(true);
    window.addEventListener("hashchange", read);
    return () => window.removeEventListener("hashchange", read);
  }, []);

  const go = useCallback((view: ViewId, sub?: string | null) => {
    window.location.hash = hrefFor(view, sub).slice(1);
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

  // Moving between views lands at the top; moving between tabs keeps the scroll.
  useEffect(() => { window.scrollTo({ top: 0 }); }, [route.view]);

  return { ...route, ready, go };
}
