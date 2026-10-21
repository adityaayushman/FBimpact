"use client";

import { useRef } from "react";
import { hrefFor, type ViewId } from "@/lib/route";

/* -- icons ----------------------------------------------------------------- */

const PATHS: Record<string, string> = {
  home: "M3 10.5 12 3l9 7.5M5 9.5V20h5v-6h4v6h5V9.5",
  pulse: "M3 12h4l2.5-7 4 14 2.5-7H21",
  chart: "M4 20V4M4 20h16M8 16v-4M12 16V8M16 16v-6",
  flask: "M9 3h6M10 3v6l-5 9a2 2 0 0 0 1.8 3h10.4a2 2 0 0 0 1.8-3l-5-9V3M8 15h8",
  globe: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18ZM3 12h18M12 3c2.5 2.6 3.8 5.6 3.8 9s-1.3 6.4-3.8 9c-2.5-2.6-3.8-5.6-3.8-9S9.5 5.6 12 3Z",
  database: "M12 8c4.4 0 8-1.3 8-3s-3.6-3-8-3-8 1.3-8 3 3.6 3 8 3ZM4 5v7c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12v7c0 1.7 3.6 3 8 3s8-1.3 8-3v-7",
  layers: "m12 3 9 5-9 5-9-5 9-5ZM3 13l9 5 9-5M3 17.5l9 5 9-5",
  book: "M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2V5ZM4 19a2 2 0 0 0 2 2h13M9 7h6",
  server: "M4 5h16v6H4zM4 13h16v6H4zM8 8h.01M8 16h.01",
  menu: "M4 6h16M4 12h16M4 18h16",
  info: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18ZM12 11v5M12 8h.01",
  arrow: "M5 12h14M13 6l6 6-6 6",
  external: "M14 4h6v6M20 4 10 14M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5",
};

export function Icon({ name, size = 18 }: { name: string; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
         strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d={PATHS[name] ?? PATHS.info} />
    </svg>
  );
}

/* -- page structure ---------------------------------------------------------- */

/** The title block every page opens with: one h1, a plain-language lede, optional actions. */
export function PageHeader({
  eyebrow, title, lede, actions,
}: { eyebrow?: string; title: string; lede?: React.ReactNode; actions?: React.ReactNode }) {
  return (
    <header className="page-head">
      <div className="titles">
        {eyebrow && <span className="eyebrow">{eyebrow}</span>}
        <h1 className="page-title">{title}</h1>
        {lede && <p className="page-lede">{lede}</p>}
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </header>
  );
}

/** A titled block within a page. Headings are h2 so the page outline stays h1 > h2 > h3. */
export function Section({
  id, title, sub, actions, children,
}: { id?: string; title?: string; sub?: React.ReactNode; actions?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="section" id={id}>
      {(title || actions) && (
        <div className="section-head">
          <div>
            {title && <h2 className="section-title">{title}</h2>}
            {sub && <p className="section-sub">{sub}</p>}
          </div>
          {actions}
        </div>
      )}
      {children}
    </section>
  );
}

/* -- tabs ---------------------------------------------------------------------- */

export interface TabDef { id: string; label: string; count?: number }

/**
 * Accessible tab bar (WAI-ARIA tabs pattern): arrow keys move between tabs, Home
 * and End jump to the ends, and only the selected tab is in the tab order.
 * Selection lives in the URL, so a tab can be linked to and survives a reload.
 */
export function Tabs({
  view, tabs, value, label,
}: { view: ViewId; tabs: readonly TabDef[]; value: string; label: string }) {
  const refs = useRef<(HTMLAnchorElement | null)[]>([]);

  const onKeyDown = (e: React.KeyboardEvent, index: number) => {
    const last = tabs.length - 1;
    const next =
      e.key === "ArrowRight" ? (index === last ? 0 : index + 1)
      : e.key === "ArrowLeft" ? (index === 0 ? last : index - 1)
      : e.key === "Home" ? 0
      : e.key === "End" ? last
      : null;
    if (next === null) return;
    e.preventDefault();
    refs.current[next]?.focus();
    window.location.hash = hrefFor(view, tabs[next].id).slice(1);
  };

  return (
    <div className="tabs">
      <div className="tablist" role="tablist" aria-label={label}>
        {tabs.map((t, i) => (
          <a
            key={t.id}
            ref={(el) => { refs.current[i] = el; }}
            id={`tab-${view}-${t.id}`}
            role="tab"
            href={hrefFor(view, t.id)}
            className="tab"
            aria-selected={t.id === value}
            aria-controls={`panel-${view}-${t.id}`}
            tabIndex={t.id === value ? 0 : -1}
            onKeyDown={(e) => onKeyDown(e, i)}
          >
            {t.label}
            {t.count != null && <span className="count">{t.count}</span>}
          </a>
        ))}
      </div>
    </div>
  );
}

export function TabPanel({ view, id, value, children }: { view: ViewId; id: string; value: string; children: React.ReactNode }) {
  if (id !== value) return null;
  return (
    <div role="tabpanel" id={`panel-${view}-${id}`} aria-labelledby={`tab-${view}-${id}`} className="tabpanel" tabIndex={-1}>
      {children}
    </div>
  );
}
