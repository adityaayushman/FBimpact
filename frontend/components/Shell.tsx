"use client";

import { useEffect, useState } from "react";
import { SUBPAGES, VIEWS, hrefFor, resolveSub, type ViewId } from "@/lib/route";
import { PROJECT } from "@/lib/content";
import { Icon } from "./ui";

interface Props {
  view: ViewId;
  sub: string | null;
  apiOk: boolean | null;
  attempt: number;
  children: React.ReactNode;
}

function Mark() {
  return (
    <span className="brand-mark" aria-hidden>
      <svg width="20" height="20" viewBox="0 0 16 16" fill="none">
        <circle cx="8" cy="3" r="1.9" fill="#5A0E24" />
        <path d="M8 5.2v4.4M8 9.6l-2.6 3.6M8 9.6l2.6 3.6M4.4 6.7L11.6 6.7"
              stroke="#5A0E24" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    </span>
  );
}

/**
 * Application frame: sidebar, top bar with breadcrumb and API status, and a
 * persistent research-prototype notice. The notice lives in the frame so it cannot
 * be scrolled past or navigated away from.
 */
export default function Shell({ view, sub, apiOk, attempt, children }: Props) {
  const [open, setOpen] = useState(false);
  const groups = Array.from(new Set(VIEWS.map((v) => v.group)));
  const current = VIEWS.find((v) => v.id === view)!;
  const subLabel = (() => {
    const pages = (SUBPAGES as Record<string, readonly { id: string; label: string }[]>)[view];
    const id = resolveSub(view, sub);
    return pages?.find((p) => p.id === id)?.label ?? null;
  })();

  // Escape closes the mobile drawer.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  const apiText = apiOk === null
    ? (attempt > 1 ? `Waking API · try ${attempt}` : "Waking API…")
    : apiOk ? "API live" : "API unreachable";

  return (
    <div className="app">
      <a href="#main" className="skip">Skip to content</a>

      <aside className={`side${open ? " open" : ""}`} aria-label="Primary">
        <a href="#/overview" className="side-brand" onClick={() => setOpen(false)}>
          <Mark />
          <span className="brand-name">
            FBimpact
            <span className="brand-sub">Pre-impact fall anticipation</span>
          </span>
        </a>

        <nav aria-label="Main">
          {groups.map((g) => (
            <div key={g} className="side-group">
              <div className="side-label">{g}</div>
              {VIEWS.filter((v) => v.group === g).map((v) => (
                <a key={v.id} href={hrefFor(v.id)} title={v.hint}
                   aria-current={v.id === view ? "page" : undefined}
                   className={`side-link${v.id === view ? " active" : ""}`}
                   onClick={() => setOpen(false)}>
                  <Icon name={v.icon} />
                  <span>{v.label}</span>
                  <kbd>{v.key}</kbd>
                </a>
              ))}
            </div>
          ))}
        </nav>

        <div className="side-foot">
          <div className="links">
            <a href={PROJECT.repo} target="_blank" rel="noreferrer">Source ↗</a>
            <a href={`${PROJECT.api}/docs`} target="_blank" rel="noreferrer">API docs ↗</a>
          </div>
          <span>{PROJECT.author} · {PROJECT.year}</span>
        </div>
      </aside>

      {open && <div className="scrim" onClick={() => setOpen(false)} aria-hidden />}

      <div className="main-col">
        <header className="topbar">
          <button className="menu-btn" aria-label="Open navigation" aria-expanded={open}
                  onClick={() => setOpen((o) => !o)}>
            <Icon name="menu" />
          </button>
          <nav className="crumb" aria-label="Breadcrumb">
            <span>{current.group}</span>
            <span className="sep" aria-hidden>/</span>
            {subLabel ? <a href={hrefFor(view)} style={{ color: "inherit", textDecoration: "none" }}>{current.label}</a>
                      : <b aria-current="page">{current.label}</b>}
            {subLabel && <><span className="sep" aria-hidden>/</span><b aria-current="page">{subLabel}</b></>}
          </nav>
          <a className="status-pill" href={hrefFor("status")}
             title="The API sleeps when idle and can take up to a minute to wake">
            <span className={`status-dot ${apiOk === null ? "wait" : apiOk ? "ok" : "bad"}`} />
            {apiText}
          </a>
        </header>

        <div className="proto-bar" role="note">
          <Icon name="info" size={16} />
          <span>
            <b>Not a medical device.</b> A research prototype on public, acted-fall datasets.
            Skeletons only: no pixels reach the server.
          </span>
        </div>

        <main id="main" className="view" tabIndex={-1}>
          {children}
          <footer className="footer">
            <p>
              <b style={{ color: "var(--text-2)" }}>{PROJECT.title} {PROJECT.subtitle}</b> · {PROJECT.author} · {PROJECT.year}
            </p>
            <p>
              Lead time is measured from an annotated impact frame, and a warning counts as an
              anticipation only if it fires inside the imminent window. Public fall datasets use
              acted falls performed by young volunteers, so results describe a controlled study
              and not a clinical guarantee.
            </p>
          </footer>
        </main>
      </div>
    </div>
  );
}
