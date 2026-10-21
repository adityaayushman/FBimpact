"use client";

import { useState } from "react";
import { VIEWS, type ViewId } from "@/lib/route";
import { PROJECT } from "@/lib/content";

interface Props {
  view: ViewId;
  apiOk: boolean | null;
  attempt: number;
  children: React.ReactNode;
}

function Mark() {
  return (
    <span className="brand-mark" aria-hidden>
      <svg width="15" height="15" viewBox="0 0 16 16" fill="none">
        <circle cx="8" cy="3" r="1.9" fill="#3A0F34" />
        <path d="M8 5.2v4.4M8 9.6l-2.6 3.6M8 9.6l2.6 3.6M4.4 6.7L11.6 6.7"
              stroke="#3A0F34" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    </span>
  );
}

/**
 * Application frame: persistent sidebar, top bar and the research-prototype notice.
 *
 * The notice is part of the frame, not of any one view, so it cannot be scrolled
 * past or navigated away from.
 */
export default function Shell({ view, apiOk, attempt, children }: Props) {
  const [open, setOpen] = useState(false);
  const groups = Array.from(new Set(VIEWS.map((v) => v.group)));
  const current = VIEWS.find((v) => v.id === view)!;

  const apiText = apiOk === null
    ? (attempt > 1 ? `waking · try ${attempt}` : "waking api…")
    : apiOk ? "api live" : "api down";

  return (
    <div className="app">
      <a href="#main" className="skip">Skip to content</a>

      <aside className={`side${open ? " open" : ""}`} aria-label="Primary">
        <a href="#/overview" className="brand side-brand" onClick={() => setOpen(false)}>
          <Mark />
          <span>FBimpact<span className="dim" style={{ fontWeight: 400 }}> · research</span></span>
        </a>

        <nav>
          {groups.map((g) => (
            <div key={g} className="side-group">
              <div className="side-label">{g}</div>
              {VIEWS.filter((v) => v.group === g).map((v) => (
                <a key={v.id} href={`#/${v.id}`} title={v.hint}
                   aria-current={v.id === view ? "page" : undefined}
                   className={`side-link${v.id === view ? " active" : ""}`}
                   onClick={() => setOpen(false)}>
                  <span>{v.label}</span>
                  <kbd>{v.key}</kbd>
                </a>
              ))}
            </div>
          ))}
        </nav>

        <div className="side-foot">
          <a href={PROJECT.repo} target="_blank" rel="noreferrer">Source ↗</a>
          <a href={`${PROJECT.api}/docs`} target="_blank" rel="noreferrer">API reference ↗</a>
        </div>
      </aside>

      {open && <div className="scrim" onClick={() => setOpen(false)} aria-hidden />}

      <div className="main-col">
        <header className="topbar">
          <button className="menu-btn" aria-label="Open navigation" aria-expanded={open}
                  onClick={() => setOpen((o) => !o)}>
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round">
              <path d="M3 5h12M3 9h12M3 13h12" />
            </svg>
          </button>
          <div className="crumb">
            <span className="dim">FBimpact</span>
            <span className="dim"> / </span>
            <b>{current.label}</b>
          </div>
          <span className="status" style={{ marginLeft: "auto" }}
                title="The API sleeps when idle and takes up to a minute to wake">
            <span className={`status-dot ${apiOk === null ? "wait" : apiOk ? "ok" : "bad"}`} />
            <a href="#/status" className="mono" style={{ fontSize: 11, color: "inherit", textDecoration: "none" }}>{apiText}</a>
          </span>
        </header>

        <div className="proto-bar" role="note">
          <b>Not a medical device.</b> Research prototype on public, acted-fall datasets. Skeletons
          only: no pixels reach the server.
        </div>

        <main id="main" className="view" tabIndex={-1}>
          {children}
          <footer className="footer">
            <p>
              <b style={{ color: "var(--text-2)" }}>{PROJECT.title} {PROJECT.subtitle}</b> ·{" "}
              {PROJECT.author} · {PROJECT.year}
            </p>
            <p>
              Lead time is measured from an annotated impact frame, and a warning counts as an
              anticipation only if it fires inside the imminent window. Public fall datasets use
              acted falls performed by young volunteers, so results describe a controlled study and
              not a clinical guarantee.
            </p>
          </footer>
        </main>
      </div>
    </div>
  );
}
