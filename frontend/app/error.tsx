"use client";

export default function Error({ error, reset }: { error: Error; reset: () => void }) {
  return (
    <main style={{ maxWidth: 560, margin: "18vh auto", padding: "0 22px" }}>
      <h1 style={{ fontSize: 24 }}>Something went wrong</h1>
      <p className="card-sub">{error.message || "An unexpected error occurred while rendering this view."}</p>
      <button className="btn primary" onClick={reset}>Try again</button>{" "}
      <a className="btn" href="#/overview" onClick={reset}>Back to overview</a>
    </main>
  );
}
