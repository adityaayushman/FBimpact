export default function NotFound() {
  return (
    <main style={{ maxWidth: 560, margin: "18vh auto", padding: "0 22px" }}>
      <h1 style={{ fontSize: 24 }}>Page not found</h1>
      <p className="card-sub">There is no page at this address. The dashboard lives at the root.</p>
      <a className="btn primary" href="/">Go to the dashboard</a>
    </main>
  );
}
