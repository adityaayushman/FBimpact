# Deployment

Two services: a **Next.js frontend on Vercel** and a **FastAPI inference API on Render**.

```
Browser (Vercel)                                    Render
┌──────────────────────────────┐                  ┌────────────────────────┐
│ video / webcam                │                  │ FastAPI                │
│   ↓ MoveNet (TF.js, in-page)  │  skeletons only  │   ↓ ST-GCN (torch CPU) │
│ [T,17,3] keypoints            │ ───────────────► │ per-frame p_t          │
│   ↑ scores, evidence          │ ◄─────────────── │ joint relevance        │
│ skeleton + chart              │                  │ deletion/insertion     │
└──────────────────────────────┘                  └────────────────────────┘
        pixels never leave the device
```

**Pose estimation runs in the browser**, so video never reaches the server and the API has
no endpoint that accepts an image. §19's privacy claim is a property of the architecture,
not a promise about server-side handling. It also keeps the free-tier container small —
running RTMPose server-side would need OpenCV, ONNX Runtime and a few hundred MB of models.

---

## 1. Push to GitHub

```bash
git remote -v          # should show https://github.com/adityaayushman/FBimpact.git
gh auth login          # or configure a credential helper
git push -u origin main
```

## 2. Backend → Render

The blueprint is [`render.yaml`](render.yaml). On [render.com](https://render.com):
**New → Blueprint** → connect the repo → apply.

Or configure a Web Service by hand:

| Setting | Value |
|---|---|
| Root directory | *(repo root — **not** `backend/`)* |
| Runtime | Python 3.12 |
| Build command | `pip install -r backend/requirements.txt` |
| Start command | `uvicorn backend.app:app --host 0.0.0.0 --port $PORT --workers 1` |
| Health check path | `/health` |

The root is the repo root because the API imports the project's own packages (`data/`,
`models/`, `evaluation/`, `explain/`) rather than reimplementing them — running the demo
through the same code as `eval.py` is what keeps the two from drifting apart.

`backend/requirements.txt` pins the **CPU** torch wheel. The default `torch` pulls ~2.5 GB
of CUDA libraries that a free-tier container has neither the disk nor the GPU for.

Once the frontend is live, set `ALLOWED_ORIGINS` to its domain. Until then the app falls
back to allowing `*.vercel.app`, which is handy for preview deployments and too permissive
for anything else.

**Free-tier realities.** The container spins down after ~15 minutes idle, so the first
request afterwards takes 30–60 s; the client shows a cold-start message rather than
"failed to fetch". `/analyze` on a 140-frame clip takes ~2.7 s on a local CPU and will be
slower on a shared one. `/faithfulness` is ~150 forward passes and takes several seconds —
that is why it is a separate, opt-in endpoint rather than part of `/analyze`.

## 3. Frontend → Vercel

On [vercel.com](https://vercel.com): **Add New → Project** → import the repo.

| Setting | Value |
|---|---|
| Framework | Next.js (auto-detected) |
| **Root directory** | **`frontend`** ← must be set; the repo root is not a Next app |
| Environment variable | `NEXT_PUBLIC_API_URL` = your Render URL, e.g. `https://fbimpact-api.onrender.com` |

`NEXT_PUBLIC_API_URL` is inlined at build time, so **changing it requires a redeploy**, not
just a restart.

## 4. Refreshing the deployed model

The bundled checkpoint lives at `backend/model/best.pt` and the demo clips at
`backend/demo_clips/`. Both are committed, so a deploy needs no external storage.

```bash
python train.py --config configs/ours_preimpact.yaml
python scripts/prepare_demo.py --checkpoint results/<run>/best.pt
git add backend/model backend/demo_clips && git commit -m "Update demo checkpoint"
```

`prepare_demo.py` picks three falls and four normal activities, weighted towards **hard
negatives** — sitting down, bending to pick something up and lying down are all controlled
descents. A demo of walking-versus-falling would hide the failure mode that matters.

---

## Running locally

```bash
# API
pip install -r backend/requirements.txt
python scripts/make_synthetic.py
python train.py --config configs/ours_preimpact.yaml
python scripts/prepare_demo.py --checkpoint results/<run>/best.pt
TORCH_NUM_THREADS=4 uvicorn backend.app:app --reload --port 8000   # → /docs for OpenAPI

# UI
cd frontend && npm install
cp .env.example .env.local
npm run dev                                                        # → localhost:3000
```

`TORCH_NUM_THREADS` defaults to 1, which is right for Render's fraction of a shared core
and wasteful on a real machine.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness plus the loaded model's identity and frozen operating point |
| `GET /skeleton` | Joint names and bones, so the client draws the graph the model uses |
| `GET /clips`, `GET /clips/{id}` | Bundled demo clips |
| `POST /analyze` | Score a `[T,17,3]` sequence → per-frame `p_t`, warnings, joint evidence |
| `POST /faithfulness` | Deletion/insertion test on one warning (RQ2) |

`/analyze` accepts `impact_frame`. **Lead time is only defined when it is given** — without
a known `t*` the API returns warnings with `lead_time: null` rather than inventing one. A
trigger that fires before the imminent window comes back with
`within_imminent_window: false`, and the UI labels it a false alarm rather than an
anticipation, matching `evaluation/metrics.py`.

---

**Not a medical device.** The deployed checkpoint is trained on a synthetic fixture, not on
real fall data. The UI says so above the fold, and the API repeats it in `/health`.
