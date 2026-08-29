# Pre-Impact Fall Anticipation with Grounded Skeletal Evidence

Vision-based, privacy-preserving fall **anticipation** for elderly care: predict a fall
*before impact* from ordinary video, and name the body joints whose instability signalled
it — with a test that checks whether that evidence is actually faithful.

Full proposal: [`docs/proposal.md`](docs/proposal.md). Section numbers throughout the code
refer to it.

```
RGB video → [A] pose (frozen) ─┤privacy boundary├→ [B] normalise+features → [C] window+graph
   → [D] ST-GCN → per-frame p_t → [G] threshold+persistence → [H] warning + lead time
                      └→ [E] joint relevance → [F] deletion/insertion → grounded explanation
```

Stages A–C are plumbing. **D** and **E–F** are the contribution.

---

## Quick start

```bash
pip install -r requirements.txt

python scripts/make_synthetic.py              # a runnable fixture; no download needed
python train.py --config configs/ours_preimpact.yaml
python eval.py  --checkpoint results/ours_preimpact_seed0_*/best.pt --explain --curve
python infer.py --checkpoint results/ours_preimpact_seed0_*/best.pt \
                --clip data/cache/synthetic/S02_fall00.npz
python -m pytest tests -q
```

The synthetic generator is a **smoke-test fixture, not a benchmark** — it exists so every
stage can be run and tested before UP-Fall is downloaded and annotated. Nothing measured
on it belongs in the paper. Its ADL set is weighted towards hard negatives (sitting down,
bending to pick up, lying down) so a model that fires on any downward motion fails early
rather than at submission.

## Real data (UP-Fall, the primary benchmark)

```bash
# 1. Skeletons. The only step that touches pixels.
python scripts/cache_poses.py --dataset upfall --root <UP-Fall videos> --out data/cache/upfall

# 2. Impact frames. UP-Fall labels activity intervals, not ground contact,
#    so t* must be annotated — see the protocol in the script's docstring.
python scripts/annotate_onsets.py --cache data/cache/upfall --out data/annotations/upfall_onsets.csv
python scripts/annotate_onsets.py --agreement annotator_a.csv annotator_b.csv --fps 18

# 3. Train / evaluate.
python train.py --config configs/upfall.yaml
python scripts/run_ablations.py --seeds 0 1 2
```

Falls with no annotated `t*` are **dropped, not guessed**: a heuristic `t*` derived from the
same skeleton the model consumes would make lead time partly circular. A partially annotated
cache degrades to a smaller experiment rather than a wrong one.

## Layout

| Path | Stage | What lives here |
|---|---|---|
| [`pose/`](pose/) | A | Frozen estimator wrappers, subject selection, video → skeleton cache |
| [`data/`](data/) | B–C | Normalisation, pre-impact labelling, windowing, splits, augmentation, streaming |
| [`models/`](models/) | D ★ | Graph adjacency, causal ST-GCN / GCN-LSTM, per-frame head, joint attention |
| [`losses/`](losses/) | D ★ | Pre-impact time-weighted BCE |
| [`explain/`](explain/) | E–F ★ | Joint relevance, deletion/insertion faithfulness |
| [`evaluation/`](evaluation/) | G–H | Trigger rule, lead time, recall / FA-per-hour, operating curves |
| [`configs/`](configs/) | — | YAML; `_base_` inheritance, `--set key=value` overrides |
| [`scripts/`](scripts/) | — | Synthetic fixture, pose caching, onset annotation, ablation grid |
| [`backend/`](backend/) | — | FastAPI inference API (Render) |
| [`frontend/`](frontend/) | — | Next.js demo UI (Vercel) |
| `train.py` `eval.py` `infer.py` | — | Entry points |

`evaluation/` and `utils/` are the two directories added to the proposal's §21 layout — the
decision logic and metrics needed somewhere to live that was neither data nor model.

## Live demo

Frontend on Vercel, API on Render — see [`DEPLOY.md`](DEPLOY.md).

**Pose estimation runs in the browser** (TF.js MoveNet, which emits COCO-17 — exactly the
layout in [`data/skeleton.py`](data/skeleton.py)), and only skeletons are sent to the API.
There is no endpoint that accepts an image, so §19's privacy claim is a property of the
architecture rather than a promise about server-side handling.

The API routes through the same `ClipDataset` windowing, the same frame alignment and the
same trigger rule as `eval.py`. If the demo and the reported metrics disagreed, one would be
wrong and there would be no way to tell which.

```bash
uvicorn backend.app:app --port 8000     # /docs for the OpenAPI page
cd frontend && npm install && npm run dev
```

---

## Decisions the proposal left open

Each of these changes the headline number, so each is a named, switchable option with a
documented default rather than a silent convention.

| Question | Default here | Why | Where |
|---|---|---|---|
| Labels **after** `t*`? | `ignore` (masked) | The person is already down. Labelling them positive turns anticipation back into post-fall detection. | [`data/labels.py`](data/labels.py) |
| Which of the `k` persistence frames is `t_warn`? | the **last** | The system cannot know a run of `k` occurred until the `k`-th arrives. Timestamping the first would credit `(k−1)/fps` — 67 ms at the defaults — that the model never had. | [`evaluation/decision.py`](evaluation/decision.py) |
| A warning fired **before** the imminent window? | `false_alarm` | That frame is labelled *normal*. Scoring it as anticipation rewards contradicting the ground truth and lets a trigger-happy model post huge lead times. `hit` / `ignore` available as sensitivity checks. | [`evaluation/metrics.py`](evaluation/metrics.py) |
| Repeat triggers in one episode? | refractory period (~1 s) | One sustained episode is one alarm to a caregiver. Counting per frame inflates FA/hour by roughly the frame rate. | [`evaluation/decision.py`](evaluation/decision.py) |
| False alarms per hour of *what*? | per hour of **negative** time | Stays comparable when the fall/ADL ratio of the test set changes. | [`evaluation/metrics.py`](evaluation/metrics.py) |
| Which score does frame `f` use? | the window **ending** at `f` | The only one a live system could have. Averaging overlapping windows — the natural offline choice — leaks the future. | [`evaluation/runner.py`](evaluation/runner.py) |

## Guards that protect the result

Three properties, if silently broken, make every number in the paper meaningless while
producing no error — so each is enforced in code and asserted in tests.

- **Causality.** Temporal convolutions pad left only; the LSTM is unidirectional; velocity is
  a backward difference. `test_causal_model_ignores_the_future` perturbs frames `t+1…T` and
  asserts the score at `t` does not move. A centre-padded ST-GCN would let the model see the
  impact it claims to predict.
- **Subject independence.** Every split partitions *subjects*, and `assert_subject_disjoint`
  runs on every dataset construction. Leakage here measures identity memorisation and shows
  up as suspiciously good results, not a crash.
- **No test-set threshold fitting.** `train.py` selects `(τ, k)` on validation under a
  false-alarm budget and freezes it into the checkpoint; `eval.py` applies it unchanged.
  `--curve` reports the whole sweep so no single point has to be defended.

## Measured on this machine (RTX 4050 laptop, 6 GB)

| | |
|---|---|
| Model | 393 k parameters, receptive field 25 frames ≤ 30-frame window |
| Batch-1 latency | **16.3 ms/frame (61 fps) CUDA**, 35.0 ms/frame (28.6 fps) CPU |
| Epoch (synthetic fixture, 40 clips) | ~3.5 s |

Two caveats behind the "real-time on cheap hardware" claim in §17. At batch 1 the model is
**kernel-launch bound, not compute bound** — the GPU is 44× faster at batch 110 than 1× the
batch-1 rate would suggest — so single-stream throughput is a launch-overhead figure, and CPU
inference sits just below 30 fps here. Measure with `infer.py --device cpu` before claiming a
deployment number, and note that these figures exclude pose estimation, which will dominate.

**`run.deterministic` defaults to `false`.** cuDNN deterministic kernels cost **45×** on this
GPU (2145 ms vs 48 ms per batch-110 forward) — days instead of hours for a six-variant,
three-seed grid. The seed already fixes initialisation, data order, augmentation and splits;
determinism only pins floating-point reduction *order*, and the residual noise is what the
seeds `{0,1,2}` and the reported standard deviation exist to absorb. Turn it on for a final
bitwise-reproducible run and budget for it.

## Known gap: online vs. offline features

`data/normalize.py` (offline) interpolates a dropped joint from *both* sides of the gap and
scales by the clip's median torso length. `data/stream.py` (live) can do neither — it holds
the last value and tracks a running median. Measured over the synthetic fixture:

| | mean | max |
|---|---|---|
| position drift | 0.001 | 0.002 |
| velocity drift | 0.032 | 0.085 |
| any channel, worst frame | 0.017 | **11.98** |

Aggregate drift is small, but individual frames around a dropout spike hard, and that is
enough to change which joints an explanation names — `infer.py` and `eval.py` reported
different top-3 joints for the same warning on the same clip. Offline metrics are therefore a
mild **upper bound** on live behaviour. `data.stream.compare_offline` measures this per clip;
report it rather than assuming it away.

## Reproducing a run

Every run writes `config.yaml` (fully resolved), `environment.json` (versions, git commit,
dirty flag), `split.json`, `history.jsonl` and `best.pt` into its results directory.
Reproducing means re-running the saved config — there is no hidden state in a shell history.

---

**Not a medical device.** A research prototype. Public fall datasets use *acted* falls by
young volunteers, not recorded elderly falls; results are a controlled study, not a clinical
guarantee. See §18–19 of the proposal.
