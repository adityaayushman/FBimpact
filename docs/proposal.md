# Pre-Impact Fall Anticipation with Grounded Skeletal Evidence

## Full Project Documentation

Vision-based, privacy-preserving fall prevention for elderly care · Aditya Sahoo · 2026

*This document consolidates the research proposal, project plan, and technical pipeline into a single reference. It contains no schedule or timeline; ordering is described only as logical build dependencies.*

---

## 1. Abstract

This project predicts a **fall before impact** from ordinary video and justifies each warning by naming the **body joints whose instability signalled it**. It departs from the mainstream fall literature on two axes at once: it performs *pre-impact anticipation* rather than *post-fall detection* (buying the fraction of a second needed for a protective response or faster alert), and it operates on **privacy-preserving skeleton data** rather than raw RGB (removing the main barrier to in-home cameras). The design is deliberately compact — an off-the-shelf pose estimator is frozen and reused, and the entire research effort concentrates on a small temporal model, a joint-level explanation, and a test that proves the explanation is faithful. The result is completable by one student, runs in real time on modest hardware, and yields a clean, reproducible result suitable for a workshop paper.

## 2. Problem and Motivation

Most fall systems answer a backward-looking question — "did a fall occur?" — and raise an alarm afterwards. That is useful for summoning help but cannot prevent injury, and it is typically measured with balanced-accuracy figures that flatter the method. The higher-value question is forward-looking: how many milliseconds in advance can a fall be predicted, and can that prediction be justified by visible evidence? For a frail person, the operational currency is **lead time before impact**, because even a fraction of a second enables a protective action and a faster call for help.

The 2024–2025 anomaly and action-understanding literature has moved toward reasoning and explanation. This project adopts the useful half of that shift — interpretability — while discarding its unfalsifiable half. Free-text "causal" narratives and counterfactual percentages cannot be validated from passive video, which has no interventions and no counterfactual ground truth. This work keeps only claims that are identifiable and measurable: **temporal precedence** (which cues precede the fall) and **evidence grounding** (which joints the model relied on), both checkable against labels.

## 3. Aim, Objectives, and Research Questions

**Aim.** To anticipate falls *before impact* at higher recall and/or greater lead time than a reproduced baseline, with explanations grounded in verifiable skeletal evidence.

**Objectives.**

1. Reproduce an established skeleton-based fall model (ST-GCN or GCN-LSTM) on a public benchmark and confirm its behaviour.
2. Build a compact temporal model that raises lead time before impact while keeping recall high.
3. Attach a joint-level grounding head that names the joints driving each warning, and measure whether that evidence is faithful.
4. Quantify each component's contribution through a controlled ablation.

**Research questions.**

- **RQ1.** Can a lightweight skeleton model warn *before* impact with useful lead time at high recall?
- **RQ2.** Are joint-attention explanations actually *faithful*, or merely plausible-looking?
- **RQ3.** Does an explicit pre-impact objective trade recall / false-alarm rate for lead time, and by how much?

## 4. Positioning Against Related Work

The fall-monitoring field is crowded but concentrated in places this project deliberately avoids. The majority of published systems perform **post-fall detection** — recognising a fall after it happens. A second cluster achieves genuine **pre-impact** warning but uses **wearable inertial sensors**, which the elderly frequently forget, refuse, or remove. A third cluster uses **raw RGB video**, unacceptable in the bedrooms and bathrooms where most serious falls occur. Recent skeleton-based work (ST-GCN variants, GCN-LSTM, fusion models such as BioST-GCN) improves detection accuracy but rarely targets pre-impact lead time and rarely tests whether its explanations are faithful.

**The gap this project fills:** vision-based, pre-impact, privacy-preserving anticipation with an explanation that is *scored for faithfulness* against skeletal evidence — not free-text, not a causal claim. To be explicit: this work does **not** perform causal inference; it measures temporal precedence and evidence grounding, both identifiable from the data.

## 5. Contribution

The project makes one primary contribution and one modest secondary contribution — deliberately narrow, so each can be executed and defended.

- **Primary.** A lightweight anticipation model that predicts falls *before impact* earlier or at higher recall than a reproduced baseline, with each warning grounded in the specific joints that preceded it, quantified by a deletion/insertion faithfulness metric.
- **Secondary (optional).** A small, documented fall-onset annotation protocol and evaluation subset, enabling faithfulness and lead time to be measured rather than asserted.

The novelty budget is spent entirely on the anticipation objective and the verified explanation; perception is treated as solved plumbing.

## 6. Scope

**In scope:** single elderly-care domain; frozen off-the-shelf pose estimation; one novel temporal + joint-grounding module; one reproduced baseline; pre-impact metrics; one ablation; a small onset-annotation subset.

**Out of scope (deliberately):** causal/counterfactual claims; multi-domain evaluation; reimplementing the pose estimator; wearable-sensor fusion for the core claim (vision-only); raw-RGB pipelines (privacy); any claim not measurable against ground truth.

Stating the exclusions is itself a rigour contribution: it prevents scope creep and matches the claims to the resources.

## 7. System Architecture

Three stages, only the middle of which is a research contribution.

**Stage A — Frozen perception.** Each frame passes through a pretrained 2D pose estimator, yielding a skeleton (joint coordinates + confidence) per person per frame. Weights are frozen; raw pixels are discarded after pose extraction, giving privacy by construction.

**Stage B — Temporal anticipation model (the contribution).** A compact spatio-temporal graph model (ST-GCN or GCN-LSTM) consumes the skeleton sequence across a sliding window and outputs, at every frame, a fall-imminence score, trained with a pre-impact, time-weighted loss that rewards earlier confident-correct warnings. Graph attention over joints is exposed as the explanation.

**Stage C — Grounded explanation + faithfulness.** For each warning, the top-contributing joints are reported; a deletion/insertion test then verifies that removing those joints collapses the imminence score faster than removing random joints, turning "the model looked here" into a measurable property.

## 8. Full Technical Pipeline

End-to-end flow: `RGB video → [A] pose estimation (frozen) → privacy boundary → [B] normalise + features → [C] window + joint graph → [D] ST-GCN temporal model → per-frame imminence score → [G] decision logic → [H] warning + lead time`, with a branch `[D] → [E] joint attention → [F] faithfulness test → grounded explanation`.

| # | Stage | Input | Process | Output | Tool |
|---|---|---|---|---|---|
| A | Pose estimation (frozen) | RGB frame `[H,W,3]` | Detect person; estimate 2D skeleton; keep the largest/most-central subject | `[17,3]` = (x, y, confidence) per frame | RTMPose / YOLO-Pose / AlphaPose |
| B | Normalise + features | Skeleton seq `[T,17,3]` | Translate to mid-hip origin; scale by torso length; interpolate low-confidence joints; add velocity | `[C,T,V]`, C=4 → (x, y, vx, vy) | NumPy |
| C | Window + graph | Feature stream | Slide a T-frame window (stride 1 online); attach skeleton adjacency `A [V,V]` | `[N,C,T,V,M]`, M=1 | Custom loader |
| D | Temporal model ★ | `[N,4,30,17,1]` | Spatio-temporal graph convolutions → per-frame logit → sigmoid | `p_t ∈ [0,1]` per frame | ST-GCN / GCN-LSTM (PyTorch) |
| E | Joint attention ★ | Model activations at warning frame | Per-joint relevance via attention or gradient×activation | ranked joints, top-k | Custom |
| F | Faithfulness test ★ | Ranked joints + model | Delete/insert top joints; area under score-change curve vs. random | faithfulness AUC | Custom |
| G | Decision logic | `p_t` stream | Fire when `p_t ≥ τ` for `k` consecutive frames | warning flag + trigger frame | Custom |
| H | Output | Trigger + explanation | Compute lead time `(t*−t_warn)/fps`; emit alert + top-k joints | warning, lead-time, evidence | — |

Stages A–C are reusable plumbing; D and E–F are the contribution.

## 9. Pre-Impact Labelling and Lead-Time Definition

For each fall clip, mark the **impact frame `t*`** (first ground contact). The **imminent window** is `[t* − W_pre, t*]` with `W_pre ≈ 20 frames` (~0.5–1.0 s at 30 fps): frames inside it are positive, frames before it are negative, and every frame of a normal-activity (ADL) clip is negative. At inference, the warning fires at the first frame `t_warn` where the score crosses threshold with persistence. **Lead time = `(t* − t_warn)/fps`**, reported only over correctly warned falls. A fall with no warning before `t*` is a miss; any warning during a normal clip is a false alarm. This is the project's headline metric.

## 10. Decision Logic

Because a per-frame score is noisy, the trigger uses **threshold + persistence**: warn when `p_t ≥ τ` sustained for `k` consecutive frames (defaults `τ = 0.70`, `k = 3`). Increasing `k` cuts false alarms but shortens lead time; sweeping it produces an operating-point curve (lead time vs. false-alarm rate), which is reported instead of a single cherry-picked threshold.

## 11. Explanation and Faithfulness

At the warning frame, rank joints by relevance and report the top-k (e.g. *trunk lean, hip drop, knee buckling*). Then prove the explanation: **deletion** progressively zeroes the highest-relevance joints and records how fast `p_t` falls (steep = faithful); **insertion** starts from a blank skeleton and adds joints in relevance order (fast rise = faithful); both are compared against deleting/inserting *random* joints. The gap is the faithfulness result reported in the paper. This is the element that most fall systems omit and that makes the explanation defensible.

## 12. Technology Stack and Resources

| Layer | Choice | Notes |
|---|---|---|
| Perception (pose) | Off-the-shelf 2D pose estimator (RTMPose / YOLO-Pose / AlphaPose) | Frozen; privacy-preserving; not a contribution |
| Representation | Normalised joint coordinates + velocities | Tiny feature footprint |
| Temporal model | Compact ST-GCN or GCN-LSTM over skeleton window | The novel module; runs on modest hardware |
| Explanation | Joint/edge attention + deletion/insertion metric | Faithfulness is the headline of RQ2 |
| Frameworks | PyTorch (+ a skeleton-action library), NumPy | Reproducible environment file |
| Compute | One modest GPU; skeleton streams are lightweight | Real-time inference is realistic |
| Reproducibility | Git, config files, fixed seeds, experiment logging | Required for a top score |

## 13. Datasets

| Dataset | Composition | Role |
|---|---|---|
| UP-Fall | 17 subjects, 11 activities, RGB + IR + inertial; supports a pre-impact framing | **Primary benchmark** |
| UR Fall Detection (URFD) | RGB-D + accelerometer, fall vs. normal | Secondary / transfer check |
| Le2i (ImViA) | 143 fall + 48 normal videos, varied scenes/lighting | Secondary / transfer check |

Recommendation: use UP-Fall as the primary benchmark; report a second dataset only as a transfer check. Cross-dataset transfer is a stretch goal, not a core claim.

## 14. Training Methodology

- **Split subject-independently** (leave-subjects-out). The same person must never appear in both train and test, otherwise the model measures identity memorisation rather than fall anticipation. This single choice protects the validity of the whole result.
- **Augmentation:** horizontal flip (swap left/right joints), temporal jitter/crop, small Gaussian joint noise, random scale.
- **Loss:** binary cross-entropy with a pre-impact exponential time-weighting that rewards earlier confident-correct warnings, plus class weighting (or focal loss) for the fall/normal imbalance.
- **Optimisation:** Adam, LR 1e-3 with cosine decay, weight decay 1e-4, batch 32, early stopping on validation recall/lead-time.
- **Repeatability:** seeds {0, 1, 2}; report mean ± standard deviation, never a single run.

## 15. Evaluation Methodology and Success Criteria

**Metrics:** recall/sensitivity (**primary** — missed falls are the costly error), specificity, false alarms per hour, mean lead time before impact, ROC-AUC, F1, and faithfulness AUC.

**Baseline:** reproduce a standard skeleton fall model (ST-GCN) and evaluate it through the same windowing and decision logic, so the comparison is fair.

**Tiered success (the project succeeds even if the headline is missed):**

- **Minimum:** baseline reproduced; full pipeline runs end-to-end; joint explanations and a faithfulness number produced.
- **Target:** ours matches baseline recall while increasing lead time before impact, with a completed ablation.
- **Stretch:** measurable faithfulness advantage and/or a cross-dataset transfer result.

## 16. Ablation Plan

Each variant removes exactly one component; anticipation and faithfulness metrics are re-measured.

| Variant | Change from full model | Isolates |
|---|---|---|
| Full model | — | Reference result |
| – temporal modelling | Mean-pool the window | Value of the temporal module |
| – velocity features | Positions only | Value of motion features |
| – pre-impact loss | Plain classification loss | Contribution of the lead-time objective |
| – grounding head | No evidence output | Cost of explanation on accuracy |

## 17. Why This Project Is Stronger and More Efficient Than Others

This project occupies the intersection the field leaves empty: **vision-based, pre-impact, privacy-preserving, and explanation-verified.** The advantages are on the axes that decide real-world adoption and evaluation, not on raw benchmark accuracy:

- **Anticipation, not post-hoc detection.** Optimising *lead time before impact* targets the quantity that enables a protective response, instead of merely confirming a fall after it happened.
- **Privacy by construction.** Raw pixels are discarded after pose extraction; only skeletons are stored and modelled — removing the biggest barrier to in-home camera acceptance.
- **Compute and inference efficiency.** Skeleton streams are far smaller than video tensors, so the model trains on modest hardware and runs in real time on a cheap edge device — feasible for actual deployment, unlike heavy RGB or transformer-video pipelines.
- **No forgotten wearable.** A passive camera needs nothing worn, charged, or remembered — the practical failure mode of the wearable pre-impact literature.
- **Verified explanations, not decorative ones.** Reporting *which joints* signalled the fall **and** proving it with a deletion/insertion score gives a caregiver-interpretable, trustworthy justification — an under-occupied niche.
- **Recall-first, deployment-honest metrics.** Foregrounding recall, lead time, and false-alarm rate reflects what a care setting actually cares about, rather than a single balanced-accuracy figure.
- **Completability and reproducibility.** Scope matched to one student; public code, fixed seeds, multi-run variance make the result trustworthy in review.

**Honest boundary of the claim.** Vision-based anticipation will likely give shorter lead times than a well-worn IMU (which senses acceleration directly); it is vulnerable to occlusion and single-camera viewpoints; and public fall datasets are acted by young volunteers, not recorded from real elderly falls. The paper frames results as a controlled study, not a clinical guarantee. The edge is **pre-impact timing, privacy, efficiency, and verifiable explanation** — axes on which a lightweight, well-scoped system can legitimately win.

| Dimension | Post-fall detection | Wearable pre-impact (IMU) | RGB video systems | **This project** |
|---|---|---|---|---|
| Warns *before* impact | No | Yes | Sometimes | **Yes (primary metric)** |
| Nothing to wear | Yes | No | Yes | **Yes** |
| Privacy-preserving | Varies | Yes | No | **Yes (skeleton only)** |
| Explanation *verified* | Rarely | Rarely | Rarely | **Yes (faithfulness score)** |
| Real-time on cheap hardware | Often | Yes | Often not | **Yes (by design)** |
| Recall / lead-time focus | Rarely | Yes | Rarely | **Yes** |

## 18. Risks, Limitations, and Mitigations

| Risk / limitation | Mitigation |
|---|---|
| Occlusion / single-camera viewpoint | Report per-view results; frame as a controlled study; multi-view as a stretch goal |
| Datasets use *acted* falls, not real ones | State plainly as a limitation; avoid clinical claims; cross-dataset check |
| Class imbalance (few falls) | Recall-focused metrics; loss weighting; report false-alarm rate |
| Pose-estimator errors on the floor / odd poses | Confidence filtering; robustness noted; error analysis in the paper |
| Baseline under-performs its paper | Compare against the reproduced number, transparently reported |
| Joint attention proves unfaithful | A reportable negative result documented by the faithfulness metric |
| Fall-onset timing is subjective | Publish the annotation protocol; report inter-annotator agreement |

## 19. Ethical and Practical Considerations

The project uses public research datasets under their licences; no new footage of identifiable people is collected. Privacy is protected by design — only skeletons are stored, not video. Because the system concerns the safety of vulnerable people, the documentation and paper state clearly that it is a research prototype and **not a medical device**, report false-alarm behaviour honestly, and avoid any deployment or clinical claim not supported by measured evidence. The limitation that public datasets use acted rather than real elderly falls is stated prominently.

## 20. Deliverables

A reproducible code repository (training + evaluation, configs, seeds); the populated results and ablation tables with variance across runs; the fall-onset annotation protocol and subset with inter-annotator agreement; architecture and results figures; and a workshop-length paper positioning the work against the 2024–2025 fall-prediction literature.

## 21. Suggested Repository Layout

```
fall-anticipation/
├── data/         # dataset loaders, pre-impact windowing, splits
├── pose/         # frozen pose wrapper + skeleton caching
├── models/       # ST-GCN / GCN-LSTM + per-frame head
├── losses/       # pre-impact time-weighted loss
├── explain/      # joint relevance + deletion/insertion faithfulness
├── configs/      # YAML hyperparameters
├── train.py  eval.py  infer.py
└── results/      # tables, figures, logs, checkpoints
```

**Logical build order (dependencies, not a schedule):** get A→B→C producing cached skeleton windows first; reproduce the ST-GCN baseline through G–H; then add the pre-impact loss in D; then the explanation E–F; then run the ablations. Each stage is independently testable, so a working system always exists to fall back on.

## 22. Selected References for Positioning

1. Yan et al., *Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition* (ST-GCN), AAAI 2018.
2. UP-Fall Detection Dataset — Martínez-Villaseñor et al., *Sensors*, 2019.
3. Kwolek & Kepski, *Human fall detection on embedded platform using depth maps and wireless accelerometer* (UR Fall Detection Dataset).
4. Charfi et al., *Optimized spatio-temporal descriptors for real-time fall detection* (Le2i / ImViA dataset).
5. *PreFallKD: Pre-Impact Fall Detection via CNN-ViT Knowledge Distillation*, arXiv:2303.03634.
6. *KFall: A Large-Scale Open Motion Dataset and Benchmark Algorithms for Detecting Pre-impact Fall of the Elderly Using Wearable Inertial Sensors.*
7. *Fusing Biomechanical and Spatio-Temporal Features for Fall Prediction* (BioST-GCN), arXiv:2511.14620.
8. *Fall recognition using a three-stream spatio-temporal GCN model with adaptive feature aggregation*, Scientific Reports, 2025.
9. Petsiuk et al., *RISE: Randomized Input Sampling for Explanation* (deletion/insertion faithfulness metric), BMVC 2018.

---

## Appendix A — Implementation notes

*Added during implementation; the sections above are the proposal as written.*

The specification above leaves several choices open that turn out to change the headline
number. Each is implemented as a named, switchable option with a documented default rather
than a silent convention — see the "Decisions the proposal left open" table in
[`../README.md`](../README.md) for the full list and rationale. In brief:

1. **Frames after `t*`** are masked out of the loss and the metrics (§9 defines the imminent
   window but not the aftermath). Labelling them positive would turn anticipation back into
   post-fall detection.
2. **`t_warn` is the last of the `k` persistence frames**, not the first (§10). The difference
   is `(k−1)/fps` — 67 ms at the stated defaults — added to every reported lead time.
3. **A warning before the imminent window counts as a false alarm** (§9 says "any warning
   during a normal clip", but is silent on a normal *frame* of a fall clip). Otherwise a model
   that fires early on everything posts arbitrarily large lead times.
4. **Causality is enforced structurally**, not assumed: left-only temporal padding,
   unidirectional recurrence, backward-difference velocity, and per-frame scores taken from
   the window *ending* at that frame. §8 does not require this, but without it the model can
   read the impact it claims to predict and every lead time is an artefact.
5. **UP-Fall runs at ~18 fps, not 30.** All frame-count parameters in `configs/upfall.yaml`
   are re-derived rather than inherited, since lead time is reported in seconds.
6. **UP-Fall ships no impact frames** — its labels are activity intervals. `t*` must be
   annotated before any lead time can be measured, which promotes the §5 "optional" secondary
   contribution to a prerequisite for the primary one.

Two measured findings that bear on claims in §12 and §17 — inference latency at batch 1, and
the drift between online and offline features — are recorded in the README rather than here,
since they are properties of a particular machine rather than of the design.
