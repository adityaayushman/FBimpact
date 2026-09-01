/**
 * Research content, as structured data.
 *
 * Kept out of the components so the claims live in one auditable place. Every
 * figure here is either a definition from the methodology or a value measured
 * on this project - nothing is a placeholder, and nothing asserts a result that
 * has not been produced. Where a result does not exist yet (real-data
 * benchmarks), the field says so rather than showing a plausible number.
 */

export const PROJECT = {
  title: "Pre-Impact Fall Anticipation",
  subtitle: "with Grounded Skeletal Evidence",
  author: "Aditya Sahoo",
  year: "2026",
  repo: "https://github.com/adityaayushman/FBimpact",
  api: "https://fbimpact-api.onrender.com",
};

export const ABSTRACT =
  "This project predicts a fall before impact from ordinary video, and justifies each " +
  "warning by naming the body joints whose instability signalled it. It departs from the " +
  "mainstream fall literature on two axes at once: it performs pre-impact anticipation " +
  "rather than post-fall detection, buying the fraction of a second needed for a protective " +
  "response, and it operates on privacy-preserving skeleton data rather than raw RGB, " +
  "removing the main barrier to in-home cameras.";

export const HERO_STATS = [
  { value: "0.53 s", label: "mean lead time on real falls (UR Fall, 5-fold)" },
  { value: "0.78", label: "recall across all 30 real falls tested" },
  { value: "0 px", label: "pixels leaving the device: skeletons only" },
  { value: "393 k", label: "parameters — real-time on modest hardware" },
];

/**
 * What the experiments actually found.
 *
 * The headline is negative: the project's own novel component did not help.
 * That is stated first and plainly rather than buried under the ablations that
 * did work, because a results page that leads with its successes and files its
 * central failure under "limitations" is not reporting, it is marketing.
 */
export const FINDINGS = [
  {
    verdict: "negative" as const,
    rq: "RQ3",
    title: "The pre-impact objective did not buy lead time",
    body:
      "On real data the time-weighted loss made things worse on every axis. Setting λ = 0 — " +
      "the same architecture with plain class-weighted cross-entropy — reached recall 0.783 " +
      "and 0.533 s of lead time, against 0.642 and 0.459 s with the objective switched on. " +
      "The reproduced baseline beat both on recall at 0.808. The answer to RQ3 is that the " +
      "objective costs recall and lead time rather than trading one for the other.",
  },
  {
    verdict: "positive" as const,
    rq: "Ablation",
    title: "Temporal modelling and velocity are doing the work",
    body:
      "Removing temporal modelling collapses recall from 0.783 to 0.365 and lead time to " +
      "0.336 s; removing velocity features drops recall to 0.512 and pushes false alarms to " +
      "267 per hour. The architecture is earning its keep even though the novel objective is " +
      "not — anticipation is coming from motion over the window, exactly as designed.",
  },
  {
    verdict: "negative" as const,
    rq: "Deployment",
    title: "False-alarm rates are nowhere near deployable",
    body:
      "Every variant fires between 64 and 267 times per hour of normal activity. A care " +
      "setting would tolerate a small number per day. Nothing here is close, and the " +
      "operating-point curve shows no threshold that fixes it without destroying recall.",
  },
  {
    verdict: "caution" as const,
    rq: "RQ2",
    title: "Faithfulness gaps are positive but inside the noise",
    body:
      "Deletion gaps run from +0.006 to +0.095 across variants, with standard deviations of " +
      "the same magnitude. The joint rankings are not clearly beating a random ordering. " +
      "Notably the gradient-based attribution used by the no-grounding variant scores as well " +
      "as the attention head built for the purpose.",
  },
  {
    verdict: "caution" as const,
    rq: "Power",
    title: "Thirty falls is too few to be confident",
    body:
      "Per-fold recall ranges from 0.12 to 1.00 for the same variant, because each fold tests " +
      "about six falls. The fold-to-fold spread is larger than every difference between " +
      "variants, so these numbers rank methods weakly at best. UP-Fall, with 17 subjects, is " +
      "the benchmark that would settle it.",
  },
];

/* -- the gap ------------------------------------------------------------- */

export const POSITIONING = {
  columns: ["Post-fall detection", "Wearable IMU", "RGB video", "This project"],
  rows: [
    { dimension: "Warns before impact", values: [false, true, "sometimes", true] },
    { dimension: "Nothing to wear", values: [true, false, true, true] },
    { dimension: "Privacy-preserving", values: ["varies", true, false, true] },
    { dimension: "Explanation verified", values: ["rarely", "rarely", "rarely", true] },
    { dimension: "Real-time, cheap hardware", values: ["often", true, "often not", true] },
    { dimension: "Recall / lead-time focus", values: ["rarely", true, "rarely", true] },
  ],
};

export const RESEARCH_QUESTIONS = [
  {
    id: "RQ1",
    question: "Can a lightweight skeleton model warn before impact with useful lead time at high recall?",
    approach:
      "A compact spatio-temporal graph network emits a fall-imminence score at every frame. " +
      "Lead time is measured from an annotated impact frame and reported only over falls that " +
      "were correctly warned, alongside the false-alarm rate that buys it.",
  },
  {
    id: "RQ2",
    question: "Are joint-attention explanations actually faithful, or merely plausible-looking?",
    approach:
      "Deletion and insertion tests, after RISE. Joints are removed in relevance order and the " +
      "score is tracked; a faithful ranking collapses the score faster than a random ordering " +
      "of the same joints on the same model. The reported number is that gap, not the raw curve.",
  },
  {
    id: "RQ3",
    question: "Does an explicit pre-impact objective trade recall or false alarms for lead time, and by how much?",
    approach:
      "A time-weighted loss rewards earlier confident-correct warnings. Setting its weighting " +
      "term to zero recovers plain class-weighted cross-entropy exactly, so the ablation changes " +
      "one scalar and nothing else.",
  },
];

export const CONTRIBUTIONS = [
  {
    kind: "Primary",
    title: "Anticipation with verified evidence",
    body:
      "A lightweight model that predicts falls before impact earlier or at higher recall than a " +
      "reproduced baseline, with each warning grounded in the specific joints that preceded it and " +
      "quantified by a deletion/insertion faithfulness metric.",
  },
  {
    kind: "Secondary",
    title: "A fall-onset annotation protocol",
    body:
      "A documented protocol and evaluation subset for marking the impact frame, with " +
      "inter-annotator agreement reported — so lead time and faithfulness are measured rather " +
      "than asserted. UP-Fall labels activity intervals, not ground contact, so this is a " +
      "prerequisite rather than an optional extra.",
  },
];

/* -- pipeline ------------------------------------------------------------ */

export const PIPELINE = [
  { id: "A", name: "Pose estimation", novel: false, body: "Detect the subject and estimate a 2D skeleton. Weights frozen; pixels discarded immediately after.", io: "[H,W,3] → [17,3]" },
  { id: "B", name: "Normalise + features", novel: false, body: "Translate to the mid-hip origin, scale by torso length, interpolate low-confidence joints, append velocity.", io: "[T,17,3] → [C,T,V]" },
  { id: "C", name: "Window + graph", novel: false, body: "Slide a T-frame window at stride 1 and attach the skeleton adjacency used by the graph convolution.", io: "→ [N,C,T,V,M]" },
  { id: "D", name: "Temporal model", novel: true, body: "Causal spatio-temporal graph convolutions produce a fall-imminence logit for every frame, trained with the pre-impact time-weighted loss.", io: "→ p(t) ∈ [0,1]" },
  { id: "E", name: "Joint relevance", novel: true, body: "Rank joints at the warning frame by attention or gradient × input, and name the top contributors.", io: "→ ranked joints" },
  { id: "F", name: "Faithfulness test", novel: true, body: "Delete and insert joints in relevance order; compare the area under the score curve against a random ordering.", io: "→ faithfulness gap" },
  { id: "G", name: "Decision logic", novel: false, body: "Fire when the score holds at or above τ for k consecutive frames, with a refractory period so one episode is one alarm.", io: "→ warning flag" },
  { id: "H", name: "Lead time + evidence", novel: false, body: "Report (t* − t_warn)/fps for warnings inside the imminent window, with the joints that drove them.", io: "→ seconds + joints" },
];

/* -- definitions --------------------------------------------------------- */

export const METRICS = [
  { name: "Recall", primary: true, body: "Fraction of falls warned before impact, inside the imminent window. Primary metric: a missed fall is the costly error." },
  { name: "Lead time", primary: true, body: "(t* − t_warn) / fps, over correctly warned falls only. The quantity that decides whether a protective response is possible." },
  { name: "False alarms / hour", primary: true, body: "Triggers per hour of genuinely normal time. Stays comparable when the fall/ADL ratio of a test set changes." },
  { name: "Specificity", primary: false, body: "Fraction of normal-activity clips that produce no trigger at all." },
  { name: "Frame ROC-AUC", primary: false, body: "Threshold-free separability of imminent from normal frames, with tied scores averaged." },
  { name: "Faithfulness gap", primary: false, body: "Deletion and insertion area under curve, minus the same statistic under a random joint ordering. Positive means the explanation does work." },
];

export const DECISIONS = [
  {
    question: "How are frames after impact labelled?",
    answer: "Masked out",
    why: "The person is already down: those frames are neither imminent nor normal. Labelling them positive would quietly turn anticipation back into post-fall detection, which is the thing this project exists not to be.",
  },
  {
    question: "Which of the k persistence frames is the warning time?",
    answer: "The last",
    why: "The system cannot know a run of k frames has occurred until the k-th arrives. Timestamping the first would credit the model with (k−1)/fps seconds it never had — 67 ms at the defaults, a material share of the lead times being compared.",
  },
  {
    question: "What if a warning fires before the imminent window?",
    answer: "False alarm",
    why: "That frame is labelled normal by the project's own scheme. Counting it as an anticipation would reward contradicting the ground truth, and let a model that fires early on everything post enormous lead times.",
  },
  {
    question: "How are repeat triggers in one episode counted?",
    answer: "One alarm",
    why: "A sustained high-score episode is a single alarm to a caregiver. Counting per frame would inflate the false-alarm rate by roughly the frame rate and make the operating-point curve meaningless.",
  },
  {
    question: "Which score does a given frame use?",
    answer: "The window ending on it",
    why: "That is the only score a live system could hold at that instant. Averaging the overlapping windows — the natural offline choice — mixes in outputs computed from frames after it, leaking the future into the prediction.",
  },
];

export const ABLATIONS = [
  { variant: "Full model", change: "—", isolates: "Reference result" },
  { variant: "− temporal modelling", change: "Temporal kernels collapse to 1; the window is mean-pooled", isolates: "Value of the temporal module" },
  { variant: "− velocity features", change: "Positions only (C = 2)", isolates: "Value of explicit motion features" },
  { variant: "− pre-impact loss", change: "λ = 0, i.e. plain class-weighted BCE", isolates: "Contribution of the lead-time objective" },
  { variant: "− grounding head", change: "Uniform joint pooling, no attention", isolates: "Accuracy cost of the explanation" },
];

export const GUARDS = [
  {
    title: "Causality",
    body: "Temporal convolutions pad left only, the recurrence is unidirectional, and velocity is a backward difference. A test perturbs every frame after t and asserts the score at t does not move.",
    consequence: "Without it, the model can read the impact it claims to predict, and every lead time is an artefact.",
  },
  {
    title: "Subject independence",
    body: "Every split partitions subjects, never clips, and a disjointness assertion runs on each dataset construction.",
    consequence: "Leakage here measures identity memorisation, and shows up as suspiciously good results rather than an error.",
  },
  {
    title: "No test-set threshold fitting",
    body: "τ and k are selected on validation under a false-alarm budget and frozen into the checkpoint; evaluation applies them unchanged and reports the whole sweep as a curve.",
    consequence: "Otherwise the operating point is fitted to the test set and the headline number is not reproducible.",
  },
];

export const LIMITATIONS = [
  "Public fall datasets use acted falls performed by young volunteers, not recorded elderly falls. Results are a controlled study, not a clinical guarantee.",
  "Vision-based anticipation will likely give shorter lead times than a well-worn inertial sensor, which measures acceleration directly rather than inferring it.",
  "A single camera is vulnerable to occlusion and viewpoint; per-view results are reported rather than pooled.",
  "Pose estimators degrade exactly when a person is on the floor, which is when the signal matters most. Confidence filtering mitigates but does not remove this.",
  "The deployed demo checkpoint is trained on a synthetic fixture and carries no clinical meaning whatsoever.",
];

export const DEPLOYMENT_FACTS = [
  { label: "Backbone", value: "Causal ST-GCN" },
  { label: "Parameters", value: "393,046" },
  { label: "Trained on", value: "UR Fall, fold 0" },
  { label: "Receptive field", value: "25 frames ≤ 30-frame window" },
  { label: "Batch-1 latency", value: "16.3 ms GPU / 35.0 ms CPU" },
  { label: "Peak inference RSS", value: "449 MB (chunked)" },
  { label: "Test suite", value: "49 passing" },
];

/**
 * Provenance of the deployed checkpoint.
 *
 * Fold 0 is an arbitrary, pre-committed choice - not the best of the five.
 * Fold 3 reached recall 1.000 against fold 0's 0.875, and picking it would have
 * made the demo look better while making the demo a lie about the method.
 */
export const DEMO_CHECKPOINT = {
  variant: "− pre-impact loss (λ = 0)",
  why:
    "The best-performing configuration that still produces joint evidence. The pre-impact " +
    "objective measurably hurt on this data, so deploying it would mean shipping a worse model " +
    "to make the write-up sound better.",
  fold: "Fold 0 of 5 — arbitrary and fixed in advance, not the best. Fold 3 scored higher.",
  metrics: "recall 0.875 · lead 0.429 s · 75.7 false alarms per hour · frame AUC 0.987",
};
