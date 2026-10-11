/**
 * Typed client for the inference API.
 *
 * The API accepts skeletons and nothing else - there is no endpoint that takes a
 * frame. Pose estimation runs in the browser (see lib/pose.ts), so video never
 * leaves the device.
 */

/**
 * The deployed inference API.
 *
 * This is the default rather than a required environment variable. There is one
 * public backend and its URL is fixed, so making every deployment configure it
 * bought nothing and cost a broken page whenever it was missed - Next inlines
 * `NEXT_PUBLIC_*` at build time, so a deploy made without it stays broken until
 * someone rebuilds.
 *
 * Set `NEXT_PUBLIC_API_URL` to override it, which is what local development
 * against your own uvicorn does.
 */
const DEFAULT_API_URL = "https://fbimpact-api.onrender.com";

const RAW_API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "").trim();

export const API_URL = (RAW_API_URL || DEFAULT_API_URL).replace(/\/$/, "");

/** True when running against the built-in public backend rather than an override. */
export const API_IS_DEFAULT = RAW_API_URL.length === 0;

/** [T][17][3] of (x, y, confidence) in pixels. */
export type Keypoints = number[][][];

export interface ModelInfo {
  backbone: string;
  parameters: number;
  in_channels: number;
  window: number;
  w_pre_frames: number;
  with_velocity: boolean;
  attention: boolean;
  causal: boolean;
  explain_method: string;
  trained_on: string;
  epoch: number | null;
  device: string;
  /** Absent on schema v1 backends; treat missing as false. */
  has_phase_head?: boolean;
  feature_set?: string;
}

export interface Decision {
  threshold: number;
  persistence: number;
  refractory_frames: number;
}

export interface Health {
  status: string;
  model: ModelInfo;
  decision: Decision;
  limits: { max_frames: number };
  disclaimer: string;
}

export interface SkeletonSpec {
  joints: string[];
  bones: number[][];
  flip_pairs: number[][];
  num_joints: number;
}

export interface ClipSummary {
  clip_id: string;
  label: "fall" | "adl";
  activity: string;
  num_frames: number;
  fps: number;
  duration_s: number;
  impact_frame: number | null;
  source: string;
}

export interface ClipPayload extends Omit<ClipSummary, "duration_s"> {
  keypoints: Keypoints;
}

export interface Evidence {
  method: string;
  phrase: string;
  top_joints: { joint: string; index: number; relevance: number }[];
  relevance: number[];
}

export interface Warning {
  frame: number;
  time_s: number;
  score: number;
  /** Null when t* is unknown, or when the warning fired after impact. */
  lead_time: number | null;
  /** Null when t* is unknown. False means it fired before the imminent window. */
  within_imminent_window: boolean | null;
  evidence?: Evidence;
}

/** Per-frame phase probabilities straight from the model's phase head. */
export interface Phases {
  names: string[];
  /** [T][P], each row summing to 1. */
  probs: number[][];
}

export interface Analysis {
  /** Absent on v1 backends. A version this client does not know => render the score only. */
  schema_version?: number;
  clip_id: string;
  frames: number;
  fps: number;
  impact_frame: number | null;
  scores: number[];
  /** Null when the checkpoint has no phase head, or on a v1 backend. Never synthesised. */
  phases?: Phases | null;
  warnings: Warning[];
  decision: Decision;
  imminent_window: [number, number] | null;
  latency_ms: number;
  /** Nobody was detected in any frame: "no data", not "no fall". */
  no_person?: boolean;
  /** Missing frames were bridged; `interpolated_frames` lists them. */
  degraded?: boolean;
  interpolated_frames?: number[];
}

export const KNOWN_SCHEMA_VERSIONS = [1, 2];

export interface Faithfulness {
  frame: number;
  method: string;
  baseline: string;
  num_random: number;
  deletion: number[];
  insertion: number[];
  deletion_random: number[];
  insertion_random: number[];
  deletion_auc: number;
  insertion_auc: number;
  deletion_gap: number;
  insertion_gap: number;
  faithful: boolean;
  latency_ms: number;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch (cause) {
    // The API sleeps on the free tier and takes 30-60s to answer the first
    // request after waking. Say that, rather than "failed to fetch".
    throw new Error(
      `Cannot reach the API at ${API_URL}. It sleeps when idle and takes up to a ` +
        `minute to wake, so this usually clears on a retry.`
    );
  }
  if (!response.ok) {
    throw await toApiError(response);
  }
  return (await response.json()) as T;
}

/**
 * An error from the API with the fields the UI branches on.
 *
 * The backend sends `{error: {code, message, request_id, ...}}` from schema v2.
 * A v1 backend sends FastAPI's `{detail}`; both are understood, so the frontend
 * can ship before or after the backend does.
 */
export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public requestId?: string,
    public retryAfter?: number,
    public extra: Record<string, unknown> = {}
  ) {
    super(message);
    this.name = "ApiError";
  }

  /** The UI should offer a retry rather than a correction. */
  get retryable(): boolean {
    return this.code === "busy" || this.code === "out_of_memory" || this.status >= 500;
  }
}

async function toApiError(response: Response): Promise<ApiError> {
  const text = await response.text();
  const retryAfter = Number(response.headers.get("Retry-After")) || undefined;
  try {
    const parsed = JSON.parse(text);
    if (parsed?.error?.code) {
      const { code, message, request_id, ...extra } = parsed.error;
      return new ApiError(response.status, code, message, request_id, retryAfter, extra);
    }
    const detail = typeof parsed?.detail === "string" ? parsed.detail : text;
    return new ApiError(response.status, "legacy_error", detail, undefined, retryAfter);
  } catch {
    return new ApiError(response.status, "unparseable_error", text.slice(0, 300), undefined, retryAfter);
  }
}

/** `GET /model-card`: absent on backends older than schema v2, so callers must tolerate a 404. */
export interface ModelCard {
  generation: "current" | "earlier";
  generation_reasons: string[];
  model: { trained_on: string | null; feature_set: string | null; input_channels: number | null; phase_head: boolean };
  intended_use: string;
  out_of_scope: string[];
  limits: string[];
  evaluation: string;
}

/**
 * Whether a checkpoint is the project's current configuration. Mirrors the
 * backend's rule so the label still appears when the deployed backend predates
 * `/model-card`; a missing field counts as "earlier", never as "current".
 */
export function checkpointGeneration(model: ModelInfo): { current: boolean; reasons: string[] } {
  const reasons: string[] = [];
  if (model.feature_set !== "full") {
    reasons.push(`reads ${model.in_channels} input channels (${model.feature_set ?? "(x, y, vx, vy)"} features), not the full set`);
  }
  if (!model.has_phase_head) reasons.push("has no four-phase head (binary imminence only)");
  return { current: reasons.length === 0, reasons };
}

export const api = {
  modelCard: () => request<ModelCard>("/model-card"),

  /**
   * Health check that tolerates a cold start.
   *
   * Render holds the connection open while a sleeping container boots, so one
   * fetch usually succeeds after ~50 s. But a proxy or a flaky link can drop it
   * first, and the page would then report "api down" for a service that is
   * merely waking. Retrying a few times costs nothing and removes the most
   * common false alarm about the backend being broken.
   */
  health: async (attempts = 3, onAttempt?: (n: number) => void): Promise<Health> => {
    let lastError: unknown;
    for (let i = 0; i < attempts; i += 1) {
      onAttempt?.(i + 1);
      try {
        return await request<Health>("/health");
      } catch (error) {
        lastError = error;
        if (i < attempts - 1) await new Promise((r) => setTimeout(r, 3000));
      }
    }
    throw lastError;
  },

  skeleton: () => request<SkeletonSpec>("/skeleton"),

  clips: () => request<{ clips: ClipSummary[] }>("/clips").then((r) => r.clips),

  clip: (id: string) => request<ClipPayload>(`/clips/${encodeURIComponent(id)}`),

  analyze: (body: {
    keypoints: Keypoints;
    fps: number;
    impact_frame?: number | null;
    threshold?: number | null;
    persistence?: number | null;
    explain?: boolean;
    top_k?: number;
    clip_id?: string;
  }) => request<Analysis>("/analyze", { method: "POST", body: JSON.stringify(body) }),

  faithfulness: (body: {
    keypoints: Keypoints;
    frame: number;
    fps: number;
    impact_frame?: number | null;
    num_random?: number;
    baseline?: "zero" | "neighbour";
    clip_id?: string;
  }) => request<Faithfulness>("/faithfulness", { method: "POST", body: JSON.stringify(body) }),
};
