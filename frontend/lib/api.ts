/**
 * Typed client for the inference API.
 *
 * The API accepts skeletons and nothing else - there is no endpoint that takes a
 * frame. Pose estimation runs in the browser (see lib/pose.ts), so video never
 * leaves the device.
 */

export const API_URL = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000"
).replace(/\/$/, "");

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

export interface Analysis {
  clip_id: string;
  frames: number;
  fps: number;
  impact_frame: number | null;
  scores: number[];
  warnings: Warning[];
  decision: Decision;
  imminent_window: [number, number] | null;
  latency_ms: number;
}

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
    // A free-tier container that has spun down takes ~30-60s to answer its first
    // request, and the fetch fails outright rather than returning a status - so
    // say that, instead of "failed to fetch".
    throw new Error(
      `Cannot reach the API at ${API_URL}. If it is hosted on Render's free tier ` +
        `it may be cold-starting; wait ~30s and retry.`
    );
  }
  if (!response.ok) {
    const detail = await response.text();
    let message = detail;
    try {
      message = JSON.parse(detail).detail ?? detail;
    } catch {
      /* the body was not JSON; use it as-is */
    }
    throw new Error(`${response.status}: ${message}`);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<Health>("/health"),

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
