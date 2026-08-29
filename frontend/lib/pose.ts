/**
 * In-browser pose estimation - the privacy boundary, client side.
 *
 * MoveNet outputs COCO-17 keypoints in exactly the order `data/skeleton.py`
 * expects, so no remapping is needed. Frames are decoded, posed and discarded
 * inside this module; the only thing that ever leaves it is a `[T, 17, 3]`
 * array of coordinates, which is the same thing `pose/cache.py` writes to disk
 * on the training side.
 *
 * The libraries are imported dynamically so the ~1 MB TensorFlow.js bundle is
 * only fetched when a user actually opens a video, rather than on every page
 * load for the demo-clip path that does not need it.
 */

import type { Keypoints } from "./api";

export const NUM_JOINTS = 17;

type Detector = {
  estimatePoses: (
    image: HTMLVideoElement | HTMLCanvasElement,
    config?: { flipHorizontal?: boolean }
  ) => Promise<
    { keypoints: { x: number; y: number; score?: number; name?: string }[] }[]
  >;
  dispose?: () => void;
};

let detectorPromise: Promise<Detector> | null = null;

/** Load MoveNet once per page. */
export async function getDetector(): Promise<Detector> {
  if (!detectorPromise) {
    detectorPromise = (async () => {
      const tf = await import("@tensorflow/tfjs-core");
      await import("@tensorflow/tfjs-backend-webgl");
      await import("@tensorflow/tfjs-converter");
      const poseDetection = await import("@tensorflow-models/pose-detection");

      await tf.setBackend("webgl");
      await tf.ready();

      return (await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        {
          modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
          enableSmoothing: true,
        }
      )) as unknown as Detector;
    })();
  }
  return detectorPromise;
}

/** A pose with no detection: zero coordinates at zero confidence. */
function emptyFrame(): number[][] {
  return Array.from({ length: NUM_JOINTS }, () => [0, 0, 0]);
}

function toFrame(
  poses: Awaited<ReturnType<Detector["estimatePoses"]>>
): number[][] {
  if (!poses.length || !poses[0].keypoints?.length) return emptyFrame();
  const kp = poses[0].keypoints;
  return Array.from({ length: NUM_JOINTS }, (_, i) => {
    const k = kp[i];
    // Zero confidence is the honest signal for a missing joint: the training
    // pipeline interpolates across it rather than trusting the coordinates.
    if (!k) return [0, 0, 0];
    return [k.x ?? 0, k.y ?? 0, k.score ?? 0];
  });
}

export interface ExtractOptions {
  /** Cap on frames, matching the API's own limit. */
  maxFrames?: number;
  /** Sample every Nth frame; the effective fps is divided to match. */
  stride?: number;
  onProgress?: (done: number, total: number) => void;
  signal?: AbortSignal;
}

/**
 * Decode a video element frame by frame and return its skeleton sequence.
 *
 * Seeks explicitly rather than playing in real time, so a 10-second clip is
 * processed as fast as the GPU allows and every frame is sampled deterministically
 * instead of being subject to dropped frames during playback.
 */
export async function extractFromVideo(
  video: HTMLVideoElement,
  options: ExtractOptions = {}
): Promise<{ keypoints: Keypoints; fps: number; frames: number }> {
  const { maxFrames = 900, stride = 1, onProgress, signal } = options;
  const detector = await getDetector();

  // Browsers do not expose a video's true frame rate. 30 is the safe assumption
  // for phone and webcam footage; lead time is reported in seconds, so a wrong
  // value here would scale every number - it is surfaced in the UI for that reason.
  const assumedFps = 30;
  const duration = video.duration;
  if (!isFinite(duration) || duration <= 0) {
    throw new Error("could not read the video's duration");
  }

  const step = stride / assumedFps;
  const total = Math.min(Math.floor(duration / step), maxFrames);
  if (total < 30) {
    throw new Error(
      `the video is too short: ${total} sampled frames, and the model needs at least 30`
    );
  }

  const frames: number[][][] = [];
  for (let i = 0; i < total; i += 1) {
    if (signal?.aborted) throw new Error("cancelled");
    await seek(video, i * step);
    const poses = await detector.estimatePoses(video, { flipHorizontal: false });
    frames.push(toFrame(poses));
    if (onProgress && i % 5 === 0) onProgress(i + 1, total);
  }
  onProgress?.(total, total);

  return { keypoints: frames, fps: assumedFps / stride, frames: total };
}

function seek(video: HTMLVideoElement, time: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const onSeeked = () => {
      video.removeEventListener("seeked", onSeeked);
      video.removeEventListener("error", onError);
      resolve();
    };
    const onError = () => {
      video.removeEventListener("seeked", onSeeked);
      video.removeEventListener("error", onError);
      reject(new Error("seek failed"));
    };
    video.addEventListener("seeked", onSeeked);
    video.addEventListener("error", onError);
    video.currentTime = time;
  });
}

/** Bounding box over every confident joint, for fitting a skeleton to a canvas. */
export function keypointBounds(
  keypoints: Keypoints,
  minConfidence = 0.2
): { minX: number; minY: number; maxX: number; maxY: number } {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const frame of keypoints) {
    for (const [x, y, c] of frame) {
      if (c < minConfidence) continue;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
  }
  if (!isFinite(minX)) return { minX: 0, minY: 0, maxX: 1, maxY: 1 };
  return { minX, minY, maxX, maxY };
}
