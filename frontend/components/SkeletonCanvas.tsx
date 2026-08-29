"use client";

/**
 * Skeleton playback with joint relevance overlaid.
 *
 * This is the whole point of the demo made visible: at the frame a warning
 * fired, the joints the model relied on are drawn larger and hotter than the
 * rest. It renders the same 17 joints and 16 bones the graph convolution uses,
 * fetched from the API rather than hard-coded, so the picture cannot drift out
 * of step with the model.
 */

import { useEffect, useMemo, useRef } from "react";
import type { Keypoints, SkeletonSpec } from "@/lib/api";
import { keypointBounds } from "@/lib/pose";

interface Props {
  keypoints: Keypoints;
  skeleton: SkeletonSpec;
  frame: number;
  /** [17] relevance, summing to 1, or null outside a warning. */
  relevance?: number[] | null;
  /** Colour the figure by imminence. */
  score?: number;
  threshold?: number;
  height?: number;
}

const MIN_CONFIDENCE = 0.2;

function cssVar(name: string, fallback: string): string {
  if (typeof window === "undefined") return fallback;
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

export default function SkeletonCanvas({
  keypoints,
  skeleton,
  frame,
  relevance = null,
  score = 0,
  threshold = 0.7,
  height = 340,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // The view is fitted to the whole clip, not to the current frame: a per-frame
  // fit would rescale as the person falls and visually cancel out the collapse
  // that the model is detecting.
  const bounds = useMemo(() => keypointBounds(keypoints, MIN_CONFIDENCE), [keypoints]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const pose = keypoints[Math.max(0, Math.min(frame, keypoints.length - 1))];
    if (!pose) return;

    const padding = 26;
    const spanX = Math.max(bounds.maxX - bounds.minX, 1);
    const spanY = Math.max(bounds.maxY - bounds.minY, 1);
    const scale = Math.min((width - padding * 2) / spanX, (height - padding * 2) / spanY);
    const offsetX = (width - spanX * scale) / 2 - bounds.minX * scale;
    const offsetY = (height - spanY * scale) / 2 - bounds.minY * scale;

    const project = (x: number, y: number): [number, number] => [
      x * scale + offsetX,
      y * scale + offsetY,
    ];

    const calm = cssVar("--calm", "#5a8f7b");
    const watch = cssVar("--watch", "#b98229");
    const imminent = cssVar("--imminent", "#c25a34");
    const faint = cssVar("--text-faint", "#8b8f99");
    const surface = cssVar("--surface", "#fff");

    // The figure itself carries the imminence reading, so the score is legible
    // without looking away at the chart.
    const bodyColour = score >= threshold ? imminent : score >= threshold * 0.6 ? watch : calm;

    // Ground line: a fall is defined relative to it, and without one the
    // skeleton floats in space and the drop is hard to read.
    const [, groundY] = project(0, bounds.maxY);
    ctx.strokeStyle = faint;
    ctx.globalAlpha = 0.28;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(padding * 0.4, groundY + 6);
    ctx.lineTo(width - padding * 0.4, groundY + 6);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;

    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    for (const [a, b] of skeleton.bones) {
      const ja = pose[a];
      const jb = pose[b];
      if (!ja || !jb || ja[2] < MIN_CONFIDENCE || jb[2] < MIN_CONFIDENCE) continue;
      const [x1, y1] = project(ja[0], ja[1]);
      const [x2, y2] = project(jb[0], jb[1]);
      ctx.strokeStyle = bodyColour;
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // Scale relevance by its own maximum rather than by 1/17, so the top joints
    // are visible whether the attention is peaked or nearly uniform.
    const peak = relevance ? Math.max(...relevance, 1e-6) : 0;

    pose.forEach((joint, index) => {
      if (joint[2] < MIN_CONFIDENCE) return;
      const [x, y] = project(joint[0], joint[1]);
      const weight = relevance ? relevance[index] / peak : 0;

      if (relevance && weight > 0.25) {
        ctx.fillStyle = imminent;
        ctx.globalAlpha = 0.16 * weight;
        ctx.beginPath();
        ctx.arc(x, y, 6 + 16 * weight, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.globalAlpha = 1;
      ctx.beginPath();
      ctx.arc(x, y, relevance ? 2.6 + 4 * weight : 3.2, 0, Math.PI * 2);
      ctx.fillStyle = relevance && weight > 0.25 ? imminent : bodyColour;
      ctx.fill();
      ctx.strokeStyle = surface;
      ctx.lineWidth = 1.2;
      ctx.stroke();
    });
    ctx.globalAlpha = 1;
  }, [keypoints, skeleton, frame, relevance, score, threshold, height, bounds]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: "100%", height, display: "block", borderRadius: 8 }}
      role="img"
      aria-label={
        relevance
          ? "Skeleton at the warning frame, with the joints driving the warning highlighted"
          : "Skeleton playback"
      }
    />
  );
}
