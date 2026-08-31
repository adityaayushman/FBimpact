/**
 * Skeleton geometry helpers shared by the 3D scene.
 *
 * A note on dimensionality, because it matters for honesty: the pipeline is
 * **2D pose**. There is no depth estimate anywhere in this project, and none is
 * invented here. The skeleton is rendered as a planar figure placed in a real
 * 3D scene - lit, casting a shadow onto a ground plane, viewed from an orbiting
 * camera. The 3D is the *stage*, not fabricated Z data, and the viewer is
 * labelled as such.
 *
 * The ground plane is not decoration either: a fall is defined by contact with
 * it, so having it in frame is what makes a collapse legible as a collapse
 * rather than as a shape changing.
 */

export const COCO_BONES: [number, number][] = [
  [0, 1], [0, 2], [1, 3], [2, 4],
  [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 6], [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16],
];

export const JOINT_NAMES = [
  "nose", "left_eye", "right_eye", "left_ear", "right_ear",
  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
  "left_wrist", "right_wrist", "left_hip", "right_hip",
  "left_knee", "right_knee", "left_ankle", "right_ankle",
];

/** Joints drawn small - the face reads as clutter at scene scale. */
export const MINOR_JOINTS = new Set([1, 2, 3, 4]);

export interface Frame2D {
  /** [17][3] of (x, y, confidence) in whatever pixel space the source used. */
  points: number[][];
}

/**
 * Fit a clip into a fixed world box, preserving aspect.
 *
 * Fitted over the *whole* clip rather than per frame: a per-frame fit would
 * rescale as the person goes down and visually cancel the very collapse the
 * model is detecting.
 */
export function computeTransform(
  keypoints: number[][][],
  minConfidence = 0.2,
  worldHeight = 3.4
) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const frame of keypoints) {
    for (const [x, y, c] of frame) {
      if (c < minConfidence) continue;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
  }
  if (!isFinite(minX)) {
    return { scale: 1, offsetX: 0, offsetY: 0, groundY: 0 };
  }

  const spanY = Math.max(maxY - minY, 1e-6);
  const scale = worldHeight / spanY;
  return {
    scale,
    // Centre horizontally; put the lowest observed point on y = 0 (the floor).
    offsetX: -((minX + maxX) / 2) * scale,
    offsetY: maxY * scale,
    groundY: 0,
  };
}

export interface Transform {
  scale: number;
  offsetX: number;
  offsetY: number;
}

/** Pixel space (y down) -> world space (y up), on the z = 0 plane. */
export function toWorld(x: number, y: number, t: Transform): [number, number, number] {
  return [x * t.scale + t.offsetX, t.offsetY - y * t.scale, 0];
}

/* ---------------------------------------------------------------------------
   Schematic pose generator for the hero.

   This is a *diagram of the graph*, not model output and not data - it is
   labelled that way in the UI. It exists so the landing view can show the
   17-joint topology the model operates on without waiting on the API, and
   without a still image standing in for a moving system.
   ------------------------------------------------------------------------ */

function rot(x: number, y: number, a: number): [number, number] {
  const c = Math.cos(a), s = Math.sin(a);
  return [x * c - y * s, x * s + y * c];
}

/**
 * Build a schematic COCO-17 pose from joint angles, in pixel-like coordinates
 * (y increases downward), mirroring `data/synthetic.py::_pose` so the hero and
 * the real pipeline agree on proportions.
 */
export function schematicPose(
  trunkAngle: number,
  hipFlex: number,
  kneeFlex: number,
  armRaise: number,
  drop: number,
  torso = 100
): number[][] {
  const P = {
    hipHalf: 0.18, shoulderHalf: 0.22, neck: 0.16, head: 0.22,
    upperArm: 0.32, forearm: 0.3, thigh: 0.5, shank: 0.48,
  };
  const pts: number[][] = Array.from({ length: 17 }, () => [0, 0, 1]);

  const rootX = 0;
  const rootY = drop * torso;

  const [tdx, tdy] = rot(0, -1, trunkAngle);          // trunk direction
  const sx = tdy, sy = -tdx;                           // perpendicular

  const set = (i: number, x: number, y: number) => { pts[i] = [x, y, 1]; };

  set(11, rootX + sx * P.hipHalf * torso, rootY + sy * P.hipHalf * torso);
  set(12, rootX - sx * P.hipHalf * torso, rootY - sy * P.hipHalf * torso);

  const shX = rootX + tdx * torso, shY = rootY + tdy * torso;
  set(5, shX + sx * P.shoulderHalf * torso, shY + sy * P.shoulderHalf * torso);
  set(6, shX - sx * P.shoulderHalf * torso, shY - sy * P.shoulderHalf * torso);

  const nkX = shX + tdx * P.neck * torso, nkY = shY + tdy * P.neck * torso;
  const noX = nkX + tdx * P.head * torso, noY = nkY + tdy * P.head * torso;
  set(0, noX, noY);
  set(1, noX + sx * 0.04 * torso, noY + sy * 0.04 * torso);
  set(2, noX - sx * 0.04 * torso, noY - sy * 0.04 * torso);
  set(3, nkX + sx * 0.08 * torso + tdx * 0.12 * torso, nkY + sy * 0.08 * torso + tdy * 0.12 * torso);
  set(4, nkX - sx * 0.08 * torso + tdx * 0.12 * torso, nkY - sy * 0.08 * torso + tdy * 0.12 * torso);

  ([[1, 5, 7, 9], [-1, 6, 8, 10]] as const).forEach(([sign, sh, el, wr]) => {
    const [ux, uy] = rot(-tdx, -tdy, sign * armRaise);
    const ex = pts[sh][0] + ux * P.upperArm * torso;
    const ey = pts[sh][1] + uy * P.upperArm * torso;
    const [fx, fy] = rot(ux, uy, sign * 0.35);
    set(el, ex, ey);
    set(wr, ex + fx * P.forearm * torso, ey + fy * P.forearm * torso);
  });

  ([[1, 11, 13, 15], [-1, 12, 14, 16]] as const).forEach(([sign, hp, kn, an]) => {
    const [thx, thy] = rot(0, 1, hipFlex * sign);
    const kx = pts[hp][0] + thx * P.thigh * torso;
    const ky = pts[hp][1] + thy * P.thigh * torso;
    const [shx, shy] = rot(thx, thy, -kneeFlex);
    set(kn, kx, ky);
    set(an, kx + shx * P.shank * torso, ky + shy * P.shank * torso);
  });

  return pts;
}

/**
 * A looping schematic sequence: stand, lose balance, collapse, reset.
 *
 * The ballistic phase uses a squared ramp so lean and drop accelerate into
 * contact - the same second-derivative signature that distinguishes a fall from
 * a controlled descent, and the thing the temporal model is looking for.
 */
export function schematicFallLoop(frames = 150): number[][][] {
  const out: number[][][] = [];
  const onset = Math.round(frames * 0.36);
  const impact = Math.round(frames * 0.72);

  for (let t = 0; t < frames; t += 1) {
    let lean = 0, hip = 0.05, knee = 0.1, arm = 0, drop = 0;

    if (t < onset) {
      const sway = Math.sin((t / onset) * Math.PI * 2) * 0.035;
      lean = sway;
      knee = 0.1 + Math.abs(sway) * 0.5;
    } else if (t <= impact) {
      const p = (t - onset) / (impact - onset);
      const ballistic = p * p;
      lean = (Math.PI / 2 - 0.16) * ballistic;
      drop = 0.86 * ballistic;
      hip = 0.35 * ballistic;
      knee = 0.1 + 0.95 * ballistic * (1 - 0.4 * ballistic);
      arm = 1.15 * p;
    } else {
      const p = Math.min((t - impact) / (frames - impact), 1);
      const ease = p * p * (3 - 2 * p);
      lean = (Math.PI / 2 - 0.16) * (1 - ease);
      drop = 0.86 * (1 - ease);
      hip = 0.35 * (1 - ease);
      knee = 0.1 + 0.95 * (1 - ease);
      arm = 1.15 * (1 - ease);
    }
    out.push(schematicPose(lean, hip, knee, arm, drop));
  }
  return out;
}

/** Frame index of ground contact in `schematicFallLoop`. */
export const SCHEMATIC_IMPACT = (frames = 150) => Math.round(frames * 0.72);
