"use client";

/**
 * three.js skeleton viewer.
 *
 * Joints are spheres, bones are cylinders re-oriented each frame, and the whole
 * figure sits on a lit ground plane that receives its shadow. The camera orbits
 * within a limited arc and can be dragged.
 *
 * The scene is 3D; the *data* is not. There is no depth estimate in this
 * project, so the figure is planar and labelled "2D pose - 3D stage" in the
 * viewer. The ground plane earns its place: a fall is defined by contact with
 * it, and without it a collapse reads as a shape change rather than a fall.
 *
 * Joint relevance from Stage E drives emissive intensity and radius, so the
 * evidence for a warning is legible on the body itself rather than only in a
 * list beside it.
 */

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { COCO_BONES, MINOR_JOINTS, computeTransform, toWorld } from "@/lib/skeleton3d";

interface Props {
  keypoints: number[][][];
  frame: number;
  /** [17] relevance summing to 1, or null when not at a warning. */
  relevance?: number[] | null;
  /** Per-frame imminence, drives the body colour. */
  score?: number;
  threshold?: number;
  height?: number;
  /** Slow continuous orbit; disabled while the user drags. */
  autoRotate?: boolean;
  label?: string;
  interactive?: boolean;
}

const CALM = new THREE.Color("#4ec9a5");
const WATCH = new THREE.Color("#f0b04a");
const IMMINENT = new THREE.Color("#ff8354");
const EVIDENCE = new THREE.Color("#ff8354");
const MIN_CONF = 0.2;

export default function Scene3D({
  keypoints,
  frame,
  relevance = null,
  score = 0,
  threshold = 0.7,
  height = 380,
  autoRotate = true,
  label = "2D pose · 3D stage",
  interactive = true,
}: Props) {
  const mountRef = useRef<HTMLDivElement>(null);

  // Latest props, read inside the animation loop without re-creating the scene.
  // Written in an effect rather than during render: React may render a
  // component twice before committing, and a ref mutation during render is not
  // guaranteed to survive that. One frame of staleness (~16 ms) is invisible.
  const state = useRef({ keypoints, frame, relevance, score, threshold, autoRotate });
  useEffect(() => {
    state.current = { keypoints, frame, relevance, score, threshold, autoRotate };
  });

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const width = mount.clientWidth || 400;
    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0x05070c, 9, 22);

    const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 100);
    const camState = { theta: -0.32, phi: 1.28, radius: 8.4, target: new THREE.Vector3(0, 1.5, 0) };

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    } catch {
      // No WebGL (locked-down browser, headless). Leave the container empty
      // rather than throwing during render - the rest of the page still works.
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    mount.appendChild(renderer.domElement);

    // -- lighting ------------------------------------------------------------
    scene.add(new THREE.AmbientLight(0x5c7a92, 0.85));

    const key = new THREE.DirectionalLight(0xffffff, 2.1);
    key.position.set(4.5, 8, 5.5);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.near = 1;
    key.shadow.camera.far = 24;
    key.shadow.camera.left = -6;
    key.shadow.camera.right = 6;
    key.shadow.camera.top = 8;
    key.shadow.camera.bottom = -2;
    key.shadow.bias = -0.0012;
    scene.add(key);

    const rim = new THREE.DirectionalLight(0x3ddad7, 1.5);
    rim.position.set(-5, 3.5, -4.5);
    scene.add(rim);

    const fill = new THREE.PointLight(0x6a8dff, 22, 20, 2);
    fill.position.set(0, 4.5, 5);
    scene.add(fill);

    // -- ground --------------------------------------------------------------
    const ground = new THREE.Mesh(
      new THREE.CircleGeometry(9, 64),
      new THREE.MeshStandardMaterial({
        color: 0x0b1119, roughness: 0.86, metalness: 0.12,
        transparent: true, opacity: 0.92,
      })
    );
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    scene.add(ground);

    const grid = new THREE.GridHelper(18, 36, 0x3ddad7, 0x1b2430);
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.15;
    grid.position.y = 0.004;
    scene.add(grid);

    // -- skeleton ------------------------------------------------------------
    const jointGeo = new THREE.SphereGeometry(1, 20, 16);
    const boneGeo = new THREE.CylinderGeometry(1, 1, 1, 12, 1, true);

    const joints = Array.from({ length: 17 }, (_, i) => {
      const mat = new THREE.MeshStandardMaterial({
        color: CALM.clone(), emissive: CALM.clone(), emissiveIntensity: 0.4,
        roughness: 0.32, metalness: 0.55,
      });
      const mesh = new THREE.Mesh(jointGeo, mat);
      mesh.castShadow = true;
      mesh.scale.setScalar(MINOR_JOINTS.has(i) ? 0.038 : 0.072);
      scene.add(mesh);
      return mesh;
    });

    // A translucent halo per joint, scaled by relevance - this is what makes
    // "these joints drove the warning" visible at a glance.
    const halos = Array.from({ length: 17 }, () => {
      const mat = new THREE.MeshBasicMaterial({
        color: EVIDENCE.clone(), transparent: true, opacity: 0, depthWrite: false,
      });
      const mesh = new THREE.Mesh(jointGeo, mat);
      mesh.visible = false;
      scene.add(mesh);
      return mesh;
    });

    const bones = COCO_BONES.map(() => {
      const mat = new THREE.MeshStandardMaterial({
        color: CALM.clone(), emissive: CALM.clone(), emissiveIntensity: 0.16,
        roughness: 0.38, metalness: 0.4, transparent: true, opacity: 0.9,
      });
      const mesh = new THREE.Mesh(boneGeo, mat);
      mesh.castShadow = true;
      scene.add(mesh);
      return mesh;
    });

    // -- interaction ---------------------------------------------------------
    let dragging = false;
    let lastX = 0, lastY = 0, idleUntil = 0;

    const onDown = (e: PointerEvent) => {
      if (!interactive) return;
      dragging = true; lastX = e.clientX; lastY = e.clientY;
      renderer.domElement.setPointerCapture(e.pointerId);
    };
    const onMove = (e: PointerEvent) => {
      if (!dragging) return;
      camState.theta -= (e.clientX - lastX) * 0.006;
      camState.phi = THREE.MathUtils.clamp(camState.phi - (e.clientY - lastY) * 0.005, 0.62, 1.62);
      lastX = e.clientX; lastY = e.clientY;
      idleUntil = performance.now() + 4000;   // pause auto-orbit after a drag
    };
    const onUp = (e: PointerEvent) => {
      dragging = false;
      try { renderer.domElement.releasePointerCapture(e.pointerId); } catch { /* already released */ }
    };
    if (interactive) {
      renderer.domElement.style.cursor = "grab";
      renderer.domElement.addEventListener("pointerdown", onDown);
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    }

    const resize = () => {
      const w = mount.clientWidth || width;
      camera.aspect = w / height;
      camera.updateProjectionMatrix();
      renderer.setSize(w, height);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(mount);

    // -- loop ----------------------------------------------------------------
    const up = new THREE.Vector3(0, 1, 0);
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    const mid = new THREE.Vector3();
    const dir = new THREE.Vector3();
    const bodyColour = new THREE.Color();

    let raf = 0;
    let prev = performance.now();

    const tick = (now: number) => {
      raf = requestAnimationFrame(tick);
      const dt = Math.min((now - prev) / 1000, 0.1);
      prev = now;

      const s = state.current;
      const kp = s.keypoints;
      if (!kp?.length) { renderer.render(scene, camera); return; }

      if (s.autoRotate && !dragging && now > idleUntil) camState.theta += dt * 0.14;

      camera.position.set(
        camState.target.x + camState.radius * Math.sin(camState.phi) * Math.sin(camState.theta),
        camState.target.y + camState.radius * Math.cos(camState.phi),
        camState.target.z + camState.radius * Math.sin(camState.phi) * Math.cos(camState.theta)
      );
      camera.lookAt(camState.target);

      const t = computeTransform(kp, MIN_CONF);
      const pose = kp[Math.max(0, Math.min(s.frame, kp.length - 1))];
      if (!pose) { renderer.render(scene, camera); return; }

      // Body colour tracks imminence, so risk is readable without the chart.
      const ratio = s.threshold > 0 ? s.score / s.threshold : 0;
      if (ratio >= 1) bodyColour.copy(IMMINENT);
      else if (ratio >= 0.6) bodyColour.copy(WATCH).lerp(IMMINENT, (ratio - 0.6) / 0.4);
      else bodyColour.copy(CALM).lerp(WATCH, Math.max(ratio, 0) / 0.6);

      const peak = s.relevance ? Math.max(...s.relevance, 1e-6) : 0;

      for (let i = 0; i < 17; i += 1) {
        const p = pose[i];
        const visible = !!p && p[2] >= MIN_CONF;
        joints[i].visible = visible;
        halos[i].visible = false;
        if (!visible) continue;

        const [x, y, z] = toWorld(p[0], p[1], t);
        joints[i].position.set(x, y, z);

        const w = s.relevance ? s.relevance[i] / peak : 0;
        const mat = joints[i].material as THREE.MeshStandardMaterial;
        const base = MINOR_JOINTS.has(i) ? 0.038 : 0.072;

        if (s.relevance && w > 0.22) {
          mat.color.copy(EVIDENCE);
          mat.emissive.copy(EVIDENCE);
          mat.emissiveIntensity = 0.55 + 1.5 * w;
          joints[i].scale.setScalar(base * (1 + 1.15 * w));
          halos[i].visible = true;
          halos[i].position.set(x, y, z);
          halos[i].scale.setScalar(base * (2.6 + 6.5 * w));
          (halos[i].material as THREE.MeshBasicMaterial).opacity = 0.1 + 0.2 * w;
        } else {
          mat.color.copy(bodyColour);
          mat.emissive.copy(bodyColour);
          mat.emissiveIntensity = 0.4;
          joints[i].scale.setScalar(base);
        }
      }

      COCO_BONES.forEach(([i, j], k) => {
        const pi = pose[i], pj = pose[j];
        const bone = bones[k];
        if (!pi || !pj || pi[2] < MIN_CONF || pj[2] < MIN_CONF) { bone.visible = false; return; }
        bone.visible = true;

        a.set(...toWorld(pi[0], pi[1], t));
        b.set(...toWorld(pj[0], pj[1], t));
        mid.addVectors(a, b).multiplyScalar(0.5);
        dir.subVectors(b, a);
        const len = dir.length() || 1e-6;

        bone.position.copy(mid);
        bone.quaternion.setFromUnitVectors(up, dir.normalize());
        bone.scale.set(0.026, len, 0.026);

        const mat = bone.material as THREE.MeshStandardMaterial;
        mat.color.copy(bodyColour);
        mat.emissive.copy(bodyColour);
      });

      renderer.render(scene, camera);
    };
    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
      observer.disconnect();
      if (interactive) {
        renderer.domElement.removeEventListener("pointerdown", onDown);
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
      }
      // three.js does not free GPU memory on garbage collection; without this a
      // few navigations between panes leak enough buffers to stall the tab.
      jointGeo.dispose();
      boneGeo.dispose();
      ground.geometry.dispose();
      (ground.material as THREE.Material).dispose();
      grid.geometry.dispose();
      (grid.material as THREE.Material).dispose();
      [...joints, ...halos, ...bones].forEach((m) => (m.material as THREE.Material).dispose());
      renderer.dispose();
      if (renderer.domElement.parentNode === mount) mount.removeChild(renderer.domElement);
    };
  }, [height, interactive]);

  return (
    <div className="scene" style={{ height }}>
      <div ref={mountRef} style={{ width: "100%", height }} />
      <span className="scene-tag">{label}</span>
      {interactive && <span className="scene-hint">drag to orbit</span>}
    </div>
  );
}
