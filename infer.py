"""Run a trained model as a live warning system.

    python infer.py --checkpoint <run>/best.pt --clip data/cache/synthetic/S00_fall00.npz
    python infer.py --checkpoint <run>/best.pt --video hallway.mp4 --pose rtmpose

Everything here is strictly causal and strictly streaming: one frame in, one
score out, a fixed-size buffer, no lookahead. That is what makes the throughput
number it prints a real claim about deployment (Section 17, "real-time on cheap
hardware") rather than an offline batch benchmark.

Each warning is emitted with the joints that drove it. Whether that evidence is
*faithful* is not something inference can establish - run `eval.py --explain` for
the deletion/insertion test that answers RQ2.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from data.clips import ClipRecord
from data.stream import SkeletonStream
from evaluation.decision import DecisionConfig, OnlineTrigger
from explain.relevance import joint_relevance
from utils.logging import get_logger, save_json
from utils.seed import resolve_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--clip", help="a cached .npz skeleton clip")
    source.add_argument("--video", help="a video file (needs a pose backend)")
    parser.add_argument("--pose", default="rtmpose", help="pose backend for --video")
    parser.add_argument("--pose-device", default="cuda")
    parser.add_argument("--device", default="auto",
                        help="auto | cuda | cpu. At batch 1 this model is kernel-launch "
                             "bound, so measure both before claiming a deployment number")
    parser.add_argument("--threshold", type=float, default=None,
                        help="override the checkpoint's tau (not recommended)")
    parser.add_argument("--persistence", type=int, default=None, help="override k")
    parser.add_argument("--explain", action="store_true", default=True,
                        help="attach joint evidence to each warning")
    parser.add_argument("--no-explain", dest="explain", action="store_false")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--out", default=None, help="write warnings to this JSON file")
    return parser.parse_args(argv)


def stream_from_video(video_path: str, backend: str, device: str, logger):
    """Yield `[V, 3]` keypoints frame by frame, discarding each frame after use."""
    import cv2

    from pose.backends import build_estimator
    from pose.base import empty_keypoints, select_subject, track_greedy

    estimator = build_estimator(backend, device=device)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise IOError(f"could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    logger.info("video %s at %.1f fps via %s", video_path, fps, estimator.name)

    def generator():
        previous = None
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                detections = estimator.detect(frame)
                chosen = (
                    track_greedy(previous, detections)
                    if previous is not None
                    else select_subject(detections, frame.shape[:2])
                )
                previous = chosen
                yield chosen.keypoints if chosen is not None else empty_keypoints()
        finally:
            capture.release()
            estimator.close()

    return generator(), fps


def main(argv: list[str] | None = None) -> list[dict]:
    args = parse_args(argv)
    logger = get_logger("infer")
    device = resolve_device(args.device)

    from eval import load_checkpoint

    model, cfg, checkpoint = load_checkpoint(args.checkpoint, device)
    stored = checkpoint.get("decision") or cfg["decision"]
    decision = DecisionConfig(
        threshold=float(args.threshold if args.threshold is not None else stored["threshold"]),
        persistence=int(args.persistence if args.persistence is not None else stored["persistence"]),
        refractory_frames=int(stored["refractory_frames"]),
    )
    logger.info("tau=%.2f k=%d refractory=%d",
                decision.threshold, decision.persistence, decision.refractory_frames)

    # -- source ----------------------------------------------------------------
    impact_frame = None
    clip_id = "stream"
    if args.clip:
        clip = ClipRecord.load(args.clip)
        frames = iter(clip.keypoints)
        fps = clip.fps
        impact_frame = clip.impact_frame
        clip_id = clip.clip_id
        total = clip.num_frames
        logger.info("clip %s | %d frames at %.1f fps | label=%s",
                    clip.clip_id, total, fps, clip.label)
    else:
        frames, fps = stream_from_video(args.video, args.pose, args.pose_device, logger)
        clip_id = Path(args.video).stem
        total = None

    features_cfg = cfg["features"]
    stream = SkeletonStream(
        window=int(features_cfg["window"]),
        fps=fps,
        conf_threshold=float(features_cfg.get("conf_threshold", 0.3)),
        with_velocity=bool(features_cfg.get("with_velocity", True)),
    )
    trigger = OnlineTrigger(decision)

    # -- loop ------------------------------------------------------------------
    warnings: list[dict] = []
    scores: list[float] = []
    started = time.perf_counter()
    processed = 0

    for index, keypoints in enumerate(frames):
        window = stream.push(keypoints)
        processed += 1
        if window is None:
            scores.append(float("nan"))
            continue

        tensor = torch.from_numpy(window).unsqueeze(0).unsqueeze(-1).to(device)
        with torch.no_grad():
            score = float(torch.sigmoid(model(tensor))[0, -1].item())
        scores.append(score)

        if not trigger.update(score):
            continue

        entry = {"clip_id": clip_id, "frame": index, "score": score,
                 "time_s": index / fps}
        if impact_frame is not None:
            lead = (impact_frame - index) / fps
            entry["lead_time"] = lead if index <= impact_frame else None
        if args.explain:
            relevance = joint_relevance(model, tensor, -1, device,
                                        method=str(cfg["explain"].get("method", "attention")))
            entry["evidence"] = relevance.phrase(args.top_k)
            entry["top_joints"] = [
                {"joint": n, "relevance": s} for n, s in relevance.top_k(args.top_k)
            ]
        warnings.append(entry)

        lead_text = (
            f", {entry['lead_time']:.2f}s before impact"
            if entry.get("lead_time") is not None else ""
        )
        logger.warning(
            "FALL IMMINENT | frame %d (%.2fs%s) | p=%.2f | evidence: %s",
            index, entry["time_s"], lead_text, score, entry.get("evidence", "n/a"),
        )

    elapsed = time.perf_counter() - started
    throughput = processed / max(elapsed, 1e-9)
    logger.info(
        "processed %d frames in %.2fs = %.1f fps (%.1fx real time at %.0f fps source)",
        processed, elapsed, throughput, throughput / fps, fps,
    )

    if impact_frame is not None and not warnings:
        logger.info("no warning fired before impact - this clip is a miss")

    result = {
        "clip_id": clip_id,
        "fps": fps,
        "frames": processed,
        "throughput_fps": throughput,
        "warnings": warnings,
        "scores": [None if np.isnan(s) else round(s, 5) for s in scores],
    }
    if args.out:
        save_json(result, args.out)
        logger.info("wrote %s", args.out)
    return warnings


if __name__ == "__main__":
    main()
