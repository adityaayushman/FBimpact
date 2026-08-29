"""Concrete pose backends, imported lazily.

None of these are dependencies of the training or evaluation path: once
`scripts/cache_poses.py` has run, the project works on cached skeletons alone.
That is why every import happens inside `__init__` and every failure produces an
instruction rather than a stack trace - a reviewer reproducing the results from
the released skeleton cache should never need to install a pose estimator at all.
"""

from __future__ import annotations

import numpy as np

from data.skeleton import NUM_JOINTS

from .base import Detection, PoseEstimator

_INSTALL_HINTS = {
    "yolo": "pip install ultralytics",
    "rtmpose": "pip install rtmlib onnxruntime  # or onnxruntime-gpu",
}


class YoloPose(PoseEstimator):
    """Ultralytics YOLO-Pose (COCO-17 output, single stage)."""

    name = "yolo-pose"

    def __init__(
        self,
        weights: str = "yolo11n-pose.pt",
        device: str = "cuda",
        conf: float = 0.25,
        imgsz: int = 640,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                f"YoloPose needs ultralytics ({_INSTALL_HINTS['yolo']})"
            ) from exc

        self.model = YOLO(weights)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        # Frozen: this wrapper never calls train() or requires_grad_(True).
        self.model.model.eval()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame, device=self.device, conf=self.conf, imgsz=self.imgsz, verbose=False
        )
        detections: list[Detection] = []
        for result in results:
            if result.keypoints is None or result.keypoints.data is None:
                continue
            keypoints = result.keypoints.data.cpu().numpy()          # [n, V, 3]
            boxes = (
                result.boxes.xyxy.cpu().numpy() if result.boxes is not None else None
            )
            scores = (
                result.boxes.conf.cpu().numpy()
                if result.boxes is not None
                else np.ones(len(keypoints), dtype=np.float32)
            )
            for i, kp in enumerate(keypoints):
                if kp.shape[0] != NUM_JOINTS:
                    continue
                detections.append(
                    Detection(
                        keypoints=kp.astype(np.float32),
                        box=tuple(boxes[i]) if boxes is not None else None,
                        score=float(scores[i]),
                    )
                )
        return detections


class RtmPose(PoseEstimator):
    """RTMPose via rtmlib (ONNX Runtime), the default in Section 12."""

    name = "rtmpose"

    def __init__(
        self,
        mode: str = "balanced",
        backend: str = "onnxruntime",
        device: str = "cuda",
        conf: float = 0.3,
    ) -> None:
        try:
            from rtmlib import Wholebody  # noqa: F401  (import check only)
            from rtmlib import Body
        except ImportError as exc:
            raise ImportError(
                f"RtmPose needs rtmlib ({_INSTALL_HINTS['rtmpose']})"
            ) from exc

        self.model = Body(mode=mode, backend=backend, device=device, to_openpose=False)
        self.conf = conf

    def detect(self, frame: np.ndarray) -> list[Detection]:
        keypoints, scores = self.model(frame)
        detections: list[Detection] = []
        for kp, sc in zip(np.asarray(keypoints), np.asarray(scores)):
            if kp.shape[0] < NUM_JOINTS:
                continue
            # rtmlib's Body returns COCO-17 first; higher-order layouts append.
            merged = np.concatenate(
                [kp[:NUM_JOINTS].astype(np.float32), sc[:NUM_JOINTS, None].astype(np.float32)],
                axis=1,
            )
            detections.append(Detection(keypoints=merged, score=float(sc[:NUM_JOINTS].mean())))
        return detections


class ReplayPose(PoseEstimator):
    """Replays pre-extracted keypoints, for tests and for re-running the cache.

    Lets the whole Stage A path be exercised - subject selection, tracking,
    caching - without a model, a GPU or a video file.
    """

    name = "replay"

    def __init__(self, keypoints: np.ndarray) -> None:
        self.keypoints = np.asarray(keypoints, dtype=np.float32)
        self.cursor = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self.cursor >= len(self.keypoints):
            return []
        kp = self.keypoints[self.cursor]
        self.cursor += 1
        return [Detection(keypoints=kp, score=float(kp[:, 2].mean()))]


BACKENDS = {"yolo": YoloPose, "rtmpose": RtmPose, "replay": ReplayPose}


def build_estimator(name: str = "rtmpose", **kwargs) -> PoseEstimator:
    """Instantiate a backend by name."""
    key = name.lower()
    if key not in BACKENDS:
        raise ValueError(f"unknown pose backend {name!r}; available: {sorted(BACKENDS)}")
    return BACKENDS[key](**kwargs)
