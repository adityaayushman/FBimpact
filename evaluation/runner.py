"""Turning a model into a per-frame score stream, one clip at a time.

The mapping from windowed outputs back to clip frames is the step where an
anticipation result is most easily corrupted, so it is written out explicitly
here rather than left implicit in a training loop.

For every frame `f`, the score used is the one the model produced at the **last
position of the window ending at `f`**. That is the only score a live system
could have at frame `f`. Averaging the several windows that overlap `f`, which
is the natural-looking thing to do offline, would mix in outputs computed from
windows extending past `f` and quietly leak the future into the score - the same
failure the causal convolutions exist to prevent. The leading frames of a clip,
which no full window ends on, are taken from the first window's early positions.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.clips import ClipRecord
from data.datasets import ClipDataset, collate_clips

from .metrics import ClipScores


@torch.no_grad()
def score_clip(
    model: torch.nn.Module,
    item: dict,
    device: torch.device,
    chunk_size: int = 256,
    amp: bool = False,
) -> ClipScores:
    """Score every frame of one clip.

    Args:
        model: a model in eval mode returning `[N, T]` logits.
        item: one item from `ClipDataset`.
        device: where to run.
        chunk_size: windows per forward pass, to bound peak memory on long clips.
        amp: run the forward pass in mixed precision.

    Returns:
        A `ClipScores` aligned frame-for-frame with the original clip.
    """
    windows: torch.Tensor = item["windows"]                    # [N, C, T, V]
    n_windows, _, window_len, _ = windows.shape

    outputs = []
    for start in range(0, n_windows, chunk_size):
        batch = windows[start : start + chunk_size].to(device).unsqueeze(-1)  # [n,C,T,V,1]
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(batch)
        outputs.append(torch.sigmoid(logits.float()).cpu())
    probs = torch.cat(outputs).numpy()                          # [N, T]

    padded_len = n_windows + window_len - 1
    stream = np.empty(padded_len, dtype=np.float32)
    # Leading frames: no window ends on them, so use the first window's own
    # early positions - which are themselves causal, having seen only frames 0..f.
    stream[: window_len - 1] = probs[0, : window_len - 1]
    # Every remaining frame takes the last position of the window ending on it.
    stream[window_len - 1 :] = probs[:, -1]

    n_padded = int(item["n_padded"])
    clip: ClipRecord = item["clip"]
    return ClipScores(
        clip=clip,
        scores=stream[n_padded:],
        labels=np.asarray(item["labels"])[n_padded:],
        tti=np.asarray(item["tti"])[n_padded:],
    )


@torch.no_grad()
def score_dataset(
    model: torch.nn.Module,
    dataset: ClipDataset,
    device: torch.device,
    chunk_size: int = 256,
    amp: bool = False,
    num_workers: int = 0,
) -> list[ClipScores]:
    """Score every clip in a dataset."""
    was_training = model.training
    model.eval().to(device)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_clips,
    )
    results = [score_clip(model, item, device, chunk_size, amp) for item in loader]
    if was_training:
        model.train()
    return results
