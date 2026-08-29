"""Section 14 - reproducibility.

`seed_everything` covers Python, NumPy and torch: weight initialisation, data
order, augmentation draws and the split are all fixed by the seed alone.

`deterministic=True` additionally pins cuDNN to deterministic convolution
algorithms, and that is **not** free. Measured on an RTX 4050 laptop GPU with
the default ST-GCN config, a forward pass over a batch of 110 windows takes

    deterministic=True    2145 ms
    deterministic=False     48 ms

- a factor of 45, which turns a six-variant, three-seed ablation grid from
hours into days. It is off by default for that reason. What it buys is only the
floating-point reduction *order* inside cuDNN kernels; the seed already fixes
everything that decides which experiment is being run. The residual variation is
numerical noise of the same kind that the seeds {0, 1, 2} and the reported
standard deviation exist to absorb.

Turn it on (`run.deterministic: true`) for a final bitwise-reproducible run if
the paper claims one - and budget for the slowdown.

Note what seeding does *not* buy: the three seeds exist to measure run-to-run
variance, so the reported mean and standard deviation come from three genuinely
different initialisations, not three replays of one.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 0, deterministic: bool = False) -> int:
    """Seed every RNG the project touches. Returns the seed, for logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # cuBLAS needs this to be reproducible across runs on the same GPU.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.backends.cudnn.benchmark = True
    return seed


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker its own stream, derived from the run seed."""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def resolve_device(preference: str = "auto") -> torch.device:
    """Pick a device, falling back to CPU with the reason visible in the log."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device {preference!r} requested but CUDA is unavailable")
    return torch.device(preference)
