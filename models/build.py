"""Model factory driven by the YAML config."""

from __future__ import annotations

import torch.nn as nn

from .gcn_lstm import GCNLSTM, GCNLSTMConfig
from .stgcn import STGCN, STGCNConfig

REGISTRY = {
    "stgcn": (STGCN, STGCNConfig),
    "gcn_lstm": (GCNLSTM, GCNLSTMConfig),
}


def build_model(cfg: dict, in_channels: int | None = None, num_joints: int = 17) -> nn.Module:
    """Instantiate a backbone from a config dict.

    Args:
        cfg: the `model` block of the run config. `name` selects the backbone;
            every other key is passed to that backbone's config dataclass.
        in_channels: overrides `cfg["in_channels"]`, so the feature pipeline
            stays the authority on channel count (2 without velocity, 4 with).
        num_joints: skeleton size, from the pose estimator's layout.

    Returns:
        An un-trained model on the CPU.
    """
    cfg = dict(cfg or {})
    name = str(cfg.pop("name", "stgcn")).lower()
    if name not in REGISTRY:
        raise ValueError(f"unknown model {name!r}; available: {sorted(REGISTRY)}")

    model_cls, config_cls = REGISTRY[name]
    if in_channels is not None:
        cfg["in_channels"] = int(in_channels)
    cfg["num_joints"] = int(num_joints)

    fields = {f.name for f in config_cls.__dataclass_fields__.values()}

    # A config that inherits from `default.yaml` carries the other backbone's
    # keys. Setting them to null is the documented way to switch them off, so a
    # null value for a key this backbone does not have is dropped...
    cfg = {k: v for k, v in cfg.items() if k in fields or v is not None}

    # ...but a key with a real value that this backbone does not understand is an
    # error. Silently dropping a misspelt hyperparameter is how an ablation ends
    # up accidentally identical to the model it is meant to differ from.
    unknown = set(cfg) - fields
    if unknown:
        raise ValueError(
            f"unknown {name} options: {sorted(unknown)}; valid keys are {sorted(fields)}"
        )

    # Tuple-valued fields arrive from YAML as lists.
    for key in ("blocks", "gcn_channels"):
        if key in cfg and cfg[key] is not None:
            cfg[key] = tuple(tuple(v) if isinstance(v, list) else v for v in cfg[key])

    return model_cls(config_cls(**cfg))
