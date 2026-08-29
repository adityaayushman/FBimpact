"""Stage D - the per-frame anticipation model.

`STGCN` is the shared backbone. With `pre_impact` training it is the proposed
model; with a plain classification loss and the same architecture it is the
reproduced baseline, so the comparison in Section 15 isolates the objective
rather than confounding it with capacity.

The head is the part that differs from a stock ST-GCN. Stock ST-GCN average-pools
over time and emits one label per clip; this emits a **logit per frame**, because
lead time is only defined if the model has an opinion at every frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .graph import Graph
from .layers import JointAttention, STGCNBlock


@dataclass
class STGCNConfig:
    """Architecture hyperparameters."""

    in_channels: int = 4
    """4 for (x, y, vx, vy); 2 for the `- velocity features` ablation."""

    num_joints: int = 17
    base_channels: int = 64
    blocks: tuple[tuple[int, int], ...] = ((64, 1), (64, 1), (128, 1), (128, 1))
    """`(channels, temporal dilation)` per block. Dilation grows the receptive
    field without striding, which would misalign the per-frame output.

    Sized so `receptive_field` (25 frames here) fits inside the default 30-frame
    window. A deeper or more dilated stack is not free accuracy: past the window
    length the extra kernel taps only ever see left-padding, so they cost
    parameters and latency and add nothing. Check `receptive_field` against
    `features.window` whenever either is changed - `train.py` warns if they
    diverge."""

    kernel_size: int = 7
    dropout: float = 0.1
    graph_strategy: str = "spatial"
    max_hop: int = 1
    causal: bool = True
    edge_importance: bool = True

    attention: bool = True
    """False for the `- grounding head` ablation (mean-pools joints instead)."""

    attention_hidden: int = 32
    temporal_pool: bool = False
    """True for the `- temporal modelling` ablation. It does two things
    together, and both are needed for the ablation to mean anything: the
    temporal kernels collapse to size 1, so no block can look across frames, and
    the window is mean-pooled into a single score broadcast to every frame.
    Setting only the second would leave the temporal convolutions intact and
    ablate almost nothing."""

    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.temporal_pool:
            self.kernel_size = 1
            self.blocks = tuple((c, 1) for c, _ in self.blocks)

    @property
    def receptive_field(self) -> int:
        """Frames of history the last output depends on, in input frames."""
        return 1 + sum((self.kernel_size - 1) * d for _, d in self.blocks)


class STGCN(nn.Module):
    """Spatio-temporal graph network emitting one fall-imminence logit per frame."""

    def __init__(self, config: STGCNConfig | None = None) -> None:
        super().__init__()
        self.config = config or STGCNConfig()
        cfg = self.config

        self.graph = Graph(cfg.graph_strategy, cfg.max_hop)
        if self.graph.num_joints != cfg.num_joints:
            raise ValueError(
                f"graph has {self.graph.num_joints} joints, config says {cfg.num_joints}"
            )
        self.register_buffer("adjacency", torch.from_numpy(self.graph.A), persistent=False)

        # Normalising across (C, V) rather than C alone matches ST-GCN and keeps
        # a single badly-scaled joint from dominating the first layer.
        self.input_bn = nn.BatchNorm1d(cfg.in_channels * cfg.num_joints)

        blocks = []
        in_c = cfg.in_channels
        for i, (out_c, dilation) in enumerate(cfg.blocks):
            blocks.append(
                STGCNBlock(
                    in_channels=in_c,
                    out_channels=out_c,
                    num_subsets=self.graph.num_subsets,
                    kernel_size=cfg.kernel_size,
                    dilation=dilation,
                    dropout=cfg.dropout,
                    residual=i > 0,
                    causal=cfg.causal,
                    edge_importance=cfg.edge_importance,
                    num_joints=cfg.num_joints,
                )
            )
            in_c = out_c
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = in_c

        self.joint_attention = (
            JointAttention(in_c, hidden=cfg.attention_hidden) if cfg.attention else None
        )
        self.head = nn.Conv1d(in_c, 1, kernel_size=1)

    # -- forward ---------------------------------------------------------------

    def forward(
        self, x: torch.Tensor, return_aux: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Args:
            x: `[N, C, T, V]` or `[N, C, T, V, M]` with `M` people.
            return_aux: also return joint attention and the pre-head features,
                which Stage E consumes.

        Returns:
            `logits [N, T]`, or `(logits, aux)` when `return_aux`.
        """
        if x.dim() == 5:
            n, c, t, v, m = x.shape
            x = x.permute(0, 4, 1, 2, 3).reshape(n * m, c, t, v)
        elif x.dim() == 4:
            n, c, t, v = x.shape
            m = 1
        else:
            raise ValueError(f"expected a 4D or 5D input, got shape {tuple(x.shape)}")

        nm = x.shape[0]
        x = x.permute(0, 1, 3, 2).reshape(nm, c * v, t)
        x = self.input_bn(x)
        x = x.view(nm, c, v, t).permute(0, 1, 3, 2).contiguous()   # [N*M, C, T, V]

        for block in self.blocks:
            x = block(x, self.adjacency)
        features = x                                                # [N*M, C', T, V]

        if self.joint_attention is not None:
            pooled, attention = self.joint_attention(features)      # [N*M, C', T], [N*M, T, V]
        else:
            pooled = features.mean(dim=-1)
            attention = features.new_full(
                (nm, features.shape[2], features.shape[3]), 1.0 / features.shape[3]
            )

        if self.config.temporal_pool:
            # The `- temporal modelling` ablation: collapse the window to its
            # mean and give every frame the same score.
            pooled = pooled.mean(dim=-1, keepdim=True).expand_as(pooled)

        logits = self.head(pooled).squeeze(1)                       # [N*M, T]

        if m > 1:
            # Several people in frame: the clip is imminent if anyone is falling.
            logits = logits.view(n, m, -1).max(dim=1).values
            attention = attention.view(n, m, *attention.shape[1:]).mean(dim=1)
            features = features.view(n, m, *features.shape[1:]).mean(dim=1)

        if not return_aux:
            return logits
        return logits, {"attention": attention, "features": features}

    # -- convenience -----------------------------------------------------------

    @torch.no_grad()
    def scores(self, x: torch.Tensor) -> torch.Tensor:
        """Per-frame imminence probabilities `p_t`."""
        return torch.sigmoid(self(x))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
