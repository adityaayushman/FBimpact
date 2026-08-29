"""GCN-LSTM alternative to ST-GCN (Section 3, objective 1).

Spatial graph convolutions extract a per-frame body-configuration embedding; a
**unidirectional** LSTM carries it forward in time. The LSTM must not be
bidirectional for the same reason the temporal convolutions are causal: a
backward pass would let the hidden state at frame `t` depend on the impact that
has not happened yet, and every lead-time number would be meaningless.

Kept as a second backbone so the "reproduce an established skeleton fall model"
objective can be satisfied with either family, and so the anticipation result is
not an artefact of one architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import Graph
from .layers import JointAttention, SpatialGraphConv


@dataclass
class GCNLSTMConfig:
    in_channels: int = 4
    num_joints: int = 17
    gcn_channels: tuple[int, ...] = (64, 128)
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    graph_strategy: str = "spatial"
    max_hop: int = 1
    edge_importance: bool = True
    attention: bool = True
    attention_hidden: int = 32
    temporal_pool: bool = False
    extra: dict = field(default_factory=dict)


class GCNLSTM(nn.Module):
    """Per-frame graph encoder followed by a causal recurrent temporal model."""

    def __init__(self, config: GCNLSTMConfig | None = None) -> None:
        super().__init__()
        self.config = config or GCNLSTMConfig()
        cfg = self.config

        self.graph = Graph(cfg.graph_strategy, cfg.max_hop)
        self.register_buffer("adjacency", torch.from_numpy(self.graph.A), persistent=False)
        self.edge_importance = (
            nn.Parameter(torch.ones(self.graph.num_subsets, cfg.num_joints, cfg.num_joints))
            if cfg.edge_importance
            else None
        )

        self.input_bn = nn.BatchNorm1d(cfg.in_channels * cfg.num_joints)

        convs, norms = [], []
        in_c = cfg.in_channels
        for out_c in cfg.gcn_channels:
            convs.append(SpatialGraphConv(in_c, out_c, self.graph.num_subsets))
            norms.append(nn.BatchNorm2d(out_c))
            in_c = out_c
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)

        self.joint_attention = (
            JointAttention(in_c, hidden=cfg.attention_hidden) if cfg.attention else None
        )
        self.lstm = nn.LSTM(
            input_size=in_c,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=False,        # never True: see the module docstring
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(cfg.hidden_size, 1)
        self.out_channels = in_c

    def forward(
        self, x: torch.Tensor, return_aux: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Same signature and shapes as `STGCN.forward`."""
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
        x = x.view(nm, c, v, t).permute(0, 1, 3, 2).contiguous()

        a = self.adjacency if self.edge_importance is None else self.adjacency * self.edge_importance
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x, a)))
        features = x                                                # [N*M, C', T, V]

        if self.joint_attention is not None:
            pooled, attention = self.joint_attention(features)
        else:
            pooled = features.mean(dim=-1)
            attention = features.new_full(
                (nm, features.shape[2], features.shape[3]), 1.0 / features.shape[3]
            )

        sequence = pooled.transpose(1, 2)                           # [N*M, T, C']
        if self.config.temporal_pool:
            # `- temporal modelling` ablation: collapse the window to its mean
            # before the recurrence, so the LSTM has a single step to run on and
            # every frame of the window receives the same score. The graph
            # convolutions above are per-frame, so no temporal structure
            # survives this path.
            summary, _ = self.lstm(sequence.mean(dim=1, keepdim=True))
            hidden = summary.expand(-1, sequence.shape[1], -1)
        else:
            hidden, _ = self.lstm(sequence)
        logits = self.head(hidden).squeeze(-1)                      # [N*M, T]

        if m > 1:
            logits = logits.view(n, m, -1).max(dim=1).values
            attention = attention.view(n, m, *attention.shape[1:]).mean(dim=1)
            features = features.view(n, m, *features.shape[1:]).mean(dim=1)

        if not return_aux:
            return logits
        return logits, {"attention": attention, "features": features}

    @torch.no_grad()
    def scores(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(x))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
