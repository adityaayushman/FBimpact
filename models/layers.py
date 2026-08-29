"""Building blocks for the spatio-temporal graph models.

Two decisions in here are load-bearing for the whole project:

**The temporal convolutions are causal.** A standard ST-GCN centre-pads its
temporal kernel, so the output at frame `t` sees frames up to `t + k//2`. For
action *recognition* that is harmless. For fall *anticipation* it is fatal: the
score at frame `t` would be computed partly from frames after `t`, so a model
could appear to warn before impact while actually having already seen it, and
the headline lead-time number would be an artefact. `CausalTemporalConv` pads
only on the left, which makes the per-frame score at `t` a function of `[0, t]`
alone and makes the offline and online scores identical.

**Joint attention is a first-class output, not a diagnostic.** `JointAttention`
produces the per-joint weights used to pool the graph, and those same weights
are what Stage E reports as the explanation. Because pooling actually uses them,
the deletion/insertion test in Stage F is testing the mechanism the model runs
on rather than a post-hoc story about it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGraphConv(nn.Module):
    """Graph convolution over the joint graph, one weight set per partition."""

    def __init__(self, in_channels: int, out_channels: int, num_subsets: int) -> None:
        super().__init__()
        self.num_subsets = num_subsets
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * num_subsets, kernel_size=1)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: `[N, C_in, T, V]`.
            adjacency: `[K, V, V]`.

        Returns:
            `[N, C_out, T, V]`.
        """
        n, _, t, v = x.shape
        x = self.conv(x).view(n, self.num_subsets, self.out_channels, t, v)
        # Sum each partition's message passing: out[n,c,t,i] = sum_k sum_j x[n,k,c,t,j] A[k,i,j]
        return torch.einsum("nkctv,kwv->nctw", x, adjacency).contiguous()


class CausalTemporalConv(nn.Module):
    """Depth-preserving temporal convolution that never reads the future."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 9,
        stride: int = 1,
        dilation: int = 1,
        causal: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.causal = causal
        self.left_pad = (kernel_size - 1) * dilation
        padding = (0, 0) if causal else ((kernel_size - 1) // 2 * dilation, 0)
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=padding,
            dilation=(dilation, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.causal:
            # Pad only the past, then convolve: output t depends on inputs <= t.
            x = F.pad(x, (0, 0, self.left_pad, 0))
        return self.conv(x)


class STGCNBlock(nn.Module):
    """Spatial graph convolution followed by a temporal convolution, with a residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_subsets: int,
        kernel_size: int = 9,
        dilation: int = 1,
        dropout: float = 0.1,
        residual: bool = True,
        causal: bool = True,
        edge_importance: bool = True,
        num_joints: int = 17,
    ) -> None:
        super().__init__()
        self.gcn = SpatialGraphConv(in_channels, out_channels, num_subsets)
        self.gcn_bn = nn.BatchNorm2d(out_channels)
        self.tcn = CausalTemporalConv(
            out_channels, kernel_size=kernel_size, dilation=dilation, causal=causal
        )
        self.tcn_bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)

        # A learnable per-edge gain, as in ST-GCN. Initialised at 1 so the block
        # starts from the anatomical graph and only departs from it if the data
        # asks it to.
        self.edge_importance = (
            nn.Parameter(torch.ones(num_subsets, num_joints, num_joints))
            if edge_importance
            else None
        )

        if not residual:
            self.residual = None
        elif in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        res = 0.0 if self.residual is None else self.residual(x)
        a = adjacency if self.edge_importance is None else adjacency * self.edge_importance
        out = F.relu(self.gcn_bn(self.gcn(x, a)))
        out = self.dropout(self.tcn_bn(self.tcn(out)))
        return F.relu(out + res)


class JointAttention(nn.Module):
    """Per-frame attention over joints; its weights are the explanation.

    The score for joint `v` at frame `t` is computed from that joint's own
    feature vector at that frame only - no mixing across time - so the weights
    stay causal and stay attributable to a single (frame, joint) pair, which is
    what Stage E needs to report and Stage F needs to perturb.
    """

    def __init__(self, channels: int, hidden: int = 32, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.score = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.Tanh(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: `[N, C, T, V]`.

        Returns:
            `(pooled [N, C, T], weights [N, T, V])` where the weights sum to one
            over the joints of each frame.
        """
        logits = self.score(x).squeeze(1) / self.temperature   # [N, T, V]
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.einsum("nctv,ntv->nct", x, weights)
        return pooled, weights
