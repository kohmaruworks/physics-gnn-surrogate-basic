"""
合成可能性（compositionality）を意図した PyG GNN スケルトン。

ドメイン知識としての GCN 型更新（Kipf & Welling）::

    h_i^{(l+1)} = σ( Σ_{j ∈ Ñ(i)} c_ij W^{(l)} h_j^{(l)} )

異なる物理コンポーネント（ばね・ダンパ等）は、別 ``edge_index`` / 別
``PhysicsGNNLayer`` として合成し、上位で ``HeteroConv`` 等に束ねる拡張を想定。
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class PhysicsGNNLayer(nn.Module):
    """単一関係型の 1 層（差し替え可能なメッセージブロック）。"""

    def __init__(self, in_channels: int, out_channels: int, *, bias: bool = True) -> None:
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels, bias=bias)

    def forward(
        self,
        x,
        edge_index,
        edge_weight=None,
    ):
        return self.conv(x, edge_index, edge_weight)


class PhysicsGNN(nn.Module):
    """同型 ``PhysicsGNNLayer`` を積み重ねたホモジニアス GCN スタック。"""

    def __init__(
        self,
        in_channels: int,
        hidden: int,
        out_channels: int,
        *,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.dropout = float(dropout)
        dims = [in_channels] + [hidden] * (num_layers - 1) + [out_channels]
        self.layers = nn.ModuleList(
            PhysicsGNNLayer(dims[i], dims[i + 1]) for i in range(num_layers)
        )

    def forward(self, x, edge_index, edge_weight=None):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index, edge_weight)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                if self.dropout > 0:
                    h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class TwoLayerGCN(nn.Module):
    """本リポの学習デモ既定: 入力 2 → 隠れ 16 → 出力 2（ノードごと）。"""

    def __init__(self, in_channels: int = 2, hidden_channels: int = 16, out_channels: int = 2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
