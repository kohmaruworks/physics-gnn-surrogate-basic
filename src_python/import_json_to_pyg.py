"""
`catlab_directed_graph_v1` JSON（`src_julia/export_graph_json.jl` の `save_catlab_graph_json` 出力）を
torch_geometric.data.Data に変換する。

0-based Contract
----------------
Julia は 1-based、PyG の ``edge_index`` は 0-based。**引き算は Julia のエクスポートでのみ 1 回。**
本モジュールでは ``edges`` を **そのまま** ``torch.long`` の ``(2, |E|)`` にし、
Python 側で ``edge_index -= 1`` は行わない。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data


def graph_json_to_data(path: str | Path, *, dtype_edge_index=torch.int64) -> Data:
    """
    JSON 形式:
      - format: "catlab_directed_graph_v1"
      - num_nodes: int
      - edges: [[src, tgt], ...]（0-based、ローダでは変換しない）
      - x, y: 任意
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        payload: dict[str, Any] = json.load(f)

    if payload.get("format") != "catlab_directed_graph_v1":
        raise ValueError(
            f"unsupported format: {payload.get('format')!r}; "
            "expected 'catlab_directed_graph_v1'"
        )

    num_nodes = int(payload["num_nodes"])
    edges = payload.get("edges") or []

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=dtype_edge_index)
    else:
        edge_index = torch.tensor(edges, dtype=dtype_edge_index).t().contiguous()

    data = Data(edge_index=edge_index, num_nodes=num_nodes)

    if "x" in payload:
        data.x = torch.tensor(payload["x"], dtype=torch.float32)

    if "y" in payload:
        data.y = torch.tensor(payload["y"], dtype=torch.float32)

    return data


# 既存スクリプト・記事向けの別名
catlab_json_to_data = graph_json_to_data


def _default_sample_json() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "graph_data.json"


if __name__ == "__main__":
    import sys

    p = Path(sys.argv[1]) if len(sys.argv) > 1 else _default_sample_json()
    d = graph_json_to_data(p)
    print(d)
    print("edge_index:\n", d.edge_index)
    if d.x is not None:
        print("x:\n", d.x)
