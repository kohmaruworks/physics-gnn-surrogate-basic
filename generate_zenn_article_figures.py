"""
phase1-python-training.md 用の図を生成する。
- Julia（Catlab）側の学習データの有向グラフ（1-based 表記）
- Python（PyG）へ渡すときの同じグラフ（0-based 表記）
- train_spring_mass_gcn.py と同条件の GCN 学習損失（教師 y は ODE 由来）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
_SRC = ROOT / "src_python"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.physics_gnn_base import TwoLayerGCN

ZENN_IMAGES = Path(__file__).resolve().parents[1] / "zenn-articles" / "images"


def load_edges_and_n(json_path: Path) -> tuple[list[tuple[int, int]], int]:
    with json_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    num_nodes = int(payload["num_nodes"])
    edges = [tuple(pair) for pair in (payload.get("edges") or [])]
    return edges, num_nodes


def plot_julia_graph(edges: list[tuple[int, int]], num_nodes: int, out_path: Path) -> None:
    g = nx.DiGraph()
    for i in range(1, num_nodes + 1):
        g.add_node(i)
    for s, t in edges:
        g.add_edge(s + 1, t + 1)
    pos = {i: (float(i), 0.0) for i in range(1, num_nodes + 1)}
    labels = {i: f"{i}\nx[i,:], y[i,:]" for i in range(1, num_nodes + 1)}
    fig, ax = plt.subplots(figsize=(10.0, 5.0), dpi=150)
    nx.draw_networkx_nodes(
        g,
        pos,
        node_color="#e3f2fd",
        edgecolors="#1565c0",
        linewidths=2.0,
        node_size=2200,
        ax=ax,
    )
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=9, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(
        g,
        pos,
        edge_color="#37474f",
        arrows=True,
        arrowsize=18,
        connectionstyle="arc3,rad=0.12",
        min_source_margin=20,
        min_target_margin=20,
        ax=ax,
    )
    ax.set_title("Julia (Catlab): training graph, 1-based vertex IDs", fontsize=12)
    ax.set_xlim(0.3, float(num_nodes) + 0.7)
    ax.set_ylim(-1.35, 0.95)
    ax.axis("off")

    kw = (
        "Keywords (payload):\n"
        "  vertex i  ->  JSON rows x[i,:] (u,v @ t=0), y[i,:] (u,v @ t=t1, ODE)\n"
        "  directed edges  ->  spring-chain topology (who couples whom)\n"
        "  export  ->  save_catlab_graph_json (src_julia/export_graph_json.jl): edges -> 0-based in JSON"
    )
    ax.text(
        0.02,
        0.02,
        kw,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fafafa", edgecolor="#90a4ae", linewidth=1),
        family="monospace",
    )
    ax.annotate(
        "edge = message route\n(GCN) / spring neighbor",
        xy=(2.5, 0.12),
        fontsize=8,
        ha="center",
        color="#37474f",
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_pyg_graph(edges: list[tuple[int, int]], num_nodes: int, out_path: Path) -> None:
    g = nx.DiGraph()
    for i in range(num_nodes):
        g.add_node(i)
    for s, t in edges:
        g.add_edge(s, t)
    pos = {i: (float(i), 0.0) for i in range(num_nodes)}
    labels = {i: f"{i}\nData.x[i]\nData.y[i]" for i in range(num_nodes)}
    fig, ax = plt.subplots(figsize=(10.0, 5.2), dpi=150)
    nx.draw_networkx_nodes(
        g,
        pos,
        node_color="#fff3e0",
        edgecolors="#ef6c00",
        linewidths=2.0,
        node_size=2400,
        ax=ax,
    )
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=8, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(
        g,
        pos,
        edge_color="#37474f",
        arrows=True,
        arrowsize=18,
        connectionstyle="arc3,rad=0.12",
        min_source_margin=22,
        min_target_margin=22,
        ax=ax,
    )
    ax.set_title("Python (PyG) Data: same graph, 0-based edge_index", fontsize=12)
    ax.set_xlim(-0.45, float(num_nodes - 1) + 0.45)
    ax.set_ylim(-1.45, 1.05)
    ax.axis("off")

    kw = (
        "Keywords (payload):\n"
        "  Data.x, Data.y  <-  JSON x, y  (float32, shape [N,2])\n"
        "  Data.edge_index  <-  JSON edges  (int64, shape [2,|E|])  [0-based Contract]\n"
        "  forward: GCNConv(x, edge_index)  vs  teacher  Data.y"
    )
    ax.text(
        0.02,
        0.02,
        kw,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fafafa", edgecolor="#ffb74d", linewidth=1),
        family="monospace",
    )
    ax.annotate(
        "edge_index\n(topology only)",
        xy=(2.0, 0.22),
        fontsize=8,
        ha="center",
        color="#37474f",
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_training_loss(json_path: Path, out_path: Path) -> None:
    from import_json_to_pyg import graph_json_to_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = graph_json_to_data(json_path)
    if data.x is None or data.y is None:
        raise ValueError("JSON に x, y が必要です。")
    data = data.to(device)
    target = data.y

    model = TwoLayerGCN(in_channels=2, hidden_channels=16, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.MSELoss()

    n_epochs = 100
    losses: list[float] = []
    for _ in range(n_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(data.x, data.edge_index)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=150)
    ax.plot(range(1, n_epochs + 1), losses, color="#c62828", linewidth=1.8)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("MSE loss", fontsize=11)
    ax.set_title(
        "GCN training loss (target y from Julia ODE integration at t1)",
        fontsize=11,
    )
    ax.grid(True, alpha=0.35, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    json_path = ROOT / "data" / "spring_mass_chain_5.json"
    if not json_path.is_file():
        raise FileNotFoundError(json_path)

    ZENN_IMAGES.mkdir(parents=True, exist_ok=True)

    edges, num_nodes = load_edges_and_n(json_path)
    plot_julia_graph(edges, num_nodes, ZENN_IMAGES / "phase1-julia-training-graph.png")
    plot_pyg_graph(edges, num_nodes, ZENN_IMAGES / "phase1-pyg-training-graph.png")
    run_training_loss(json_path, ZENN_IMAGES / "phase1-gcn-training-loss.png")

    print("wrote:")
    for name in (
        "phase1-julia-training-graph.png",
        "phase1-pyg-training-graph.png",
        "phase1-gcn-training-loss.png",
    ):
        p = ZENN_IMAGES / name
        print(" ", p, p.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
