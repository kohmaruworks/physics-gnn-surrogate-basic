"""
data/spring_mass_chain_5.json を読み込み、2 層 GCN でノード回帰（MSE）のダミー学習を行う。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent
if str(_ROOT / "src_python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src_python"))

from import_json_to_pyg import graph_json_to_data
from models.physics_gnn_base import TwoLayerGCN


def main() -> None:
    json_path = _ROOT / "data" / "spring_mass_chain_5.json"
    if not json_path.is_file():
        raise FileNotFoundError(
            f"見つかりません: {json_path}\n"
            "先に `julia --project=. spring_mass_chain_export.jl` を実行してください。"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    data = graph_json_to_data(json_path)
    if data.x is None:
        raise ValueError("JSON にノード特徴 x がありません。")
    data = data.to(device)

    if getattr(data, "y", None) is not None:
        target = data.y
    else:
        torch.manual_seed(7)
        # JSON に y が無い場合のフォールバック（ダミー教師）
        target = data.x + 0.05 * torch.randn_like(data.x)

    model = TwoLayerGCN(in_channels=2, hidden_channels=16, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.MSELoss()

    n_epochs = 100
    print("epoch\tloss")
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(data.x, data.edge_index)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print(f"{epoch}\t{loss.item():.6f}")


if __name__ == "__main__":
    main()
