# 🧲 1D Spring–Mass Chain Surrogate via Julia & PyTorch Geometric

**Phase 1 reference pipeline** — category-theoretic graph semantics (**Catlab.jl**), high-fidelity physics (**DifferentialEquations.jl**), and **PyTorch Geometric** GNN training, bridged by a versioned JSON intermediate representation.

[🇺🇸 English](#english) | [🇯🇵 日本語](#japanese)

---

<a id="english"></a>

## 🇺🇸 English

### Title & overview

This repository delivers a **research-grade baseline (Phase 1)** for **graph surrogate modeling** of a **one-dimensional spring–mass chain** with free ends. The scientific stack is deliberately split across languages for **modularity and auditability**:

| Layer | Role |
|--------|------|
| **Physics & topology (Julia)** | ODE ground truth + **compositional graph** encoding of interactions |
| **IR (JSON)** | Loosely coupled, tool-agnostic **intermediate representation** |
| **Learning (Python)** | **GCN** node regression on `torch_geometric.data.Data` |

Rather than a toy “hello GNN” script, the design foregrounds **applied category theory (ACT)** as an organizing principle: interaction structure is not an ad-hoc adjacency matrix, but a **Catlab `Graph`** whose semantics align with categorical models of networks. That graph is serialized once, then consumed by standard geometric deep learning tooling.

**Problem shape (default demo):** node features **`x`** = initial position & velocity at \(t{=}0\); supervised targets **`y`** = position & velocity at \(t{=}t_1\) from a high-accuracy ODE solve. Springs between neighbors are represented as **bidirected edges** to reflect symmetric coupling.

---

### Architecture

| Stage | Technology | Responsibility |
|--------|------------|----------------|
| **Ground truth** | **DifferentialEquations.jl** | Time integration of the first-order state \(z=[u; v]\) with tight tolerances (`Tsit5`, `abstol`/`reltol` \(\sim 10^{-10}\)) |
| **Topology & export** | **Catlab.jl** (`Catlab.Graphs`) | Directed graph IR of the mechanical network; JSON export with explicit schema |
| **Surrogate** | **PyTorch Geometric** | Two-layer **GCN** + MSE training (`train_spring_mass_gcn.py`) |

**JSON schema:** `format: "catlab_directed_graph_v1"` with `num_nodes`, `edges`, optional `x` / `y`.

---

### ⚠️ Crucial technical detail: 1-based → 0-based index contract

Julia and Catlab use **1-based vertex IDs**. PyTorch Geometric expects **`edge_index` in 0-based node indices**. **This repository encodes the contract at export time:** every directed edge \((s,t)\) is written to JSON as \((s{-}1,\, t{-}1)\). Downstream Python code therefore **requires no offset hacks** — the JSON is already aligned with `Data.edge_index`.

Implementation (`export_catlab_graph_json.jl`):

```julia
function catlab_graph_edges_0based(g)
    pairs = Vector{Vector{Int}}()
    for e in edges(g)
        push!(pairs, [src(g, e) - 1, tgt(g, e) - 1])
    end
    pairs
end
```

**Design rationale:** treat **index semantics as part of the IR specification**, not an implicit convention scattered across loaders. That choice reduces cross-language defects and makes the pipeline easier to validate in regulated or collaborative R&D settings.

---

### File structure

| Path | Purpose |
|------|---------|
| **`Project.toml`**, **`Manifest.toml`** | Julia environment declaration & reproducible resolution |
| **`export_catlab_graph_json.jl`** | Catlab `Graph` → JSON (`catlab_directed_graph_v1`); **0-based edges**; optional `x`, `y` |
| **`spring_mass_chain_export.jl`** | End-to-end demo: build chain graph → integrate ODE → write e.g. `spring_mass_chain_5.json` |
| **`spring_mass_chain_5.json`** | Committed sample artifact (graph + `x` / `y`) |
| **`import_catlab_json_to_pyg.py`** | JSON → `torch_geometric.data.Data` |
| **`train_spring_mass_gcn.py`** | GCN training loop (MSE node regression) |
| **`requirements.txt`** | Python dependencies (`torch`, `torch-geometric`) |

> Running **`export_catlab_graph_json.jl` as `PROGRAM_FILE`** also emits a tiny `graph_from_catlab.json` for smoke tests — distinct from the `include` path used by the spring–mass exporter.

---

### Quick start / usage

**Julia — instantiate & generate data**

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. spring_mass_chain_export.jl
```

**Python — install & train**

```bash
pip install -r requirements.txt
python train_spring_mass_gcn.py
```

**PyTorch Geometric:** CPU-only setups often work with `pip install -r requirements.txt`. For **CUDA**, install a **matching** PyTorch build first, then follow the [official PyG installation matrix](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for wheels compatible with your CUDA/toolchain.

---

### Related

Companion implementation for a **Zenn article series** (Japanese). For narrative context, theory, and roadmap, see the series (links in the Japanese section below).

---

<a id="japanese"></a>

## 🇯🇵 日本語

### タイトルと概要

本リポジトリは、**1 次元ばね–質量直列系（自由端）** を対象とした、**Julia と PyTorch Geometric を組み合わせたグラフサロゲートの Phase 1 基盤**です。単なる入門用スクリプトの寄せ集めではなく、**応用圏論（ACT）の観点から相互作用をグラフとして記述する（Catlab.jl）** と、**高精度な参照解（DifferentialEquations.jl）**、**標準的な幾何学習スタック（PyG）** を、**版付き JSON 中間表現**で疎結合に接続する **R&D 向けの最小パイプライン** を提示します。

| 層 | 役割 |
|----|------|
| **物理・トポロジ（Julia）** | ODE によるグラウンドトゥルース + **合成可能なグラフ IR** |
| **中間表現（JSON）** | ツール非依存の **疎結合データ契約** |
| **学習（Python）** | **GCN** によるノード回帰（MSE） |

**デフォルトデモ:** ノード特徴 **`x`** = \(t{=}0\) の位置・速度、教師 **`y`** = 高精密積分による \(t{=}t_1\) の位置・速度。隣接質点間のバネは **双方向有向辺** で表し、相互作用の対称性を明示します。

---

### アーキテクチャ

| 段階 | 技術 | 担当 |
|------|------|------|
| **参照解** | **DifferentialEquations.jl** | 1 階化状態 \(z=[u; v]\) の時間発展（厳しめ許容誤差） |
| **トポロジ・書き出し** | **Catlab.jl**（`Catlab.Graphs`） | 力学ネットワークの有向グラフ化と JSON シリアライズ |
| **サロゲート** | **PyTorch Geometric** | 2 層 **GCN** + MSE（`train_spring_mass_gcn.py`） |

**JSON:** `format: "catlab_directed_graph_v1"`、`num_nodes`、`edges`、任意の `x` / `y`。

---

### ⚠️ 技術上の要諦: 1-based → 0-based の契約をエクスポートで固定する

Julia / Catlab の頂点 ID は **1 始まり**ですが、PyTorch Geometric の **`edge_index` は 0 始まり**です。本プロジェクトでは **JSON 書き出し時点で** 各辺の端点から **1 を減算**し、Python 側が **追加のインデックス補正なし**で `Data` を構築できるようにしています。これは実装の都合ではなく、**中間表現の仕様としてインデックス意味を固定する**設計です（共同研究・受託開発・再現性の観点で有利）。

実装（`export_catlab_graph_json.jl`）:

```julia
function catlab_graph_edges_0based(g)
    pairs = Vector{Vector{Int}}()
    for e in edges(g)
        push!(pairs, [src(g, e) - 1, tgt(g, e) - 1])
    end
    pairs
end
```

---

### ファイル構成

| パス | 役割 |
|------|------|
| **`Project.toml`**, **`Manifest.toml`** | Julia 環境の宣言とロック |
| **`export_catlab_graph_json.jl`** | Catlab `Graph` → JSON（**辺は 0-based**）、`x` / `y` 任意 |
| **`spring_mass_chain_export.jl`** | チェーングラフ構築 → ODE 積分 → `spring_mass_chain_5.json` 等へ出力 |
| **`spring_mass_chain_5.json`** | サンプル成果物（`format`, `num_nodes`, `edges`, `x`, `y`） |
| **`import_catlab_json_to_pyg.py`** | JSON → `torch_geometric.data.Data` |
| **`train_spring_mass_gcn.py`** | GCN 学習ループ（デモ） |
| **`requirements.txt`** | Python 依存（`torch`, `torch-geometric`） |

> `export_catlab_graph_json.jl` を **単体スクリプト**として実行すると、小グラフの `graph_from_catlab.json` を吐く検証用ブロックがあります（`spring_mass_chain_export.jl` からの `include` 用途とは別）。

---

### クイックスタート / 使い方

**Julia — 環境構築とデータ生成**

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. spring_mass_chain_export.jl
```

**Python — インストールと学習**

```bash
pip install -r requirements.txt
python train_spring_mass_gcn.py
```

**PyTorch Geometric:** CPU のみなら上記で足りることが多いです。**GPU（CUDA）** 利用時は、先に **CUDA 対応の PyTorch** を入れ、その組み合わせに合わせて [PyG 公式のインストール手順](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) で wheel を選んでください。

---

### 関連情報（Zenn 連載）

本リポジトリは **Zenn 連載のコンパニオン実装**です。全体設計・理論・発展編は記事側を参照してください。

- **第1回:** [【Julia/Python】サロゲートモデル構築(基礎編)1：全体アーキテクチャとデータ連携の設計思想](https://zenn.dev/kohmaruworks/articles/phase1-architecture)
- **第2回:** （執筆中）
- **第3回:** （執筆中）
