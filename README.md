# 🧲 Physics GNN Surrogate (Phase 1) — ばね–質量チェーン × GCN 基礎実装

1 次元の **ばね–質量直列系** を対象に、**グラフニューラルネットワーク（GNN）** によるサロゲートモデル構築の **Phase 1（基礎パイプライン）** をまとめたリポジトリです。

- **Julia** で物理モデルを定式化し、高精度な時間発展（参照解）を生成
- **JSON** で疎結合にデータを受け渡し
- **Python（PyTorch Geometric）** で **GCN** によるノード回帰の最小学習例を実行

開発者がリポジトリを開いただけで「何を・どの順で・どのツールで動かすか」が追えることを目指した構成です。

---

## ✨ 概要

| 項目 | 内容 |
|------|------|
| **物理** | 質点 \(n\) 個の 1D 直列チェーン、隣接質点間はフックの法則（自由端） |
| **入力 `x`** | 各ノードの初期 **位置・速度**（\(t=0\)） |
| **教師 `y`** | `DifferentialEquations.jl` で積分した **\(t=t_1\)** における位置・速度 |
| **グラフ** | バネで結ばれた隣接ペアを **双方向の有向辺** で表現（相互作用の対称性） |
| **学習** | 2 層 **GCN** によるノードごとの **MSE 回帰**（ダミー学習デモ） |

---

## 🏗 アーキテクチャの特徴

### 1. Julia による参照解（グラウンドトゥルース）生成

- **`DifferentialEquations.jl`** … 連立 1 階化した状態 \(z=[u_1,\ldots,u_n,v_1,\ldots,v_n]\) の ODE を **高精密**（`Tsit5` + 厳しめの許容誤差）で積分し、時刻 \(t_1\) の状態を `y` として取得します。
- **`Catlab.jl`（`Catlab.Graphs`）** … 系の **トポロジー（相互作用の構造）** を有向グラフとして表し、そのまま JSON スキーマに載せやすい形に整えます。

メインのデータ生成フローは `spring_mass_chain_export.jl` に集約されています。

### 2. JSON を介したデータ連携（Julia ↔ Python）

- フォーマット識別子: **`catlab_directed_graph_v1`**
- フィールド例: `num_nodes`, `edges`, `x`（ノード特徴）, `y`（ターゲット）

#### 🔢 超重要: 頂点インデックスは **1-based（Julia）→ 0-based（JSON）** に変換済み

Julia / Catlab 側の頂点 ID は **1 始まり** ですが、JSON に書き出す段階で **各辺の `src`, `tgt` から 1 を引いて 0 始まり** にしています。  
そのため Python 側では **`torch_geometric.data.Data` の `edge_index` とそのまま整合**し、追加のインデックス補正なしで読み込めます。

該当実装（Julia）:

```14:18:export_catlab_graph_json.jl
function catlab_graph_edges_0based(g)
    pairs = Vector{Vector{Int}}()
    for e in edges(g)
        push!(pairs, [src(g, e) - 1, tgt(g, e) - 1])
    end
```

### 3. Python（PyTorch Geometric）による GNN 学習

- `import_catlab_json_to_pyg.py` … JSON → `Data` 変換
- `train_spring_mass_gcn.py` … 2 層 **GCN** で `x` から `y`（またはフォールバック）への回帰と **MSE** 最適化

---

## 📁 ファイル構成

| ファイル | 役割 |
|----------|------|
| **`Project.toml`** | Julia 依存関係（`Catlab`, `DifferentialEquations`, `JSON3`）の宣言 |
| **`Manifest.toml`** | Julia のパッケージ解決結果（再現性のためコミット推奨） |
| **`export_catlab_graph_json.jl`** | Catlab の有向グラフを PyG 向け JSON にシリアライズ。辺の **0-based 化**、`x` / `y` のオプション付与 |
| **`spring_mass_chain_export.jl`** | 5 質点チェーンの例: グラフ構築 → ODE 積分 → `spring_mass_chain_5.json` 等へエクスポート |
| **`spring_mass_chain_5.json`** | 上記パイプラインの **サンプル出力**（`format`, `num_nodes`, `edges`, `x`, `y`） |
| **`import_catlab_json_to_pyg.py`** | `catlab_directed_graph_v1` を読み、`edge_index` / `x` / `y` を持つ `Data` を構築 |
| **`train_spring_mass_gcn.py`** | サンプル JSON を読み込み、2 層 GCN の学習ループ（デモ） |
| **`requirements.txt`** | Python 側の依存関係（`torch`, `torch-geometric`） |

> `export_catlab_graph_json.jl` を **単体のスクリプト**として実行すると、小さな例グラフの `graph_from_catlab.json` を生成するブロックがあります（`include` 用途とは別の動作確認用）。

---

## 🚀 環境構築と実行手順

### Julia 側

1. **依存関係のインストール**（プロジェクトルートで）:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

2. **データ生成**（デフォルトで `spring_mass_chain_5.json` を出力）:

```bash
julia --project=. spring_mass_chain_export.jl
```

`spring_mass_chain_export.jl` 先頭のコメントにも、質点数・物理定数・出力ファイル名などを `main(; ...)` で変えられる旨が記載されています。

### Python 側

1. **仮想環境の作成（推奨）** 後、依存関係のインストール:

```bash
pip install -r requirements.txt
```

2. **学習スクリプトの実行**（先に Julia で JSON を生成しておくこと）:

```bash
python train_spring_mass_gcn.py
```

#### ⚠️ PyTorch Geometric について（軽い注意）

- **CPU のみ**の環境では、上記の `pip install` で十分なことが多いです。
- **GPU（CUDA）** を使う場合は、**PyTorch 本体を CUDA 対応ビルドで入れたうえで**、PyTorch / CUDA の組み合わせに対応した **PyG の wheel** が必要になることがあります。環境に合わせて [PyTorch Geometric のインストール案内](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) を参照してください。
- トラブル時は、まず `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` で PyTorch 側を確認し、その後 `torch_geometric` のインポートを試すと切り分けがしやすいです。

---

## 📚 関連情報

本リポジトリは、Zennの連載記事のコンパニオン（補助実装）として用意されています。背景・理論・発展的なトピックは以下の連載本編を参照してください。

* **第1回:** [【Julia/Python】サロゲートモデル構築(基礎編)1：全体アーキテクチャとデータ連携の設計思想](https://zenn.dev/kohmaruworks/articles/phase1-architecture)
* **第2回:** （執筆中）
* **第3回:** （執筆中）
