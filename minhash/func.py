from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.typing as npt
import seaborn as sns
from datetime import datetime
from graphviz import Digraph
from sklearn.metrics import roc_auc_score
from minlsh import MinHashLSHiForest, Node
import mmh3
from typing import List, Set
from collections import defaultdict
import math
import os
import tempfile
from matplotlib.colors import to_hex


def auc_by_attack(
    y_cat: pd.Series, y_label: pd.Series, scores: npt.NDArray, name: str
) -> None:
    results_df = pd.DataFrame(
        {
            "attack_cat": y_cat,
            "label": y_label,
            "score": scores,
        }
    )

    normal_df = results_df[results_df["label"] == 0]
    attack_df = results_df[results_df["label"] == 1]

    auc_scores_by_cat = {}
    unique_attack_cats = sorted(attack_df["attack_cat"].unique())

    for cat in unique_attack_cats:
        current_attack_df = attack_df[attack_df["attack_cat"] == cat]
        eval_df = pd.concat([normal_df, current_attack_df], axis=0)
        auc = roc_auc_score(eval_df["label"], eval_df["score"])
        auc_scores_by_cat[cat] = auc

    plot_df = pd.DataFrame(
        list(auc_scores_by_cat.items()), columns=["Attack Category", "AUC Score"]
    )
    plot_df = plot_df.sort_values("AUC Score", ascending=False)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(18, 10))
    sns.barplot(
        data=plot_df, x="Attack Category", y="AUC Score", color="steelblue", ax=ax
    )
    ax.set_title(f"LSHiForest Performance by Attack Category on {name}", fontsize=20)
    ax.set_xlabel("Attack Category", fontsize=14)
    ax.set_ylabel("AUC Score", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", labelsize=12)
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
    print("AUC Scores (Each Attack Category vs. Normal Traffic):")
    print(plot_df.reset_index(drop=True))
    plt.savefig(
        f"results/{datetime.today().strftime('%y%m%d')}{name}_auc_by_category.png"
    )


def plot_anomaly_score(
    y_cat: pd.Series, y_label: pd.Series, scores: npt.NDArray, name: str
) -> None:
    """
    攻撃手法ごとに異常スコアをbarplotする
    """
    results_df = pd.DataFrame({"attack_cat": y_cat, "label": y_label, "score": scores})

    normal_scores = results_df[results_df["label"] == 0]
    attack_scores = results_df[results_df["label"] == 1]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(18, 10))
    order = (
        attack_scores.groupby("attack_cat")["score"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    sns.barplot(data=attack_scores, x="attack_cat", y="score", ax=ax, order=order)
    ax.axhline(
        normal_scores["score"].mean(),
        color="red",
        linestyle="--",
        label=f"Normal Average Score: {normal_scores['score'].mean():.2f}",
    )

    ax.set_title(f"Anomaly Score by Attack Category on {name}", fontsize=20)
    ax.set_xlabel("Attack Category", fontsize=14)
    ax.set_ylabel("Anomaly Score", fontsize=14)
    ax.tick_params(axis="x", labelsize=12)
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    ax.tick_params(axis="y", labelsize=12)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
    plt.savefig(
        f"results/{datetime.today().strftime('%y%m%d')}_{name}_anomaly_scores.png"
    )


def plot_tree(model: MinHashLSHiForest, name: str, tree_index: int) -> None:
    """
    学習済みモデル内の指定されたインデックスの木をmatplotlibでプロットする。
    - hash_index ごとに階層を分けて表示
    - ノードは円形で、サンプル数に応じて大きさと色が変化
    - エッジのハッシュ値は非表示
    """
    if (
        not hasattr(model, "forest_")
        or not model.forest_
        or tree_index >= len(model.forest_)
    ):
        print(
            f"エラー: インデックス {tree_index} の木は存在しません。モデルが学習済みか確認してください。"
        )
        return

    tree_info = model.forest_[tree_index]
    tree_root = tree_info["root"]

    if not tree_root:
        print(f"エラー: インデックス {tree_index} の木にルートノードがありません。")
        return

    dot = Digraph(comment=f"MinHashLSHiForest Tree {tree_index}")
    dot.attr("node", shape="circle", style="filled")
    dot.attr(rankdir="TB", splines="line")

    cmap = plt.cm.get_cmap("Blues")
    max_log_size = math.log1p(tree_root.size)
    MIN_SIZE, MAX_SIZE = 0.5, 2.0

    nodes_by_level = defaultdict(list)

    def register_nodes(node: Node) -> None:
        """全ノードを探索し、階層ごとに登録する"""
        if node:
            nodes_by_level[node.hash_index].append(node)
            for child in node.children.values():
                register_nodes(child)

    register_nodes(tree_root)

    for level in sorted(nodes_by_level.keys()):
        with dot.subgraph() as s:
            s.attr(rank="same")
            for node in nodes_by_level[level]:
                node_id = str(id(node))
                log_size = math.log1p(node.size)
                scale_factor = log_size / max_log_size if max_log_size > 0 else 0
                size = MIN_SIZE + (MAX_SIZE - MIN_SIZE) * scale_factor
                color_val = 0.2 + 0.6 * scale_factor
                color_hex = to_hex(cmap(color_val))

                s.node(
                    node_id,
                    label=f"S: {node.size}\n H: {node.hash_index}",
                    width=str(size),
                    height=str(size),
                    fixedsize="true",
                    fillcolor=color_hex,
                )
                for parent_level in range(level):
                    for parent_node in nodes_by_level.get(parent_level, []):
                        for child_node in parent_node.children.values():
                            if id(child_node) == id(node):
                                dot.edge(str(id(parent_node)), node_id, label="")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        temp_filename = tmpfile.name
        dot.render(os.path.splitext(temp_filename)[0], format="png", cleanup=True)

    try:
        img = plt.imread(temp_filename)
        fig, ax = plt.subplots(figsize=(16, 12))  # 表示サイズを調整
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"MinHashLSHiForest Tree in {name} {tree_index}", fontsize=20)
        plt.tight_layout()
        plt.show()
    finally:
        os.remove(temp_filename)
