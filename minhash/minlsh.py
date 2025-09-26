from __future__ import annotations  # Add this as the very first line
import numpy as np
import pandas as pd
import mmh3
from typing import Set, List
from collections import defaultdict
import math
import random


# --- Node and Tree Structure ---
class Node:
    """LSHツリーのノードを表すクラス"""

    def __init__(
        self, size: int, hash_index: int, children: dict[int, Node] | None = None
    ) -> None:
        self.size = size
        self.hash_index = hash_index  # 非圧縮パス長 (Uncompressed path length)
        self.children = children if children is not None else {}  # {hash_key: Node}


class MinHashLSHiForest:
    """
    Args
        n_estimators: 生成する木の本数
        eta: 実質pathかハッシュ回数にするかの割合を指定
    """

    def __init__(
        self, n_estimators: int = 100, sample_size: int = 256, eta: float = 1.0
    ) -> None:
        self.n_estimators = n_estimators
        self.sample_size = sample_size
        self.eta = eta
        self.forest_: list[dict[str, Node | int]] = []
        self.v_avg_ = 2  # 何これ？ -> 距離の補正に使っている？

    def fit(self, X: pd.Series) -> MinHashLSHiForest:
        """
        データセットXからLSHフォレストを構築する

        Args:
            X (list of set): トークンの集合のリスト。
                         各トークンは "キー:値" 形式の文字列であることが期待される
                         例: [{'country:JP', 'device:mobile'}, {'country:US', 'device:pc'}]
        """
        self.forest_ = []
        n_samples = len(X)
        if n_samples == 0:  # 0になることある？
            return self

        total_branching_factor = 0
        total_internal_nodes = 0

        # 多分サブサンプリングしてる？
        for _ in range(self.n_estimators):
            sample_size = self.sample_size
            if sample_size == 0:
                continue  # 0の場合はこのループをスキップ おそらく0になることはない

            indices = random.sample(
                range(len(X)), sample_size
            )  # sample_sizeまでのデータからしかサンプルしてる問題を修正
            subsample = [X.iloc[i] for i in indices]

            height_limit = self._calculate_H(
                sample_size, 2
            )  # 高さ制限を決定。　完全二分木の高さをこちょこちょしている？

            tree_root, tree_branching_info = self._build_lsh_tree(
                subsample, height_limit
            )  # ->_build_lsh_tree
            self.forest_.append({"root": tree_root, "sample_size": sample_size})

            total_branching_factor += tree_branching_info["total_v"]
            total_internal_nodes += tree_branching_info["node_count"]

        if total_internal_nodes > 0:
            self.v_avg_ = total_branching_factor / total_internal_nodes

        return self

    def _build_lsh_tree(
        self, S: List[Set[str]], H: float, current_I: int = 0
    ) -> tuple[Node, dict[str, int]]:
        """LSHツリーを再帰的に構築する
        S:サンプルサイズ
        H:高さ制限
        current_I:現在のIteration
        """
        # 終了条件のチェック
        if not S or len(S) <= 1 or current_I >= H:
            return Node(size=len(S), hash_index=current_I), {
                "total_v": 0,
                "node_count": 0,
            }

        while current_I <= H:
            partitions = defaultdict(
                list
            )  # key error を発生させないdict, keyerrorの場合、空のlist[]を返す
            for x in S:  # 全てのxに対して
                min_hash_val = min(
                    (mmh3.hash(token, seed=current_I, signed=False) for token in x),
                    default=0,
                )  # seed値に偏りあり, unsigned integer を返す、処理が重すぎる
                partitions[min_hash_val].append(x)

            if len(partitions) > 1:
                break
            current_I += 1

        # double check for termination condition
        if current_I > H:
            return Node(size=len(S), hash_index=current_I - 1), {
                "total_v": 0,
                "node_count": 0,
            }

        children = {}
        total_v_subtree = len(partitions)
        node_count_subtree = 1

        # generate child nodes
        for key, subset in partitions.items():
            child_node, child_info = self._build_lsh_tree(
                subset, H, current_I + 1
            )  # -> _build_lsh_tree
            children[key] = child_node  # dict of key:child_node
            total_v_subtree += child_info["total_v"]  #
            node_count_subtree += child_info["node_count"]

        branching_info = {"total_v": total_v_subtree, "node_count": node_count_subtree}
        return (
            Node(size=len(S), hash_index=current_I, children=children),
            branching_info,
        )  # nodeと

    def decision_function(self, X: pd.Series) -> np.ndarray:
        """データセットXの各点の異常スコアを計算する"""
        anomaly_scores = np.zeros(len(X))

        for i, x in enumerate(X):
            total_scaled_path_length = 0
            for tree_info in self.forest_:
                if not tree_info["root"]:
                    continue

                path_len = self._path_length(x, tree_info["root"], 0)  # -> _path_length

                mu_val = self._calculate_mu(tree_info["sample_size"], self.v_avg_)
                normalized_path = path_len / mu_val if mu_val > 0 else path_len

                total_scaled_path_length += 2 ** (-normalized_path)

            anomaly_scores[i] = (
                total_scaled_path_length / self.n_estimators
                if self.n_estimators > 0
                else 0
            )

        return anomaly_scores

    def _path_length(self, x: Set[str], node: Node, current_hc: int) -> float:
        """データxのツリー内でのパス長を再帰的に計算する"""
        if not node.children:  # x is in end node
            hu = node.hash_index  # times of hashing
            hc = current_hc  # actual path length
            adjustment = self._calculate_mu(
                node.size, self.v_avg_
            )  # depends on node.size and v_avg?
            final_path = self._calculate_path(hc, hu, adjustment)
            return final_path

        # calc child
        min_hash_val = min(
            (mmh3.hash(token, seed=node.hash_index, signed=False) for token in x),
            default=0,
        )

        if min_hash_val in node.children:  # x in childen nodes
            return self._path_length(x, node.children[min_hash_val], current_hc + 1)
        else:  # x is out of childen node, we assume that child of x is isolated.
            hu = node.hash_index + 1
            hc = current_hc + 1
            adjustment = self._calculate_mu(node.size, self.v_avg_)
            final_path = self._calculate_path(hc, hu, adjustment)
            return final_path

    def _calculate_H(self, sample_size: int, v: float) -> float:
        """高さの上限 H(ψ) を計算 (Formula 1)"""
        if sample_size <= 1:
            return 0
        return 2 * math.log2(sample_size) + 0.8327

    def _calculate_mu(self, size: int, v: float) -> float:
        """パス長の正規化係数 μ(ψ) を計算 (Formula 2)"""
        if size <= 1:
            return 0
        if size <= v:
            return 1.0
        euler_gamma = 0.5772156649
        log_v = math.log(v) if v > 1 else 1.0
        return (math.log(size) + math.log(v - 1) + euler_gamma) / log_v - 0.5

    def _calculate_path(self, hc: int, hu: int, adjustment: float) -> float:
        """h(x)を計算"""
        return (hc * (hu / hc) ** self.eta if hc > 0 else hu) + adjustment
