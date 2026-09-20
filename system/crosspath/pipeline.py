"""第四章 CrossPath 的在线推理路径。

这一层不重写论文的算法：秩路径、边界追踪、效用估计、精确回退全部直接 import
仓库根目录的 ``weave_crosspath.py`` 与 ``weave_crosspath_gate.py``，也就是论文
实验用的同一份代码。系统只补三件工程上的事——

1. 把两个冻结端点的查询表示和图库表示凑成 2×2 兼容矩阵；
2. 把「基线 / 对角分支 / 交叉分支」三条打分拼成 17 个候选排序动作；
3. 在 K=1、5、10 三个截断上取边界候选、算效用、选出唯一的最终排序。

动作编号与论文表一致：0 号是精确回退（基线 S00 自身），1–8 是对角分支的
α=1/8…1，9–16 是交叉分支的 α=1/8…1。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from weave_crosspath import (  # noqa: E402
    DEFAULT_ALPHAS,
    boundary_trace,
    build_rank_path,
    estimate_utilities,
    select_action,
)
from weave_crosspath_gate import (  # noqa: E402
    CrossPathGate,
    aggregate_cutoff_utilities,
    build_candidate_features,
)

CUTOFFS = (1, 5, 10)
REGRESSION_COST = 2.0
NUM_ACTIONS = 2 * (len(DEFAULT_ALPHAS) - 1) + 1  # 17


@dataclass
class PathScores:
    """一次查询在 2×2 兼容矩阵上的四条路径得分。"""

    s00: np.ndarray
    s01: np.ndarray
    s10: np.ndarray
    s11: np.ndarray

    @property
    def diagonal(self) -> np.ndarray:
        """计算量匹配的对角集合 D = ½(S00 + S11)。"""
        return 0.5 * (self.s00 + self.s11)

    @property
    def cross(self) -> np.ndarray:
        """交叉均值 X = ½(S01 + S10)，零参数。"""
        return 0.5 * (self.s01 + self.s10)


def compatibility_matrix(q0, q1, g0, g1) -> PathScores:
    return PathScores(s00=q0 @ g0.T, s01=q0 @ g1.T, s10=q1 @ g0.T, s11=q1 @ g1.T)


def build_actions(candidate_ids: np.ndarray, paths: PathScores):
    """两条分支各 9 个 α，共用 α=0 的基线，拼成 17 个动作的排序表。"""
    diagonal = build_rank_path(candidate_ids, paths.s00, paths.diagonal)
    cross = build_rank_path(candidate_ids, paths.s00, paths.cross)
    orders = np.concatenate([diagonal.orders, cross.orders[1:]], axis=0)
    # 门控特征只吃两个端点的秩百分位，取基线和交叉分支两列。
    percentiles = np.stack(
        [diagonal.endpoint_percentiles[0], cross.endpoint_percentiles[1]], axis=1)
    return orders, percentiles, diagonal, cross


def membership_table(orders: np.ndarray, k: int):
    """返回在 K 截断上进出过前 K 的候选，以及它们在每个动作下的归属。"""
    n = orders.shape[1]
    table = np.zeros((n, orders.shape[0]), dtype=np.int8)
    for action, order in enumerate(orders):
        table[order[:k], action] = 1
    crossing = np.flatnonzero(np.any(table != table[:, :1], axis=1))
    return crossing, table[crossing]


class CrossPathRanker:
    """把上面几块拼成一次完整的查询级重排序。"""

    def __init__(self, gallery_e0: np.ndarray, gallery_e1: np.ndarray,
                 item_ids: list[str], gate: CrossPathGate | None = None,
                 threshold: float = 0.0) -> None:
        self.g0 = gallery_e0.astype(np.float32)
        self.g1 = gallery_e1.astype(np.float32)
        self.item_ids = list(item_ids)
        self.candidate_ids = np.asarray(self.item_ids, dtype=str)
        self.gate = gate
        self.threshold = float(threshold)

    # ------------------------------------------------------------------ 检索
    def rank(self, q0: np.ndarray, q1: np.ndarray, top_k: int = 10,
             shortlist: int = 200) -> dict:
        """返回最终排序、选中的动作，以及可以直接摆到界面上的中间量。"""
        paths = compatibility_matrix(q0[None, :], q1[None, :], self.g0, self.g1)
        paths = PathScores(paths.s00[0], paths.s01[0], paths.s10[0], paths.s11[0])

        # 只在基线前 shortlist 名内部重排，剩下的候选保持基线次序。
        base_order = np.argsort(-paths.s00, kind="stable")
        head = base_order[:shortlist]
        tail = base_order[shortlist:]
        sub = PathScores(paths.s00[head], paths.s01[head],
                         paths.s10[head], paths.s11[head])
        sub_ids = self.candidate_ids[head]

        orders, percentiles, diagonal, cross = build_actions(sub_ids, sub)
        utilities = np.zeros(orders.shape[0], dtype=np.float64)
        boundary_sizes = []
        if self.gate is not None:
            per_cutoff = []
            for k in CUTOFFS:
                crossing, table = membership_table(orders, k)
                boundary_sizes.append(int(crossing.size))
                if crossing.size == 0:
                    per_cutoff.append(np.zeros(orders.shape[0]))
                    continue
                features = build_candidate_features(
                    q0, self.g0[head][crossing], percentiles[crossing], table, k)
                with torch.no_grad():
                    probability = self.gate.probabilities(
                        torch.from_numpy(features)).numpy()
                per_cutoff.append(
                    estimate_utilities(table, probability, REGRESSION_COST))
            utilities = aggregate_cutoff_utilities(np.stack(per_cutoff, axis=0))

        action, order = select_action(utilities, self.threshold, orders)
        ranked_head = head[order]
        ranked = np.concatenate([ranked_head, tail])

        branch = "基线回退" if action == 0 else (
            "对角分支" if action < len(DEFAULT_ALPHAS) else "交叉分支")
        alpha = 0.0 if action == 0 else float(
            DEFAULT_ALPHAS[action] if action < len(DEFAULT_ALPHAS)
            else DEFAULT_ALPHAS[action - len(DEFAULT_ALPHAS) + 1])

        base_ids = [self.item_ids[i] for i in base_order[:top_k]]
        final_ids = [self.item_ids[i] for i in ranked[:top_k]]
        return {
            "final_ids": final_ids,
            "base_ids": base_ids,
            "final_scores": [float(paths.s00[i]) for i in ranked[:top_k]],
            "base_scores": [float(paths.s00[i]) for i in base_order[:top_k]],
            "action": int(action),
            "branch": branch,
            "alpha": alpha,
            "utilities": utilities.tolist(),
            "boundary_sizes": boundary_sizes,
            "path_recall": {
                "S00": [self.item_ids[i] for i in np.argsort(-paths.s00)[:top_k]],
                "S01": [self.item_ids[i] for i in np.argsort(-paths.s01)[:top_k]],
                "S10": [self.item_ids[i] for i in np.argsort(-paths.s10)[:top_k]],
                "S11": [self.item_ids[i] for i in np.argsort(-paths.s11)[:top_k]],
                "cross": [self.item_ids[i] for i in np.argsort(-paths.cross)[:top_k]],
                "diagonal": [self.item_ids[i] for i in np.argsort(-paths.diagonal)[:top_k]],
            },
        }
