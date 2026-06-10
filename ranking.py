"""
Bradley-Terry モデルによる好みランキング推定。

ペア比較ログ (勝者, 敗者) のリストから、各顔の潜在的な「好み度（強さ）」を
推定する。外部依存なし（numpy のみ）。

Bradley-Terry モデル:
    P(i が j に勝つ) = p_i / (p_i + p_j)
強さ p_i を最尤推定する。ここでは収束が安定している MM(minorization-
maximization) アルゴリズムを使う。比較が少ない/未対戦の顔でも発散しない
ように、各顔に「仮想的な引き分けを 1 回ずつ与える」弱い事前分布を入れる。
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def estimate_strengths(
    comparisons: list[tuple[str, str]],
    items: list[str] | None = None,
    prior: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-9,
) -> dict[str, float]:
    """
    ペア比較から各アイテムの強さ(好み度)を推定する。

    Args:
        comparisons: (winner_id, loser_id) のリスト。
        items: 対象とする全アイテムID。省略時は comparisons に出現したIDのみ。
        prior: 正則化のための仮想引き分け回数。0 にすると未対戦アイテムで
               発散しうる。1.0 程度を推奨。
        max_iter: MM 反復の最大回数。
        tol: 収束判定の閾値。

    Returns:
        {item_id: strength} の辞書。strength は正の値で、平均が 1 になるよう
        正規化される。比較が無いアイテムは 1.0（中立）付近になる。
    """
    if items is None:
        seen: set[str] = set()
        for w, l in comparisons:
            seen.add(w)
            seen.add(l)
        items = sorted(seen)

    n = len(items)
    if n == 0:
        return {}
    if n == 1:
        return {items[0]: 1.0}

    idx = {item: i for i, item in enumerate(items)}

    # wins[i]      : i の総勝利数（事前分布込み）
    # pair[(i,j)]  : i と j の対戦回数（i<j で集計）
    wins = np.zeros(n, dtype=float)
    pair_counts: dict[tuple[int, int], float] = defaultdict(float)

    for w, l in comparisons:
        if w not in idx or l not in idx:
            continue
        wi, li = idx[w], idx[l]
        if wi == li:
            continue
        wins[wi] += 1.0
        a, b = (wi, li) if wi < li else (li, wi)
        pair_counts[(a, b)] += 1.0

    # 弱い事前分布: 全ペアに prior 回の「引き分け」を加える。
    # 引き分け = 双方に prior/2 勝、対戦回数 prior を追加。
    if prior > 0:
        half = prior / 2.0
        wins += half * (n - 1)
        for i in range(n):
            for j in range(i + 1, n):
                pair_counts[(i, j)] += prior

    # 隣接リスト化（高速化）
    neighbors: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for (i, j), c in pair_counts.items():
        neighbors[i].append((j, c))
        neighbors[j].append((i, c))

    # MM 反復
    p = np.ones(n, dtype=float)
    for _ in range(max_iter):
        p_new = np.empty(n, dtype=float)
        for i in range(n):
            denom = 0.0
            for j, c in neighbors[i]:
                denom += c / (p[i] + p[j])
            # wins[i] は事前分布込みで必ず正
            p_new[i] = wins[i] / denom if denom > 0 else p[i]

        # 正規化（平均 1）して比較
        p_new *= n / p_new.sum()
        if np.max(np.abs(p_new - p)) < tol:
            p = p_new
            break
        p = p_new

    return {item: float(p[idx[item]]) for item in items}


def ranking(
    comparisons: list[tuple[str, str]],
    items: list[str] | None = None,
) -> list[tuple[str, float]]:
    """強さの降順にソートした (item_id, strength) のリストを返す。"""
    strengths = estimate_strengths(comparisons, items)
    return sorted(strengths.items(), key=lambda kv: kv[1], reverse=True)


def win_probability(strength_i: float, strength_j: float) -> float:
    """強さから i が j に勝つ確率を返す（UI 説明用）。"""
    return strength_i / (strength_i + strength_j)


if __name__ == "__main__":
    # 簡易動作確認: a > b > c > d を想定したログ
    demo = [
        ("a", "b"), ("a", "c"), ("a", "d"),
        ("b", "c"), ("b", "d"),
        ("c", "d"),
        ("a", "b"), ("b", "c"),
    ]
    for item, s in ranking(demo):
        print(f"{item}: {s:.4f}")
