"""讲义：第 6 章 A/B 判定规则与多重比较（notes/lessons/06-ab-rules-multiple-comparisons.md）
运行：uv run python ch06_ab_rules.py

例子：偷看的模拟（两个策略其实一样，都是 0.7）；编的 6 个 p 值做 Holm；编的任务分数算 IQM；Miller 式 (4) 的聚类 SE。
里程碑 ME-M5 的"20 个任务"模拟留给自己做，这里用 10 个。
"""

import math

import numpy as np
from scipy.stats import norm

P_VALUES = [0.004, 0.009, 0.012, 0.040, 0.130, 0.470]  # 编的 6 个比较
TASK_SCORES = np.array([0.02, 0.10, 0.35, 0.41, 0.48, 0.52, 0.55, 0.61, 0.70, 0.98])  # 编的 10 个任务分数


def family_error(k: int, alpha: float = 0.05) -> float:
    """k 个独立检验、每个都在 α 水平上做，至少误报一个的概率 1 − (1 − α)^k。"""
    return 1 - (1 - alpha) ** k


def bonferroni(pvals, alpha: float = 0.05) -> list[bool]:
    """Bonferroni：每个检验用 α / m。"""
    m = len(pvals)
    return [p < alpha / m for p in pvals]


def holm(pvals, alpha: float = 0.05) -> list[bool]:
    """Holm：p 值从小到大，第 i 个（从 0 数）和 α / (m − i) 比，一旦不过就停。"""
    m = len(pvals)
    order = np.argsort(pvals)
    reject = [False] * m
    for i, j in enumerate(order):
        if pvals[j] < alpha / (m - i):
            reject[j] = True
        else:
            break
    return reject


def peeking_false_positive(p: float = 0.7, n_max: int = 100, every: int = 10, reps: int = 20000, seed: int = 0) -> tuple[float, float]:
    """两个策略其实一样（都是 p）。每跑 every 次就做一次 z 检验，一旦 p < 0.05 就宣布"B 更好"。返回 (偷看时的误报率, 只在最后看一次的误报率)。"""
    rng = np.random.default_rng(seed)
    a = rng.random((reps, n_max)) < p
    b = rng.random((reps, n_max)) < p
    hit_any = np.zeros(reps, bool)
    hit_last = np.zeros(reps, bool)
    for n in range(every, n_max + 1, every):
        ka, kb = a[:, :n].sum(1), b[:, :n].sum(1)
        pbar = (ka + kb) / (2 * n)
        se = np.sqrt(pbar * (1 - pbar) * 2 / n)
        z = np.divide(kb - ka, n * se, out=np.zeros(reps), where=se > 0)
        sig = np.abs(z) > 1.959964
        hit_any |= sig
        if n == n_max:
            hit_last = sig
    return float(hit_any.mean()), float(hit_last.mean())


def null_tasks_significant(k: int = 10, n: int = 50, p: float = 0.6, reps: int = 5000, seed: int = 1) -> float:
    """k 个任务、两个模型其实一样：平均每轮有几个任务"显著"（不校正）。"""
    rng = np.random.default_rng(seed)
    ka, kb = rng.binomial(n, p, (reps, k)), rng.binomial(n, p, (reps, k))
    pbar = (ka + kb) / (2 * n)
    se = np.sqrt(pbar * (1 - pbar) * 2 / n)
    z = np.divide(kb - ka, n * se, out=np.zeros((reps, k)), where=se > 0)
    return float((np.abs(z) > 1.959964).sum(1).mean())


def iqm(x: np.ndarray) -> float:
    """四分位间均值：排序后去掉最低 25% 和最高 25%，剩下的取平均。"""
    x = np.sort(np.asarray(x))
    n = len(x)
    lo, hi = int(math.floor(n * 0.25)), int(math.ceil(n * 0.75))
    return float(x[lo:hi].mean())


def clustered_se(scores_by_cluster: list[list[float]]) -> tuple[float, float]:
    """Miller 2024 式 (4)：SE_clustered = √(SE_CLT² + (1/n²) Σ_c Σ_i Σ_{j≠i} (s_i − s̄)(s_j − s̄))。返回 (普通 SE, 聚类 SE)。"""
    allv = np.concatenate([np.asarray(c, float) for c in scores_by_cluster])
    n, sbar = len(allv), allv.mean()
    se_clt = math.sqrt(allv.var(ddof=0) / n)
    cross = 0.0
    for c in scores_by_cluster:
        d = np.asarray(c, float) - sbar
        cross += d.sum() ** 2 - (d**2).sum()
    return se_clt, math.sqrt(se_clt**2 + cross / n**2)


def main() -> None:
    print("\n第一步 偷看：两个策略一样（都是 0.7），每 10 次看一眼，最多 100 次")
    peek, last = peeking_false_positive()
    print(f"  每 10 次看一次、显著就停：误报率 {peek:.3f}；只在第 100 次看一次：{last:.3f}")

    print("\n第二步 同时比很多个：至少误报一个的概率")
    for k in (1, 5, 10, 20):
        print(f"  {k:2d} 个检验：{family_error(k):.3f}")
    print(f"  模拟：10 个任务、两模型一样、每任务 50 次，平均 {null_tasks_significant():.2f} 个任务\"显著\"")

    print("\n第三步 Bonferroni 和 Holm（6 个 p 值）")
    print(f"  p 值：{P_VALUES}")
    print(f"  不校正：{[p < 0.05 for p in P_VALUES]}")
    print(f"  Bonferroni（α/6 = {0.05 / 6:.4f}）：{bonferroni(P_VALUES)}")
    print(f"  Holm：{holm(P_VALUES)}")

    print("\n第四步 多任务汇总：均值、中位数、IQM")
    print(f"  10 个任务：均值 {TASK_SCORES.mean():.3f}，中位数 {np.median(TASK_SCORES):.3f}，IQM {iqm(TASK_SCORES):.3f}")

    print("\n第五步 题目成组出现：聚类标准误（Miller 式 4）")
    clusters = [[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 0], [0, 0, 1, 0], [1, 1, 1, 1]]
    a, b = clustered_se(clusters)
    print(f"  5 篇文章各 4 道题：普通 SE {a:.4f}，聚类 SE {b:.4f}（{b / a:.2f} 倍）")


if __name__ == "__main__":
    main()
